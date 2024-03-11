import argparse
import os 
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataset import ZeroShotDataset
from torch.utils.data import DataLoader
from utils import utils
import numpy as np
from model import OpenDlign
from configs import get_cfg_default
import open_clip
import logging

def initialize_metrics(num_classes):
    metrics_names = [
        'correct_each_class', 'fp_each_class', 'fn_each_class', 'tp_each_class',
        'target_each_class', 'acc_each_class', 'top_5_correct_each_class', 
        'top_5_acc_each_class', 'top_3_correct_each_class', 'top_3_acc_each_class'
    ]
    return {metric: dict.fromkeys(range(num_classes), 0) for metric in metrics_names}

def get_num_classes(cfg):
    dataset_class_map = {
        'modelnet40': 40,
        'OmniObject3D': 216,
        'scanobjectnn': 15
    }
    return dataset_class_map.get(cfg.eval_dataset, 40)

def process_text_features(model, labels, templates, cfg):
    tokenizer = model.tokenizer 
    with torch.no_grad():
        text_features = []
        for label in labels:    
            texts = [template.format(label, label) for template in templates]
            texts = torch.cat([tokenizer(text) for text in texts]).cuda(cfg.gpu, non_blocking=True)
            if len(texts.shape) < 2:    
                texts = texts[None, ...]
            class_embeddings = model.clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            text_features.append(class_embeddings)
    return torch.stack(text_features, dim=0)

def update_metrics(metrics, logits_per_depth_rgb, ground_truth):
    pred = torch.argmax(logits_per_depth_rgb, dim=-1)
    correct = pred.eq(ground_truth).sum().item()
    top3 = (logits_per_depth_rgb.topk(3, dim=-1)[1] == ground_truth.unsqueeze(-1)).any(dim=-1).sum().item()
    top5 = (logits_per_depth_rgb.topk(5, dim=-1)[1] == ground_truth.unsqueeze(-1)).any(dim=-1).sum().item()
    
    for idx, (p, gt) in enumerate(zip(pred, ground_truth)):
        if p != gt:
            metrics['fn_each_class'][gt.item()] += 1
            metrics['fp_each_class'][p.item()] += 1
        else:
            metrics['tp_each_class'][gt.item()] += 1
        metrics['correct_each_class'][gt.item()] += int(p == gt)
        metrics['target_each_class'][gt.item()] += 1
        metrics['top_3_correct_each_class'][gt.item()] += (logits_per_depth_rgb[idx].topk(3, dim=-1)[1] == gt.unsqueeze(-1)).any(dim=-1).sum().item()
        metrics['top_5_correct_each_class'][gt.item()] += (logits_per_depth_rgb[idx].topk(5, dim=-1)[1] == gt.unsqueeze(-1)).any(dim=-1).sum().item()

    return correct, top3, top5

def calculate_class_metrics(metrics, num_class):
    acc_each_class = [
        metrics['correct_each_class'][i] / metrics['target_each_class'][i] * 100 if metrics['target_each_class'][i] > 0 else 0 
        for i in range(num_class)
    ]
    top_3_acc_each_class = [
        metrics['top_3_correct_each_class'][i] / metrics['target_each_class'][i] * 100 if metrics['target_each_class'][i] > 0 else 0 
        for i in range(num_class)
    ]
    top_5_acc_each_class = [
        metrics['top_5_correct_each_class'][i] / metrics['target_each_class'][i] * 100 if metrics['target_each_class'][i] > 0 else 0 
        for i in range(num_class)
    ]
    return acc_each_class, top_3_acc_each_class, top_5_acc_each_class

def calculate_macro_metrics(metrics, num_class):
    f1_scores = []
    recalls = []
    precisions = []
    for i in range(num_class):
        TP = metrics['tp_each_class'][i]
        FP = metrics['fp_each_class'][i]
        FN = metrics['fn_each_class'][i]
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_scores.append(f1)
        recalls.append(recall)
        precisions.append(precision)
    macro_f1 = sum(f1_scores) / num_class
    macro_recall = sum(recalls) / num_class
    macro_precision = sum(precisions) / num_class
    return macro_f1, macro_recall, macro_precision

def log_results(metrics, num_class, top_1_acc, top_3_acc, top_5_acc, dataset_size, label_dict):
    acc_each_class, top_3_acc_each_class, top_5_acc_each_class = calculate_class_metrics(metrics, num_class)
    macro_f1, macro_recall, macro_precision = calculate_macro_metrics(metrics, num_class)

    for i in range(num_class):
        logging.info(f"class: {label_dict[i]}, top(1)_acc: {acc_each_class[i]}, top(3)_acc: {top_3_acc_each_class[i]}, top(5)_acc: {top_5_acc_each_class[i]}")
    
    logging.info(f"top_1_acc: {top_1_acc / dataset_size}")
    logging.info(f"top_3_acc: {top_3_acc / dataset_size}")
    logging.info(f"top_5_acc: {top_5_acc / dataset_size}")
    logging.info(f"Macro F1-Score: {macro_f1}")
    logging.info(f"Macro Recall: {macro_recall}")
    logging.info(f"Macro Precision: {macro_precision}")
    
def zero_shot_eval(model, test_loader, label_dict, cfg):
    model.eval()
    num_class = get_num_classes(cfg)
    metrics = initialize_metrics(num_class)
    top_1_acc, top_5_acc, top_3_acc = 0, 0, 0

    with open(os.path.join("", 'templates.json')) as f:
        templates = json.load(f)[cfg.eval_dataset]

    with open(os.path.join("", 'labels.json')) as f:
        labels = json.load(f)[cfg.eval_dataset]

    text_features = process_text_features(model, labels, templates, cfg)

    with torch.no_grad():
        for depth, ground_truth in test_loader:
            depth = depth.float().cuda(cfg.gpu, non_blocking=True)
            ground_truth = ground_truth.cuda(cfg.gpu, non_blocking=True)

            logits_per_depth_rgb = 0
            for j in range(depth.shape[1]):
                state = 'depth_branch' if j >= 5 else 'rgb_branch'
                depth_embed = model.encode_image(depth[:, j], state=state)
                depth_embed = depth_embed / depth_embed.norm(dim=-1, keepdim=True)
                logits_per_depth_rgb += F.softmax(depth_embed @ text_features.t(), dim=1)

            correct, top3, top5 = update_metrics(metrics, logits_per_depth_rgb, ground_truth)
            top_1_acc += correct
            top_3_acc += top3
            top_5_acc += top5
    
    log_results(metrics, num_class, top_1_acc, top_3_acc, top_5_acc, len(test_loader.dataset), label_dict)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model

def generate_eval_depth_files(cfg):
    depth_path = os.path.join(cfg.root, cfg.eval_dataset, "depth_map")
    
    # Generate the list of depth files
    depth_files = [os.path.join(depth_path, f) for f in os.listdir(depth_path)]
    labels = [depth_file.split("_dm.npy")[0].split("+")[1] for depth_file in depth_files]
    
    return depth_files, labels
    
def main(cfg):
    seed = cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logging.info("loading pretrained model")
    
    if cfg.clip_model == 'ViT-H-14-quickgelu':
        pretrained_data = 'dfn5b'
    elif cfg.clip_model == 'ViT-L-14-quickgelu' or cfg.clip_model == 'ViT-B-16':
        pretrained_data = 'dfn2b'
    elif cfg.clip_model == 'ViT-B-32':
        pretrained_data = 'datacomp_xl_s13b_b90k'
    
    clip_model, _, _ = open_clip.create_model_and_transforms(cfg.clip_model, pretrained=pretrained_data)
    logging.info("loading tokenizer")
    tokenizer = open_clip.get_tokenizer(cfg.clip_model)
    
    test_depth_files, test_labels = generate_eval_depth_files(cfg)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    with open('labels.json') as f:
        classnames = json.load(f)[cfg.eval_dataset]
    label_dict = {i: category for i, category in enumerate(classnames)}
    
    test_dataset = ZeroShotDataset(test_depth_files, test_labels, transform=test_transform, classnames = classnames)
    test_loader = DataLoader(test_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False, num_workers = 4)

    model = OpenDlign(clip_model, tokenizer)
    model.cuda(cfg.gpu)
 
    model = load_checkpoint(model, os.path.join(cfg.checkpoint_path))
        
    zero_shot_eval(model, test_loader, label_dict, cfg)

def reset_config(cfg, args):
    if args.root is not None:
        cfg.root = args.root
    if args.clip_model is not None:
        cfg.clip_model = args.clip_model
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.gpu is not None:
        cfg.gpu = args.gpu
    if args.wd is not None:
        cfg.wd = args.wd
    if args.betas is not None:
        cfg.betas = args.betas
    if args.eps is not None:
        cfg.eps = args.eps
    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path
    if args.eval_dataset is not None:
        cfg.eval_dataset = args.eval_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/new_drive/YeMao_Dataset/point_cloud_dataset/", help="path to dataset")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default= "checkpoints_b128_no_feature_loss",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="modelnet40", help="modelnet40"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--clip_model", type=str, default='ViT-H-14-quickgelu', help="clip model name"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu id"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logging", help="path to print log"
    )
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)

    args = parser.parse_args()

    cfg = get_cfg_default()
    reset_config(cfg, args)
    cfg.merge_from_file(f'model_configs/{args.clip_model}.yaml')
    cfg.freeze()
    logging.basicConfig(filename=f'{cfg.log_dir}/OpenDlign_Zero_Shot_Eval.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
    main(cfg)
    
    