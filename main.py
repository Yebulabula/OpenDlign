import argparse
import math
import torch
import os 
import json
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataset import Depth_RGB
from losses import ContrastiveLoss 
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN
from utils import utils
import numpy as np
from model import OpenDlign
# from clip import clip
from configs import get_cfg_default
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import random
import torch.nn as nn
import open_clip
import logging

logging.basicConfig(filename='logging/train_ensemble_90k_3blocks.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def test(model, criterion, test_loader, cfg):
    model.eval()
    val_loss_visual = 0
    val_loss_text = 0
    avg_depth_rgb_acc = 0
    avg_rgb_depth_acc = 0
    avg_depth_text_acc = 0
    avg_text_depth_acc = 0
    with torch.no_grad():
        for i, (rgb, depth) in enumerate(test_loader):
            rgb = rgb.cuda(cfg.gpu, non_blocking=True)
            depth = depth.cuda(cfg.gpu, non_blocking=True)

            outputs = model(rgb, depth, eval_mode = True)

            loss_dict = criterion(outputs, eval_mode = True)
            loss_visual = loss_dict['loss_visual']
            loss_text = loss_dict['loss_text']
            val_loss_visual += loss_visual.item() 
            val_loss_text += loss_text.item()

            avg_depth_rgb_acc += loss_dict['ulip_depth_rgb_acc'].item()
            avg_rgb_depth_acc += loss_dict['ulip_rgb_depth_acc'].item()
            avg_depth_text_acc += loss_dict['ulip_depth_text_acc'].item()
            avg_text_depth_acc += loss_dict['ulip_text_depth_acc'].item()
            # logging.info("loss: ", loss.item(), "depth_rgb_acc: ", loss_dict['ulip_depth_rgb_acc'].item(), "rgb_depth_acc: ", loss_dict['ulip_rgb_depth_acc'].item())

    return val_loss_visual / len(test_loader), val_loss_text / len(test_loader), avg_depth_rgb_acc / len(test_loader), avg_rgb_depth_acc / len(test_loader), avg_depth_text_acc / len(test_loader), avg_text_depth_acc / len(test_loader)

def shuffle_after_second_comma(text):
    # Find the position of the second comma
    pos = [pos for pos, char in enumerate(text) if char == ','][1]  # Gets the index of the second comma

    # Split the text into two parts
    part1 = text[:pos+1]  # The part before and including the second comma
    part2 = text[pos+1:]  # The part after the second comma

    # Split the second part by comma, shuffle, and join back
    part2_parts = part2.split(',')
    random.shuffle(part2_parts)
    shuffled_part2 = ','.join(part2_parts)

    # Combine the two parts
    return part1 + shuffled_part2

def evaluate_train(model, train_loader, cfg):
    model.eval()
    total_correct = 0
    
    keep_files = []
    with torch.no_grad():
        for i, (rgb, depth, tokenized_caption, filename) in enumerate(train_loader):
            depth = depth.float().cuda(cfg.gpu, non_blocking=True)
            rgb = rgb.float().cuda(cfg.gpu, non_blocking=True)
            tokenized_caption = tokenized_caption.cuda(cfg.gpu, non_blocking=True)
            
            rgb_embeds = []
            text_embeds = []
            for j in range(rgb.shape[0]):
                rgb_embed = model.encode_image(rgb[j], state = '')
                rgb_embed = rgb_embed / rgb_embed.norm(dim=-1, keepdim=True)
                rgb_embed = rgb_embed.mean(dim=0)
                rgb_embed /= rgb_embed.norm()
                rgb_embeds.append(rgb_embed)
            
            for j in range(tokenized_caption.shape[0]):
                text_embed = model.clip_model.encode_text(tokenized_caption[j])
                text_embed /= text_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed.mean(dim=0)
                text_embed /= text_embed.norm()
                text_embeds.append(text_embed)

            rgb_embeds = torch.stack(rgb_embeds, dim=0)
            text_embeds = torch.stack(text_embeds, dim=0)
        
            logits_per_depth_rgb = rgb_embeds @ text_embeds.t()
            labels = torch.arange(0, rgb_embeds.shape[0]).cuda(cfg.gpu, non_blocking=True)
            
            for j in range(len(logits_per_depth_rgb)):
                top3_pred_idxs = logits_per_depth_rgb[j].topk(3, dim=-1)[1].tolist()
                class_idx = labels[j].item()
                if class_idx in top3_pred_idxs:
                    keep_files.append(filename[j])
            
            logging.info(len(keep_files))
            
    return keep_files

    
def evaluate(model, test_loader, label_dict, cfg):
    model.eval()
    top_1_acc = 0

    if cfg.eval_dataset == 'modelnet40':
        num_class = 40
    elif cfg.eval_dataset == 'OmniObject3D':
        num_class = 216
    else:
        num_class = 15
        
    # num_class = 40 if cfg.eval_dataset == 'modelnet40' else 15
    correct_each_class = dict.fromkeys(range(num_class), 0)
    fp_each_class = dict.fromkeys(range(num_class), 0)
    fn_each_class = dict.fromkeys(range(num_class), 0)
    tp_each_class = dict.fromkeys(range(num_class), 0)
    target_each_class = dict.fromkeys(range(num_class), 0)
    acc_each_class = dict.fromkeys(range(num_class), 0)

    top_5_acc = 0
    top_5_correct_each_class = dict.fromkeys(range(num_class), 0)
    top_5_acc_each_class = dict.fromkeys(range(num_class), 0)

    top_3_acc = 0
    top_3_correct_each_class = dict.fromkeys(range(num_class), 0)
    top_3_acc_each_class = dict.fromkeys(range(num_class), 0)
    
    logit_scale = model.clip_model.logit_scale
    with open(os.path.join("", 'templates.json')) as f:
        templates = json.load(f)[cfg.eval_dataset]

    with open(os.path.join("", 'labels.json')) as f:
        labels = json.load(f)[cfg.eval_dataset]
    
    tokenizer = model.tokenizer
    with torch.no_grad():
        text_features = []

        for i, l in enumerate(labels):    
            texts = [t.format(l, l) for t in templates]
            texts = torch.cat([tokenizer(p) for p in texts]).cuda(cfg.gpu, non_blocking=True)

            if len(texts.shape) < 2:    
                texts = texts[None, ...]
            
            # logging.info(texts.shape)
            # logging.info(texts.dtype)
            class_embeddings = model.clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            text_features.append(class_embeddings)

        text_features = torch.stack(text_features, dim=0)

        for i, (depth, ground_truth) in enumerate(test_loader):
            depth = depth.float().cuda(cfg.gpu, non_blocking=True)
            ground_truth = ground_truth.cuda(cfg.gpu, non_blocking=True)

            for j in range(depth.shape[1]):
                depth_embed = model.encode_image(depth[:,j], state = 'depth4visual') if j >= 5 else model.encode_image(depth[:,j], state = '')
                depth_embed = depth_embed / depth_embed.norm(dim=-1, keepdim=True)

                if j == 0:
                    logits_per_depth_rgb = F.softmax(depth_embed @ text_features.t(), dim = 1)
                else:   
                    if j >= 5:
                        logits_per_depth_rgb += F.softmax(depth_embed @ text_features.t(), dim = 1)
                    else:
                        logits_per_depth_rgb += F.softmax(depth_embed @ text_features.t(), dim = 1)
            
            pred = torch.argmax(logits_per_depth_rgb, dim=-1)
            
            # compute f1 score
            
            correct = pred.eq(ground_truth).sum()

            top3 = (logits_per_depth_rgb.topk(3, dim=-1)[1] == ground_truth.unsqueeze(-1)).any(dim=-1).sum()
            top5 = (logits_per_depth_rgb.topk(5, dim=-1)[1] == ground_truth.unsqueeze(-1)).any(dim=-1).sum()

            for i in range(len(pred)):
                class_idx = ground_truth[i].item()
                pred_idx = pred[i].item()

                if pred_idx != class_idx:
                    fn_each_class[class_idx] += 1
                    fp_each_class[pred_idx] += 1
                else:
                    tp_each_class[class_idx] += 1
                    
                correct_each_class[class_idx] += (pred_idx == class_idx)

                target_each_class[class_idx] += 1
                top_3_correct_each_class[class_idx] += (logits_per_depth_rgb[i].topk(3, dim=-1)[1] == ground_truth[i].unsqueeze(-1)).any(dim=-1).sum().item()
                top_5_correct_each_class[class_idx] += (logits_per_depth_rgb[i].topk(5, dim=-1)[1] == ground_truth[i].unsqueeze(-1)).any(dim=-1).sum().item()
            
            top_1_acc += correct
            top_3_acc += top3
            top_5_acc += top5
    
    logging.info(target_each_class)
    acc_each_class = [c / t * 100 for c, t in zip(correct_each_class.values(), target_each_class.values())]
    top_3_acc_each_class = [c / t * 100 for c, t in zip(top_3_correct_each_class.values(), target_each_class.values())]
    top_5_acc_each_class = [c / t * 100 for c, t in zip(top_5_correct_each_class.values(), target_each_class.values())]
    
    for i in range(num_class):
        logging.info(f"class: {label_dict[i]}, top(1)_acc: {acc_each_class[i]}, top(3)_acc: {top_3_acc_each_class[i]}, top(5)_acc: {top_5_acc_each_class[i]}")

    logging.info(f"top_1_acc: {top_1_acc / len(test_loader.dataset)}")
    logging.info(f"top_3_acc: {top_3_acc / len(test_loader.dataset)}")
    logging.info(f"top_5_acc: {top_5_acc / len(test_loader.dataset)}")

    f1_scores = []
    recalls = []
    precisions = []

    for class_idx in tp_each_class.keys():
        TP = tp_each_class[class_idx]
        FP = fp_each_class[class_idx]
        FN = fn_each_class[class_idx]

        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

        f1_scores.append(f1)
        recalls.append(recall)
        precisions.append(precision)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    macro_recall = sum(recalls) / len(recalls)
    macro_precision = sum(precisions) / len(precisions)

    logging.info(f"Macro F1-Score: {macro_f1}")
    logging.info(f"Macro Recall: {macro_recall}")
    logging.info(f"Macro Precision: {macro_precision}")   

def load_checkpoint(model, optimizer_d_visual, scheduler_d_visual, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'], strict=True) 
    
    optimizer_d_visual.load_state_dict(checkpoint['optimizer_d_visual_state_dict'])
    scheduler_d_visual.load_state_dict(checkpoint['scheduler_d_visual_state_dict'])
    return model, optimizer_d_visual, scheduler_d_visual

def train(model, contrastive_criterion, optimizer_d_visual, scheduler_d_visual, train_loader, test_loader,  label_dict, cfg):
    writer = SummaryWriter(log_dir='logs/+edge_enhancement_50000_B_256')  # You can specify a directory to save the logs
    
    for name, param in model.named_parameters():
        if 'depth_control_visual.resblocks.0.att' not in name:
            param.requires_grad_(False)
        else:
            logging.info(name)
            param.requires_grad_(True)

    state_dict_visual = model.clip_model.visual.state_dict()
    logging.info("loading pretrained weights")

    for key in state_dict_visual.keys():
        if 'resblocks.31' in key:
            # Create a new key for the destination block
            new_key = key.replace('resblocks.31', 'depth_control_visual.resblocks.0')
            # Copy the weight
            state_dict_visual[new_key] = state_dict_visual[key].clone()
        
    model.clip_model.visual.load_state_dict(state_dict_visual)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("number of trainable parameters: %d", num_params)
    
    update_freq = 3

    epochs = 10
    feature_loss = nn.MSELoss()
    
    train_state = 'rgb_depth'
    for epoch in range(epochs):
        model.train()

        ct_loss = 0
        A2B_acc = 0
        B2A_acc = 0
        total_feature_loss = 0
        
        logging.info(f"epoch {epoch}")

        for i, (rgb, depth) in enumerate(train_loader):
            rgb = rgb.cuda(cfg.gpu, non_blocking=True)
            depth = depth.cuda(cfg.gpu, non_blocking=True).float()
            outputs = model(rgb, depth, mode = train_state)
            
            feature_loss_val = feature_loss(outputs['feature_A'], outputs['feature_B'])
            loss_dict = contrastive_criterion(outputs)

            optimizer_d_visual.zero_grad()
            loss = loss_dict['loss_A_B'] + feature_loss_val
            loss.backward()
            optimizer_d_visual.step()
            scheduler_d_visual.step()

            A2B_acc += loss_dict['A_B_acc'].item()
            B2A_acc += loss_dict['B_A_acc'].item()
            ct_loss += loss_dict['loss_A_B'].item()
            total_feature_loss += feature_loss_val.item()
            
            if (i + 1) % update_freq == 0 and i != 0:
                # logging.info(type(contrastive_loss), type(g_loss), type(D_loss), type(total_depth_rgb_acc), type(total_rgb_depth_acc), type(total_depth_text_acc), type(total_text_depth_acc))
                logging.info(f"iter {i}: CV_loss: {round(ct_loss / update_freq, 2)}, feature_alignment_loss: {round(total_feature_loss / update_freq, 2)}, depth2rgb_acc: {round(A2B_acc / update_freq, 2)}, rgb2depth_acc: {round(B2A_acc / update_freq, 2)}")
                ct_loss, A2B_acc, B2A_acc, total_feature_loss = 0, 0, 0, 0

        optimizer_d_visual_state = optimizer_d_visual.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_d_visual_state_dict': optimizer_d_visual_state,
            'scheduler_d_visual_state_dict': scheduler_d_visual.state_dict(),
        }

        evaluate(model, test_loader, label_dict, cfg)
        torch.save(checkpoint, os.path.join("checkpoints_b128_obj90000_3blocks", f"checkpoint_{epoch}.pth"))
        if epoch == 9:
            evaluate(model, test_loader, label_dict, cfg)

    logging.close()
    writer.close()

def generate_rgb_depth_files(cfg):
    root = cfg.root
    dataset_list = cfg.train_dataset

    depth_files = []
    RGB_files = []
    for dataset in dataset_list:
        RGB_path = os.path.join(root, dataset,  "control_dataset")
        depth_path = os.path.join(root, dataset, "depth_map")

        file_path = os.path.join(root, dataset, 'keep_files_top3.txt')

        # # Initialize an empty list to store the lines
        rgb = []

        # Open the file for reading ('r')
        with open(file_path, 'r') as file:
            # Read each line in the file one by one
            for line in file:
                # Strip the newline character from the end of each line
                # and append it to the list
                rgb.append(line.strip().split('control_dataset/')[1])

        # rgb = os.listdir(RGB_path)
        for i in range(len(rgb)):
            depth_files.append(os.path.join(depth_path, rgb[i] + "_dm.npy"))
            RGB_files.append(os.path.join(RGB_path, rgb[i]))

    return RGB_files, depth_files

def generate_eval_rgb_depth_files(cfg):
    depth_path = os.path.join(cfg.root, cfg.eval_dataset, "color_depth_map")
    with open(os.path.join("", 'labels.json')) as f:
        category_RGB = json.load(f)[cfg.eval_dataset]

    category_RGB = [category_RGB[i].replace(" ", "_") for i in range(len(category_RGB))]
    depth_files = [os.path.join(depth_path, f) for f in os.listdir(depth_path)]
    RGB_files = [None] * len(depth_files)
    
    for i in range(len(depth_files)):
        RGB_files[i] = depth_files[i].split("_dm.npy")[0].split("+")[1]
    
    return RGB_files, depth_files, category_RGB

    
def main(cfg):
    seed = cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # RGB_path, depth_path, RGB_files, depth_files = generate_rgb_depth_files(cfg)

    # RGB_files, depth_files = generate_rgb_depth_files(cfg)

    test_RGB_files, test_depth_files, category_RGB_files = generate_eval_rgb_depth_files(cfg)
    # logging.info(len(test_depth_files))
    # breakpoint()    

    # seed = 42
    # random.seed(seed)
    # random.shuffle(RGB_files)

    # random.seed(seed)
    # random.shuffle(depth_files)

    # RGB_files = RGB_files
    # depth_files = depth_files


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    test_transform = transforms.Compose([
            normalize
        ])
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),              # Converts the image to a PyTorch tensor
        normalize                           # Normalizes the tensor
    ])
    
    logging.info("loading pretrained model")
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-quickgelu', pretrained='dfn5b')
    logging.info("loading tokenizer")
    tokenizer = open_clip.get_tokenizer('ViT-H-14-quickgelu')
    
    # train_dataset = Depth_RGB(RGB_files, depth_files, category_RGB_files = None, transform=train_transform, tokenizer = tokenizer, cfg=cfg)
    test_dataset = Depth_RGB(test_RGB_files, test_depth_files, category_RGB_files = category_RGB_files, transform=test_transform, mode='test', tokenizer = tokenizer, cfg=cfg)
    
    # logging.info(len(train_dataset), len(test_dataset))
    
    # train_loader = DataLoader(train_dataset, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False, num_workers = 4)
    

    model = OpenDlign(clip_model, tokenizer)
    model.cuda(cfg.gpu)
 
    optimizer_d_visual = torch.optim.AdamW(model.clip_model.visual.parameters(), lr=3e-5, betas = cfg.betas, eps = cfg.eps, weight_decay=cfg.wd)
    scheduler_d_visual = torch.optim.lr_scheduler.OneCycleLR(optimizer_d_visual, max_lr=3e-4, steps_per_epoch=100, epochs=10)

    contrastive_criterion = ContrastiveLoss()
    
    model, optimizer_d_visual, scheduler_d_visual  = load_checkpoint(model, optimizer_d_visual, scheduler_d_visual,  os.path.join(cfg.checkpoint_dir, "checkpoint_9.pth"))
    
    with open(os.path.join("", 'labels.json')) as f:
        labels = json.load(f)[cfg.eval_dataset]
        
    logging.info(len(labels))
    label_dict = dict.fromkeys(labels)
    for i, category in enumerate(labels):
        label_dict[i] = category

    # train(model, contrastive_criterion, optimizer_d_visual, scheduler_d_visual, train_loader, test_loader, label_dict, cfg)
    evaluate(model, test_loader, label_dict, cfg)
    

def reset_config(cfg, args):
    if args.root is not None:
        cfg.root = args.root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
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
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.eval_dataset is not None:
        cfg.eval_dataset = args.eval_dataset
    if args.train_dataset is not None:
        cfg.train_dataset = args.train_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/new_drive/YeMao_Dataset/point_cloud_dataset/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default= "checkpoints_b128_no_feature_loss",
    )
    parser.add_argument(
        "--train_dataset", type=list, default=[f"000-{i:03}" for i in range(101)], help="shapenet-55"
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="modelnet40", help="modelnet40"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu id"
    )
    parser.add_argument(
        "--config-file", type=str, default="/mnt/new_drive/Documents/point_open/model_configs/ViT-H-14-quickgelu.yaml", help="path to config file"
    )
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)

    args = parser.parse_args()

    cfg = get_cfg_default()
    reset_config(cfg, args)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    main(cfg)