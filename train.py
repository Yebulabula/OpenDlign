import argparse
import torch
import os 
import json
import torchvision.transforms as transforms
from dataset import AlignDataset, ZeroShotDataset
from losses import ContrastiveLoss 
from torch.utils.data import DataLoader
from utils import utils
import numpy as np
from model import OpenDlign
from configs import get_cfg_default
import random
import torch.nn as nn
import open_clip
import logging
import datetime
from zero_shot import zero_shot_eval, generate_eval_depth_files

def resume_checkpoint(model, optimizer_d_visual, scheduler_d_visual, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'], strict=True) 
    
    optimizer_d_visual.load_state_dict(checkpoint['optimizer_d_visual_state_dict'])
    scheduler_d_visual.load_state_dict(checkpoint['scheduler_d_visual_state_dict'])
    return model, optimizer_d_visual, scheduler_d_visual

def train(model, contrastive_criterion, optimizer_d_visual, scheduler_d_visual, train_loader, test_loader, label_dict, cfg):
    update_freq = 3
    epochs = 10
    feature_loss = nn.MSELoss()

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
            outputs = model(rgb, depth)
            
            feature_loss_val = feature_loss(outputs['feature_A'], outputs['feature_B'])
            loss_dict = contrastive_criterion(outputs)

            optimizer_d_visual.zero_grad()
            
            # contrastive loss + feature distance loss
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
                logging.info(f"iter {i}: CV_loss: {round(ct_loss / update_freq, 2)}, feat_dist_loss: {round(total_feature_loss / update_freq, 2)}, depth2rgb_acc: {round(A2B_acc / update_freq, 2)}, rgb2depth_acc: {round(B2A_acc / update_freq, 2)}")
                ct_loss, A2B_acc, B2A_acc, total_feature_loss = 0, 0, 0, 0

        optimizer_d_visual_state = optimizer_d_visual.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_d_visual_state_dict': optimizer_d_visual_state,
            'scheduler_d_visual_state_dict': scheduler_d_visual.state_dict(),
        }

        zero_shot_eval(model, test_loader, label_dict, cfg)
        torch.save(checkpoint, os.path.join(cfg.output_checkpoint_dir, f"checkpoint_{epoch}.pth"))

    logging.close()

def generate_rgb_depth_files(cfg):
    root = cfg.root
    dataset_list = cfg.train_dataset

    depth_files = []
    RGB_files = []
    for dataset in dataset_list:
        RGB_path = os.path.join(root, dataset, "depth_align_img")
        depth_path = os.path.join(root, dataset, "depth_map")
        
        filenames = os.listdir(RGB_path)

        for i in range(len(filenames)):
            depth_files.append(os.path.join(depth_path, filenames[i] + "_dm.npy"))
            RGB_files.append(os.path.join(RGB_path, filenames[i]))

    return RGB_files, depth_files

    
def main(cfg):
    seed = cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logging.info("loading pretrained model")
    clip_model, _, _ = open_clip.create_model_and_transforms(cfg.clip_model, pretrained='dfn5b')
    logging.info("loading tokenizer")
    tokenizer = open_clip.get_tokenizer(cfg.clip_model)
    
    # =================== Depth map and depth-aligned Datalodaer Definition =================== #
    RGB_files, depth_files = generate_rgb_depth_files(cfg)
    
    test_depth_files, test_labels = generate_eval_depth_files(cfg)
    
    random.seed(seed)
    random.shuffle(RGB_files)

    random.seed(seed)
    random.shuffle(depth_files)

    RGB_files = RGB_files
    depth_files = depth_files

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    train_transform = test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    with open('labels.json') as f:
        classnames = json.load(f)[cfg.eval_dataset]
    label_dict = {i: category for i, category in enumerate(classnames)}
    
    train_dataset = AlignDataset(RGB_files, depth_files, transform=train_transform, tokenizer = tokenizer)
    test_dataset = ZeroShotDataset(test_depth_files, test_labels, transform=test_transform, classnames = classnames)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE, shuffle=True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE, shuffle=False, num_workers = 4)
    
    # =================== OpenDlign model Definition =================== #
    model = OpenDlign(clip_model, tokenizer)
    model.cuda(cfg.gpu)
    
    for name, param in model.named_parameters():
        param.requires_grad_('depth_control_visual.resblocks.0.att' in name)
        state_dict_visual = model.clip_model.visual.state_dict()
    
    optimizer_d_visual = torch.optim.AdamW(model.clip_model.visual.parameters(), lr=3e-5, betas = cfg.betas, eps = cfg.eps, weight_decay=cfg.wd)
    scheduler_d_visual = torch.optim.lr_scheduler.OneCycleLR(optimizer_d_visual, max_lr=3e-4, steps_per_epoch = len(train_dataset), epochs=10)
    
    if cfg.resume:
        model, optimizer_d_visual, scheduler_d_visual = resume_checkpoint(model, optimizer_d_visual, scheduler_d_visual, cfg.resume_path)
    else:
        last_layer_num = cfg.vision_cfg.layers
        for key in list(state_dict_visual.keys()):  
            if f'resblocks.{last_layer_num}' in key:
                new_key = key.replace(f'resblocks.{last_layer_num}', 'depth_control_visual.resblocks.0')
                state_dict_visual[new_key] = state_dict_visual[key].clone()
        model.clip_model.visual.load_state_dict(state_dict_visual)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameters:", num_params)
    
    # =================== Loss Definition =================== #
    contrastive_criterion = ContrastiveLoss()
    
    train(model, contrastive_criterion, optimizer_d_visual, scheduler_d_visual, train_loader, test_loader, label_dict, cfg)

def reset_config(cfg, args):
    if args.root is not None:
        cfg.root = args.root
    if args.seed is not None:
        cfg.seed = args.seed
    if args.clip_model is not None:
        cfg.clip_model = args.clip_model
    if args.resume is not None:
        cfg.resume = args.resume
    if args.resume_path is not None:
        cfg.resume_path = args.resume_path
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir
    if args.output_checkpoint_dir is not None:
        cfg.output_checkpoint_dir = args.output_checkpoint_dir
    if args.gpu is not None:
        cfg.gpu = args.gpu
    if args.wd is not None:
        cfg.wd = args.wd
    if args.betas is not None:
        cfg.betas = args.betas
    if args.eps is not None:
        cfg.eps = args.eps
    if args.eval_dataset is not None:
        cfg.eval_dataset = args.eval_dataset
    if args.train_dataset is not None:
        cfg.train_dataset = args.train_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/mnt/new_drive/YeMao_Dataset/point_cloud_dataset/", help="path to dataset")
    parser.add_argument(
        "--train_dataset", type=list, default=['shapenet'], help="shapenet-55"
    )
    parser.add_argument(
        "--resume", type=bool, default=False, help="resume training"
    )
    parser.add_argument(
        "--resume_path", type=str, default="checkpoints_b128_no_feature_loss", help="path to resume checkpoint"
    )
    parser.add_argument(
        "--output_checkpoint_dir", type=str, default="checkpoints_b128_no_feature_loss", help="path to save checkpoint"
    )
    parser.add_argument(
        "--clip_model", type=str, default="ViT-H-14-quickgelu", help="clip model name"
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
        "--log_dir", type=str, default="logging", help="path to print log"
    )
    parser.add_argument(
        "--config-file", type=str, default="model_configs/ViT-H-14-quickgelu.yaml", help="path to config file"
    )
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)

    args = parser.parse_args()
    
    cfg = get_cfg_default()
    reset_config(cfg, args)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # Get current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Include the timestamp in the log filename
    log_filename = f"{cfg.log_dir}/OpenDlign_Training_{current_time}.log"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    main(cfg)
    
    