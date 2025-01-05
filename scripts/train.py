import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import cv2
import sys
import glob
from math import sqrt
import logging
import shutil
from torch.cuda import amp  # For mixed precision training

from utils.gradflow_check import plot_grad_flow
from utils.EMA import EMA
from utils.build_config import build_config
from datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3 
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
from utils.flops import get_info

class GradientScaler:
    def __init__(self, init_scale=2.**16, growth_factor=2, backoff_factor=0.5,
                 growth_interval=2000, max_scale=2.**24):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_scale = max_scale
        self.step_count = 0
        self.growth_step = 0

def train_model(config):
    # Update config for HD resolution
    config['img_size'] = (1920, 1080)
    config['batch_size'] = max(1, config['batch_size'] // 4)  # Reduce batch size for HD
    
    # Save config
    source_file = config['config_path']
    destination_file = os.path.join(config['save_folder'], 'config.yaml')
    shutil.copyfile(source_file, destination_file)
    
    # Setup logging
    log_file = os.path.join(config['save_folder'], 'training.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # Create dataloader, model, criterion
    dataset = build_dataset(config, phase='train')
    dataloader = data.DataLoader(
        dataset, 
        config['batch_size'], 
        True, 
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    model = build_yowov3(config)
    get_info(config, model)
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    criterion = build_loss(model, config)

    # Optimizer setup with parameter groups
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == "bias":
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):
                g[1].append(p)
            else:
                g[0].append(p)

    optimizer = torch.optim.AdamW(g[0], lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    optimizer.add_param_group({"params": g[2], "weight_decay": 0.0})
    
    # Learning rate scheduler
    warmup_lr = LinearWarmup(config)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['max_epoch'],
        eta_min=config['lr'] * 0.01
    )

    # Mixed precision training setup
    scaler = amp.GradScaler()
    
    # Training setup
    adjustlr_schedule = config['adjustlr_schedule']
    acc_grad = config['acc_grad']
    max_epoch = config['max_epoch']
    lr_decay = config['lr_decay']
    save_folder = config['save_folder']
    
    torch.backends.cudnn.benchmark = True
    cur_epoch = 1
    loss_acc = 0.0
    ema = EMA(model)
    best_loss = float('inf')
    
    # Memory management
    torch.cuda.empty_cache()
    
    while cur_epoch <= max_epoch:
        cnt_pram_update = 0
        epoch_loss = 0.0
        
        for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader):
            batch_size = batch_clip.shape[0]
            batch_clip = batch_clip.to(device)
            
            for idx in range(batch_size):
                batch_bboxes[idx] = batch_bboxes[idx].to(device)
                batch_labels[idx] = batch_labels[idx].to(device)

            # Mixed precision training
            with amp.autocast():
                outputs = model(batch_clip)
                
                # Prepare targets
                targets = []
                for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                    nbox = bboxes.shape[0]
                    nclass = labels.shape[1]
                    target = torch.zeros(nbox, 5 + nclass, device=device)
                    target[:, 0] = i
                    target[:, 1:5] = bboxes
                    target[:, 5:] = labels
                    targets.append(target)

                targets = torch.cat(targets, dim=0)
                loss = criterion(outputs, targets) / acc_grad

            # Gradient scaling
            scaler.scale(loss).backward()
            loss_acc += loss.item()
            
            if (iteration + 1) % acc_grad == 0:
                cnt_pram_update += 1
                if cur_epoch == 1:
                    warmup_lr(optimizer, cnt_pram_update)
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                # Optimizer step with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

                # Logging
                log_msg = f"epoch: {cur_epoch}, update: {cnt_pram_update}, loss: {loss_acc:.4f}"
                print(log_msg, flush=True)
                logging.info(log_msg)
                
                with open(os.path.join(config['save_folder'], "logging.txt"), "a") as f:
                    f.write(f"{log_msg}\n")

                epoch_loss += loss_acc
                loss_acc = 0.0

                # Memory management
                if iteration % 10 == 0:
                    torch.cuda.empty_cache()

        # End of epoch processing
        avg_epoch_loss = epoch_loss / cnt_pram_update
        scheduler.step()
        
        # Save checkpoints
        save_path_ema = os.path.join(save_folder, f"ema_epoch_{cur_epoch}.pth")
        save_path = os.path.join(save_folder, f"epoch_{cur_epoch}.pth")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(save_folder, "best_model.pth")
            torch.save({
                'epoch': cur_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'scaler': scaler.state_dict(),
            }, best_path)
        
        # Regular saves
        torch.save(ema.ema.state_dict(), save_path_ema)
        torch.save(model.state_dict(), save_path)
        
        logging.info(f"Saved model at epoch: {cur_epoch}, avg_loss: {avg_epoch_loss:.4f}")
        
        cur_epoch += 1

if __name__ == "__main__":
    config = build_config()
    train_model(config)
