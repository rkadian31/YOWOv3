import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob
from math import sqrt

from datasets.build_dataset import build_dataset
from utils.box import draw_bounding_box
from utils.box import non_max_suppression
from model.TSN.YOWOv3 import build_yowov3 
from utils.build_config import build_config
from utils.flops import get_info

def preprocess_hd_image(image, target_size=(1920, 1080)):
    """Preprocess HD image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    # Calculate scaling factor while maintaining aspect ratio
    scale = min(target_size[0]/width, target_size[1]/height)
    new_size = (int(width * scale), int(height * scale))
    
    # Resize image maintaining aspect ratio
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    
    # Create canvas of target size
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    # Calculate padding
    y_offset = (target_size[1] - new_size[1]) // 2
    x_offset = (target_size[0] - new_size[0]) // 2
    
    # Place image on canvas
    canvas[y_offset:y_offset+new_size[1], 
           x_offset:x_offset+new_size[0]] = resized
    
    return canvas, (scale, x_offset, y_offset)

def postprocess_boxes(boxes, scale_params, original_size):
    """Adjust detection boxes back to original image coordinates"""
    scale, x_offset, y_offset = scale_params
    
    # Remove padding offset
    boxes[:, [0, 2]] -= x_offset
    boxes[:, [1, 3]] -= y_offset
    
    # Scale back to original size
    boxes[:, :4] /= scale
    
    # Clip to image boundaries
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_size[0])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_size[1])
    
    return boxes

def detect(config):
    # Set HD resolution
    config['img_size'] = (1920, 1080)  # Update config for HD
    
    #########################################################################
    dataset = build_dataset(config, phase='test')
    model = build_yowov3(config) 
    get_info(config, model)
    ##########################################################################
    mapping = config['idx2name']
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    with torch.no_grad():  # Disable gradient computation for inference
        for idx in range(dataset.__len__()):
            origin_image, clip, bboxes, labels = dataset.__getitem__(idx, get_origin_image=True)
            original_size = origin_image.shape[:2][::-1]  # (width, height)
            
            # Preprocess HD image
            processed_image, scale_params = preprocess_hd_image(origin_image)
            
            # Process clip
            clip = clip.unsqueeze(0).to(device)
            
            # Process in chunks if memory constrained
            chunk_size = 16  # Adjust based on available memory
            outputs_list = []
            
            for i in range(0, clip.size(2), chunk_size):
                chunk = clip[:, :, i:min(i+chunk_size, clip.size(2))]
                outputs_chunk = model(chunk)
                outputs_list.append(outputs_chunk)
            
            outputs = torch.cat(outputs_list, dim=1)
            
            # Non-max suppression with adjusted thresholds for HD
            outputs = non_max_suppression(
                outputs,
                conf_threshold=0.35,  # Slightly higher for HD
                iou_threshold=0.45,   # Adjusted for HD
                max_det=100           # Increased for HD
            )[0]
            
            if outputs is not None and len(outputs):
                # Postprocess boxes to original coordinates
                outputs[:, :4] = postprocess_boxes(
                    outputs[:, :4].clone(),
                    scale_params,
                    original_size
                )
                
                # Draw boxes on original resolution image
                draw_bounding_box(
                    origin_image,
                    outputs[:, :4],
                    outputs[:, 5],
                    outputs[:, 4],
                    mapping,
                    thickness=3  # Increased thickness for HD
                )

            # Display or save results
            flag = 1  # Set to 0 to save instead of display
            if flag:
                # Resize for display if needed
                display_size = (1280, 720)  # Adjust based on monitor
                display_image = cv2.resize(origin_image, display_size)
                cv2.imshow('HD Detection', display_image)
                k = cv2.waitKey(100)
                if k == ord('q'):
                    return
            else:
                save_path = os.path.join("detection_results", f"detection_{idx:04d}.jpg")
                os.makedirs("detection_results", exist_ok=True)
                cv2.imwrite(save_path, origin_image)
                print(f"Image {idx} saved to {save_path}")

if __name__ == "__main__":
    config = build_config()
    detect(config)
