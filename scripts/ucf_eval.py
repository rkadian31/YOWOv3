import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import os
import cv2
import sys
from tqdm import tqdm
from math import sqrt

from datasets.build_dataset import build_dataset
from datasets.collate_fn import collate_fn
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from utils.box import non_max_suppression, box_iou
from evaluator.eval import compute_ap
from utils.flops import get_info

class HDEvaluator:
    def __init__(self, config):
        self.width = 1920
        self.height = 1080
        self.config = config
        self.iou_thresholds = torch.tensor([0.5]).cuda()  # Can be extended to [0.5, 0.55, 0.6, ..., 0.95]
        self.n_iou = self.iou_thresholds.numel()
        
    def scale_boxes(self, boxes, orig_shape, new_shape):
        """Scale boxes from orig_shape to new_shape"""
        ratio = min(new_shape[0] / orig_shape[0], new_shape[1] / orig_shape[1])
        scaled_boxes = boxes.clone()
        scaled_boxes[:, [0, 2]] *= ratio
        scaled_boxes[:, [1, 3]] *= ratio
        return scaled_boxes

@torch.no_grad()
def eval(config):
    # Initialize evaluator
    evaluator = HDEvaluator(config)
    
    # Setup data
    dataset = build_dataset(config, phase='test')
    batch_size = max(1, min(8, config.get('batch_size', 32)))  # Adjust batch size for HD
    
    dataloader = data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=min(6, os.cpu_count()),
        pin_memory=True
    )
    
    # Initialize model
    model = build_yowov3(config)
    get_info(config, model)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to("cuda")
    model.eval()

    # Initialize metrics
    metrics = []
    mean_precision = 0.
    mean_recall = 0.
    map50 = 0.
    mean_ap = 0.

    # Progress bar
    pbar = tqdm(dataloader, desc='Evaluating on HD dataset')

    for batch_idx, (batch_clip, batch_bboxes, batch_labels) in enumerate(pbar):
        try:
            # Move data to GPU
            batch_clip = batch_clip.to("cuda", non_blocking=True)
            
            # Prepare targets
            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                if bboxes.shape[0] > 0:  # Check if there are any boxes
                    target = torch.zeros(bboxes.shape[0], 6)
                    target[:, 0] = i  # Image index in batch
                    target[:, 1] = labels
                    target[:, 2:] = bboxes
                    targets.append(target)
            
            if not targets:  # Skip if no valid targets
                continue
                
            targets = torch.cat(targets, dim=0).to("cuda")

            # Run inference
            outputs = model(batch_clip)
            
            # Scale target boxes to HD resolution
            targets[:, 2:] *= torch.tensor((evaluator.width, evaluator.height, 
                                          evaluator.width, evaluator.height)).cuda()

            # Non-maximum suppression
            outputs = non_max_suppression(
                outputs,
                conf_thres=0.005,  # Low confidence threshold for evaluation
                iou_thres=0.5,
                multi_label=True
            )

            # Process each image in batch
            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], evaluator.n_iou, 
                                    dtype=torch.bool).cuda()

                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                    continue

                # Get detections
                detections = output.clone()

                if labels.shape[0]:
                    # Process ground truth boxes
                    tbox = labels[:, 1:5].clone()
                    
                    # Convert to boolean numpy array for speed
                    correct = np.zeros((detections.shape[0], evaluator.iou_thresholds.shape[0]))
                    correct = correct.astype(bool)

                    # Compute IoU
                    t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                    iou = box_iou(t_tensor[:, 1:], detections[:, :4])
                    
                    # Check for correct class predictions
                    correct_class = t_tensor[:, 0:1] == detections[:, 5]
                    
                    # Process each IoU threshold
                    for j in range(len(evaluator.iou_thresholds)):
                        matches = torch.nonzero(
                            (iou >= evaluator.iou_thresholds[j]) & correct_class,
                            as_tuple=False
                        )
                        
                        if matches.shape[0]:
                            matches = torch.cat((
                                matches,
                                iou[matches[:, 0], matches[:, 1]][:, None]
                            ), 1)
                            
                            matches = matches.cpu().numpy()
                            if matches.shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                            correct[matches[:, 1].astype(int), j] = True
                            
                    correct = torch.tensor(correct, dtype=torch.bool, device=evaluator.iou_thresholds.device)
                
                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

            # Update progress bar
            pbar.set_description(
                f'Processed batch {batch_idx + 1}/{len(dataloader)}'
            )

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue

    # Compute final metrics
    if metrics:
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
        if len(metrics) and metrics[0].any():
            tp, fp, mean_precision, mean_recall, map50, mean_ap = compute_ap(*metrics)

    # Print results
    print('\nEvaluation Results:')
    print(f'{"Metric":<20} {"Value":>10}')
    print('-' * 32)
    print(f'{"Precision":<20} {mean_precision:>10.3f}')
    print(f'{"Recall":<20} {mean_recall:>10.3f}')
    print(f'{"mAP@0.5":<20} {map50:>10.3f}')
    print(f'{"mAP@0.5:0.95":<20} {mean_ap:>10.3f}')
    print('\n')

    # Save results to file
    results_path = os.path.join(config['save_folder'], 'eval_results.txt')
    with open(results_path, 'w') as f:
        f.write(f'Precision: {mean_precision:.3f}\n')
        f.write(f'Recall: {mean_recall:.3f}\n')
        f.write(f'mAP@0.5: {map50:.3f}\n')
        f.write(f'mAP@0.5:0.95: {mean_ap:.3f}\n')

    return map50, mean_ap

if __name__ == "__main__":
    config = build_config()
    config['img_size'] = (1920, 1080)  # Set HD resolution
    eval(config)
