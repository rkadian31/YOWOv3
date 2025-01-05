import torch 
import time
import torchvision
import cv2
import numpy as np

def make_anchors(x, strides, offset=0.5):
    """Generate anchors from features, optimized for HD resolution"""
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        # Adjust for HD aspect ratio
        sx = torch.arange(end=w, dtype=x[i].dtype, device=x[i].device) + offset
        sy = torch.arange(end=h, dtype=x[i].dtype, device=x[i].device) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=x[i].dtype, device=x[i].device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def wh2xy(x):
    """Convert width/height to x1/y1/x2/y2 format"""
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_threshold=0.25, iou_threshold=0.45):
    """Non-max suppression optimized for HD resolution"""
    nc = prediction.shape[1] - 4  # number of classes
    xc = prediction[:, 4:4 + nc].amax(1) > conf_threshold

    # Adjusted settings for HD
    max_wh = 1920 * 1080  # maximum box width and height for HD
    max_det = 500  # increased for HD resolution
    max_nms = 50000  # increased for HD resolution

    start = time.time()
    outputs = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    
    for index, x in enumerate(prediction):
        x = x.transpose(0, -1)[xc[index]]
        
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (box, conf, cls)
        box, cls = x.split((4, nc), 1)
        box = wh2xy(box)  # convert to x1y1x2y2
        
        if nc > 1:
            i, j = (cls > conf_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_threshold]

        if not x.shape[0]:
            continue
            
        # Sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_threshold)
        i = i[:max_det]
        outputs[index] = x[i]

        if (time.time() - start) > 1.0:  # increased timeout for HD
            print(f'WARNING: NMS timeout')
            break

    return outputs

def box_label(image, box, label=None, color=(0, 255, 0), txt_color=(0, 0, 0)):
    """Draw box and label, optimized for HD resolution"""
    # Adjusted scale factors for HD
    lw = max(round(sum(image.shape) / 2 * 0.002), 2)  # reduced line width for HD
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw-1, lineType=cv2.LINE_AA)
    
    if label is not None:
        tf = max(lw - 1, 1)  # font thickness
        # Adjusted font scale for HD
        font_scale = lw / 15  # reduced for HD
        w, h = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=tf)[0]
        
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        y0, dy = 0, 15  # increased spacing for HD
        y0 = p1[1] - 2 if outside else p1[1] + h + 2
        
        for i, line in enumerate(label.split('\n')):
            y = y0 + i*dy
            x = p1[0]
            text_size, _ = cv2.getTextSize(line, cv2.LINE_AA, font_scale, tf)
            text_width, text_height = text_size
            opacity(image, (x, y), (x + text_width + 1, y + text_height + 1))
            cv2.putText(image,
                    line, (x, y + text_height),
                    0,
                    font_scale,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

def draw_bounding_box(image, bboxes, labels, confs, map_labels):
    """Draw bounding boxes on HD images"""
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0)[:, :, (2, 1, 0)].contiguous()
        image = image.detach().cpu().numpy()
    
    H, W, C = image.shape
    
    # Scale factor adjustments for HD
    scale_x = W / 1920
    scale_y = H / 1080
    
    pre_box = []
    meta_data = []

    if bboxes is not None:
        for box, label, conf in zip(bboxes, labels, confs):
            box = box.clone().detach()
            # Scale boxes for HD
            box[0] *= scale_x
            box[1] *= scale_y
            box[2] *= scale_x
            box[3] *= scale_y
            
            text = f"{map_labels[int(label.item())]} : {round(conf.item()*100, 2)}"
            pre_box.append(box)
            meta_data.append([text, H, W])

        # Rest of the function remains the same but with HD-optimized IOU threshold
        res_box = []
        res_meta_data = []

        for idx1, box1 in enumerate(pre_box):
            flag = 1
            for idx2, box2 in enumerate(res_box):
                iou = box_iou(box1.unsqueeze(0), box2.unsqueeze(0))[0, 0]
                if (iou >= 0.85):  # slightly reduced threshold for HD
                    res_meta_data[idx2].append(meta_data[idx1])
                    flag = 0
                    break
            if flag:
                res_box.append(box1)
                res_meta_data.append([meta_data[idx1]])

        for box, meta in zip(res_box, res_meta_data):
            text = '\n'.join(m[0] for m in meta)
            bbox = [
                int(max(0, box[0])),
                int(max(0, box[1])),
                int(min(W, box[2])),
                int(min(H, box[3]))
            ]
            box_label(image, bbox, text)
