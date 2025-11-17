import os, cv2
import pickle

import torch
import numpy as np
from PIL import Image, ImageDraw

def calc_iou(p, t):
    mul = (p*t).sum()
    add = (p+t).sum()
    return mul / (add - mul + 1e-6)

def debugDump(output_dir, image_name, texts, lsts, size=(256, 256), data=None):
    """
    Args:
        texts: list of list of text
        lsts: list of list of torch.Tensor H, W
    """
    out_dir = os.path.join(output_dir, "debug")
    os.makedirs(out_dir, exist_ok=True)
    outs = []
    for txts, lst in zip(texts, lsts):
        lst = [cv2.resize((x.numpy()*255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR) for x in lst]
        lst = [Image.fromarray(x) for x in lst]
        for x, t in zip(lst, txts):
            ImageDraw.Draw(x).text((0, 0), str(t), fill="red")
        out = Image.fromarray(np.concatenate([np.array(x) for x in lst], axis=1))
        outs.append(np.array(out))
    out = Image.fromarray(np.concatenate(outs, axis=0))
    out.save(os.path.join(out_dir, image_name+".png"))

    if not isinstance(data, type(None)):
        try:
            with open(os.path.join(out_dir, "latest.pk"), "wb") as f:
                pickle.dump(data, f)
        except:
            pass


def pad1d(x, dim, num, value=0.0):
    """
    Args:
        pad a torch.Tensor along dim (at the end) to be dim=num
        x: any shape torch.Tensor
        dim: int
        repeats: int

    Returns:
        x: where x.shape[dim] = num
    """
    size = list(x.shape)
    size[dim] = num - size[dim]
    assert size[dim] >= 0, "{} < 0".format(size[dim])
    v = torch.ones(size, dtype=x.dtype, device=x.device) * value
    return torch.cat([x, v], dim=dim)

def mask2Boxes(masks):
    """
    将masks转换为bounding boxes，处理空mask的情况
    
    Args:
        masks: n, H, W tensor

    Returns:
        bbox: n, 4 [(x1,y1),(x2,y2)] \in [0,1]
    """
    n, H, W = masks.shape
    device = masks.device
    
    if n == 0:
        return torch.zeros((0, 4), device=device, dtype=masks.dtype)
    
    bboxes = []
    
    for i in range(n):
        mask = masks[i]
        
        # 检查mask是否为空
        mask_sum = mask.sum()
        
        if mask_sum < 1.0:  # mask为空或几乎为空
            # 使用默认bbox（图像中心的小框）
            bbox = torch.tensor([0.45, 0.45, 0.55, 0.55], device=device, dtype=mask.dtype)
            bboxes.append(bbox)
            continue
        
        # 获取非零位置
        try:
            # 使用阈值获取前景像素
            foreground = mask > 0.5
            
            if not foreground.any():
                # 如果阈值后没有前景，使用默认bbox
                bbox = torch.tensor([0.45, 0.45, 0.55, 0.55], device=device, dtype=mask.dtype)
                bboxes.append(bbox)
                continue
            
            # 获取y和x坐标
            y_indices, x_indices = torch.where(foreground)
            
            if len(x_indices) == 0 or len(y_indices) == 0:
                bbox = torch.tensor([0.45, 0.45, 0.55, 0.55], device=device, dtype=mask.dtype)
                bboxes.append(bbox)
                continue
            
            # 计算边界框
            x_min = x_indices.min().float()
            x_max = x_indices.max().float()
            y_min = y_indices.min().float()
            y_max = y_indices.max().float()
            
            # 归一化到[0,1]
            x1 = x_min / W
            y1 = y_min / H
            x2 = x_max / W
            y2 = y_max / H
            
            # 确保x2 > x1, y2 > y1
            if x2 <= x1:
                x2 = x1 + 0.01
            if y2 <= y1:
                y2 = y1 + 0.01
            
            bbox = torch.stack([x1, y1, x2, y2])
            bboxes.append(bbox)
            
        except Exception as e:
            print(f"Error in mask2Boxes for mask {i}: {e}")
            bbox = torch.tensor([0.45, 0.45, 0.55, 0.55], device=device, dtype=mask.dtype)
            bboxes.append(bbox)
    
    # 堆叠所有bboxes
    bbox_tensor = torch.stack(bboxes, dim=0)  # n, 4
    
    # 限制在[0,1]范围内
    bbox_tensor = torch.clamp(bbox_tensor, 0.0, 1.0)
    
    return bbox_tensor

def xyhw2xyxy(bbox):
    """
    Args:
        bbox: N, 4 [x, y, h, w] in [0,1]

    Returns:
        bbox: N, 4 [x1, y1, x2, y2] in [0,1]
    """
    x, y, h, w = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x1, y1, x2, y2 = x-w/2., y-h/2., x+w/2., y+h/2.
    return torch.stack([x1, y1, x2, y2], dim=-1)  ## N, 4

def xyxy2xyhw(bbox):
    """
    Args:
        bbox: N, 4 [x1, y1, x2, y2] in [0,1]

    Returns:
        bbox: N, 4 [x, y, h, w] in [0,1]
    """
    x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x, y, h, w = (x1+x2)/2., (y1+y2)/2., y2-y1, x2-x1
    return torch.stack([x, y, h, w], dim=-1)  ## N, 4