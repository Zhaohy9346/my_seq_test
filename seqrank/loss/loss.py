import torch
import torch.nn.functional as F
import torchvision
from ..utils import xyxy2xyhw

def batch_mask_loss(preds, targets, cfg=None):
    """
    CE loss + dice loss
    添加数值稳定性检查
    """
    if cfg is None:
        ce_loss_weight = 1.0
        dice_loss_weight = 1.0
    else:
        ce_loss_weight = cfg.LOSS.MASK_CE_COST
        dice_loss_weight = cfg.LOSS.MASK_DICE_COST
        
    preds = preds.flatten(1)  ## B,-1
    targets = targets.flatten(1)  ## B, -1
    
    # 检查targets是否全为0
    target_sums = targets.sum(dim=1)
    if (target_sums == 0).any():
        print("Warning: Some targets are completely empty in batch_mask_loss")
    
    sig_preds = torch.sigmoid(preds)  ## B, -1

    # CE loss with numerical stability
    ce_loss = F.binary_cross_entropy_with_logits(
        preds, 
        targets, 
        reduction="none"
    ).mean(dim=-1)  ## B
    
    # Dice loss with epsilon for stability
    eps = 1e-7
    intersection = (sig_preds * targets).sum(dim=-1)
    union = (sig_preds + targets).sum(dim=-1)
    dice_loss = 1.0 - (2. * intersection + eps) / (union + eps)  ## B
    
    total_loss = ce_loss * ce_loss_weight + dice_loss * dice_loss_weight
    
    # 检查是否产生了NaN
    if torch.isnan(total_loss).any():
        print("Warning: NaN detected in batch_mask_loss")
        print(f"  CE loss: {ce_loss}")
        print(f"  Dice loss: {dice_loss}")
        print(f"  Target sums: {target_sums}")
        total_loss = torch.nan_to_num(total_loss, nan=10.0)
    
    return total_loss


def batch_mask_loss_in_points(preds, targets, cfg=None):
    """
    preds: *, H, W
    targets: *, H, W
    """
    H, W = preds.shape[-2::]
    K = cfg.LOSS.NUM_POINTS
    if H*W <= K:
        return batch_mask_loss(preds=preds, targets=targets, cfg=cfg)
    
    assert targets.shape[-2::]==preds.shape[-2::]
    
    # 采样点
    khi = torch.randint(low=0, high=H, size=(K,)).to(preds.device).long()
    kwi = torch.randint(low=0, high=W, size=(K,)).to(preds.device).long()
    
    sampled_preds = preds.reshape(-1, H, W)[:, khi, kwi]
    sampled_targets = targets.reshape(-1, H, W)[:, khi, kwi]
    
    return batch_mask_loss(
        preds=sampled_preds,
        targets=sampled_targets,
        cfg=cfg
    )


def batch_bbox_loss(box1, box2, cfg=None):
    """
    boxes in [(x1,y1),(x2,y2)]
    添加数值检查
    """
    if cfg is None:
        bbox_l1_weight = 1.0
        bbox_giou_weight = 1.0
    else:
        bbox_l1_weight = cfg.LOSS.BBOX_L1_COST
        bbox_giou_weight = cfg.LOSS.BBOX_GIOU_COST

    # 检查box是否有效
    if (box1 < 0).any() or (box1 > 1).any():
        print(f"Warning: box1 out of range [0,1]: min={box1.min()}, max={box1.max()}")
        box1 = torch.clamp(box1, 0.0, 1.0)
    
    if (box2 < 0).any() or (box2 > 1).any():
        print(f"Warning: box2 out of range [0,1]: min={box2.min()}, max={box2.max()}")
        box2 = torch.clamp(box2, 0.0, 1.0)
    
    # 确保box有效（x2 > x1, y2 > y1）
    eps = 1e-6
    box1 = torch.stack([
        box1[:, 0],
        box1[:, 1],
        torch.maximum(box1[:, 2], box1[:, 0] + eps),
        torch.maximum(box1[:, 3], box1[:, 1] + eps)
    ], dim=1)
    
    box2 = torch.stack([
        box2[:, 0],
        box2[:, 1],
        torch.maximum(box2[:, 2], box2[:, 0] + eps),
        torch.maximum(box2[:, 3], box2[:, 1] + eps)
    ], dim=1)

    version = [int(_) for _ in torchvision.__version__.split("+")[0].split(".")]
    if version[1] >= 15:
        gloss = torchvision.ops.generalized_box_iou_loss(box1, box2)  ## N
    else:
        gloss = -torch.diag(torchvision.ops.generalized_box_iou(box1, box2))  ## N
    
    l1loss = F.l1_loss(xyxy2xyhw(box1), xyxy2xyhw(box2), reduction="none").mean(dim=-1)
    
    total_loss = l1loss * bbox_l1_weight + gloss * bbox_giou_weight
    
    # 检查NaN
    if torch.isnan(total_loss).any():
        print("Warning: NaN in bbox_loss")
        total_loss = torch.nan_to_num(total_loss, nan=5.0)
    
    return total_loss