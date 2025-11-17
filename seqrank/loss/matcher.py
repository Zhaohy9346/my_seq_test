import scipy
import torch
import numpy as np
from typing import Dict, List, Tuple
from .loss import batch_mask_loss, batch_mask_loss_in_points

@torch.no_grad()
def hungarianMatcher(preds: Dict, targets: List, cfg=None) -> Tuple:
    """
        Params:
            preds: a dict:
                "masks": torch.Tensor [B,nq,H,W] logit
                "scores": torch.Tensor B,nq,1 logit
            targets: list of targets with length=batch_size, each is a torch.Tensor
                in shape N,H,W (binary map indicates the foreground/background)
        Returns:
            indices: tuple [Tensor, Tensor, Tensor]
                each tensor is a 1D vector, repr index.
    """
    B = len(targets)
    indices = [[], [], []]
    for b in range(B):
        tgts = targets[b].unsqueeze(1)  ## N,1,H,W
        masks = preds["masks"][b].unsqueeze(1)  ## nq, 1, h, w
        N, _, H, W = tgts.shape
        nq = len(masks)

        mask_loss = batch_mask_loss(torch.repeat_interleave(masks, N, dim=0), tgts.repeat(nq, 1, 1, 1), cfg=cfg).reshape(nq, N)  ## nq, N
        cls_loss = -torch.sigmoid(preds["scores"][b]).repeat_interleave(N, dim=1)  ## nq, N
        cost_matrix = mask_loss + cls_loss  ## nq, N
        row_idxs, col_idxs = scipy.optimize.linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        indices[0].append((torch.ones(len(row_idxs), device=cost_matrix.device) * b).long())  ## batch index
        indices[1].append(torch.tensor(row_idxs, device=cost_matrix.device, dtype=torch.long))  ## query index
        indices[2].append(torch.tensor(col_idxs, device=cost_matrix.device, dtype=torch.long))  ## target index
    return tuple(torch.cat(idx) for idx in indices)


@torch.no_grad()
def hungarianMatcherInPoints(preds: Dict, targets: List, cfg=None) -> Tuple:
    """
    修复版本：处理NaN和Inf的情况
    """
    B = len(targets)
    indices = [[], [], []]
    
    for b in range(B):
        tgts = targets[b].unsqueeze(1)  ## N,1,H,W
        masks = preds["masks"][b].unsqueeze(1)  ## nq, 1, h, w
        N, _, H, W = tgts.shape
        nq = len(masks)

        # 检查targets是否有效
        if N == 0:
            print(f"Warning: Batch {b} has no targets, skipping")
            continue
        
        # 检查是否有空的target
        target_sums = tgts.sum(dim=[1, 2, 3])
        if (target_sums == 0).any():
            print(f"Warning: Batch {b} has empty target masks")
            # 过滤掉空的targets
            valid_mask = target_sums > 0
            if valid_mask.sum() == 0:
                print(f"Error: All targets in batch {b} are empty, skipping")
                continue
            tgts = tgts[valid_mask]
            N = tgts.shape[0]
        
        try:
            # 计算mask loss
            mask_loss = batch_mask_loss_in_points(
                torch.repeat_interleave(masks, N, dim=0), 
                tgts.repeat(nq, 1, 1, 1), 
                cfg=cfg
            ).reshape(nq, N)  ## nq, N
            
            # 检查mask_loss是否包含NaN或Inf
            if torch.isnan(mask_loss).any() or torch.isinf(mask_loss).any():
                print(f"Warning: NaN or Inf detected in mask_loss at batch {b}")
                print(f"  NaN count: {torch.isnan(mask_loss).sum().item()}")
                print(f"  Inf count: {torch.isinf(mask_loss).sum().item()}")
                # 替换NaN和Inf为大数值
                mask_loss = torch.nan_to_num(mask_loss, nan=100.0, posinf=100.0, neginf=100.0)
            
            # 计算classification loss
            cls_loss = -torch.sigmoid(preds["scores"][b]).repeat_interleave(N, dim=1)  ## nq, N
            
            # 检查cls_loss
            if torch.isnan(cls_loss).any() or torch.isinf(cls_loss).any():
                print(f"Warning: NaN or Inf detected in cls_loss at batch {b}")
                cls_loss = torch.nan_to_num(cls_loss, nan=1.0, posinf=1.0, neginf=1.0)
            
            # 总cost
            cost_matrix = mask_loss + cls_loss  ## nq, N
            
            # 最终检查
            cost_matrix_np = cost_matrix.detach().cpu().numpy()
            
            # 检查是否包含无效值
            if not np.isfinite(cost_matrix_np).all():
                print(f"Error: Cost matrix contains invalid values at batch {b}")
                print(f"  NaN: {np.isnan(cost_matrix_np).sum()}")
                print(f"  Inf: {np.isinf(cost_matrix_np).sum()}")
                print(f"  Min: {np.nanmin(cost_matrix_np)}, Max: {np.nanmax(cost_matrix_np)}")
                
                # 替换所有无效值
                cost_matrix_np = np.nan_to_num(
                    cost_matrix_np, 
                    nan=100.0, 
                    posinf=100.0, 
                    neginf=100.0
                )
            
            # Hungarian matching
            row_idxs, col_idxs = scipy.optimize.linear_sum_assignment(cost_matrix_np)
            
            indices[0].append((torch.ones(len(row_idxs), device=masks.device) * b).long())
            indices[1].append(torch.tensor(row_idxs, device=masks.device, dtype=torch.long))
            indices[2].append(torch.tensor(col_idxs, device=masks.device, dtype=torch.long))
            
        except Exception as e:
            print(f"Error in hungarianMatcherInPoints for batch {b}: {e}")
            print(f"  N={N}, nq={nq}, H={H}, W={W}")
            print(f"  masks shape: {masks.shape}, tgts shape: {tgts.shape}")
            # 如果出错，跳过这个batch
            continue
    
    # 检查是否有有效的匹配
    if len(indices[0]) == 0:
        print("Error: No valid matches found in any batch!")
        # 返回空的索引
        device = preds["masks"].device
        return (
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.long, device=device)
        )
    
    return tuple(torch.cat(idx) for idx in indices)