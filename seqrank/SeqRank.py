import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .neck import build_neck_head
from .modules import build_gaze_shift_head, build_sis_head
from .component import PositionEmbeddingSine, PositionEmbeddingRandom
from .utils import calc_iou, debugDump, pad1d, mask2Boxes, xyhw2xyxy, xyxy2xyhw
from .loss import hungarianMatcherInPoints, batch_mask_loss_in_points, batch_bbox_loss

class LearnablePE(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.ape = nn.parameter.Parameter(torch.zeros((embed_dim, 25, 25)), requires_grad=True)
        nn.init.trunc_normal_(self.ape)

    def forward(self, x):
        """
        x: B, C, H, W
        return: B, C, H, W
        """
        ape = F.interpolate(self.ape.unsqueeze(0), size=x.shape[2::], mode="bilinear")  ## 1, C, H, W
        return ape.expand(len(x), -1, -1, -1)  ## B, C, H, W


@META_ARCH_REGISTRY.register()
class SeqRank(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.neck = build_neck_head(cfg)
        self.instance_seg = build_sis_head(cfg)
        self.gaze_shift = build_gaze_shift_head(cfg)

        self.pe_layer = {
            "SINE": PositionEmbeddingSine(cfg.MODEL.COMMON.EMBED_DIM // 2, normalize=True),
            "RANDOM": PositionEmbeddingRandom(cfg.MODEL.COMMON.EMBED_DIM // 2),
            "APE": LearnablePE(cfg.MODEL.COMMON.EMBED_DIM)
        }[cfg.MODEL.PE]

        self.cfg = cfg
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(1, -1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).reshape(1, 3, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batch_dict, *args, **argw):
        ## prepare image
        images = torch.stack([s["image"] for s in batch_dict], dim=0).to(self.device).contiguous()
        images = (images - self.pixel_mean) / self.pixel_std

        zs = self.backbone(images)
        zs = self.neck(zs)
        zs_pe = dict((k, self.pe_layer(zs[k])) for k in zs)
        q, qpe, out, auxs = self.instance_seg(
            feats=zs,
            feats_pe=zs_pe
        )

        pred_masks = out["masks"]  ## B, nq, H, W
        pred_bboxes = out["bboxes"].sigmoid()  ## B, nq, 4 [xyhw] in [0,1]
        pred_objs = out["scores"]  ## B, nq, 1
        gaze_shift_key = self.cfg.MODEL.GAZE_SHIFT_HEAD.KEY

        if self.training:
            ## Training
            masks = [x["masks"].to(self.device) for x in batch_dict]  ## list of k_i, Ht, Wt
            bboxes = [mask2Boxes(m) for m in masks]  ## list of k_i, 4[xyxy]
            n_max = max([len(x) for x in masks])
            gt_size = masks[0].shape[-2::]

            pred_masks = F.interpolate(pred_masks, size=gt_size, mode="bilinear")
            bi, qi, ti = hungarianMatcherInPoints(preds={"masks": pred_masks, "scores": pred_objs}, targets=masks, cfg=self.cfg)

            q_masks = torch.stack([pad1d(m, dim=0, num=n_max, value=0.0) for m in masks], dim=0)  ## B, n_max, H, W
            q_boxes = torch.stack([pad1d(bb, dim=0, num=n_max, value=0.0) for bb in bboxes], dim=0)  ## B, n_max, 4

            q_corresponse = torch.zeros_like(pred_objs)  ## B, nq, 1
            q_corresponse[bi, qi, 0] = (ti + 1).to(q_corresponse.dtype)  ## 1 to n_max

            mask_loss = batch_mask_loss_in_points(pred_masks[bi, qi], q_masks[bi, ti], cfg=self.cfg).mean()
            bbox_loss = batch_bbox_loss(xyhw2xyxy(pred_bboxes[bi, qi]), q_boxes[bi, ti], cfg=self.cfg).mean()
            
            obj_pos_weight = torch.tensor(self.cfg.LOSS.OBJ_POS, device=self.device)
            obj_neg_weight = torch.tensor(self.cfg.LOSS.OBJ_NEG, device=self.device)
            pos_mask = q_corresponse.gt(.5).float()
            neg_mask = 1.0 - pos_mask
            pos_obj_loss = F.binary_cross_entropy_with_logits(pred_objs, torch.ones_like(pred_objs), reduction="none")
            neg_obj_loss = F.binary_cross_entropy_with_logits(pred_objs, torch.zeros_like(pred_objs), reduction="none")
            obj_loss = (pos_obj_loss * obj_pos_weight * pos_mask + neg_obj_loss * obj_neg_weight * neg_mask).mean()

            # ====== 关键修复：确保aux loss始终参与计算图 ======
            if self.cfg.LOSS.AUX == "disable" or len(auxs) <= 0:
                # 不直接赋值为0，而是创建一个依赖于其他loss的虚拟loss
                # 这样可以确保所有参数都有梯度流
                aux_mask_loss = mask_loss * 0.0  # 保持计算图连接
                aux_bbox_loss = bbox_loss * 0.0  # 保持计算图连接
            else:
                aux_mask_loss = sum([
                    batch_mask_loss_in_points(
                        F.interpolate(aux["masks"], size=gt_size, mode="bilinear")[bi, qi],
                        q_masks[bi, ti],
                        cfg=self.cfg
                    ).mean()
                    for aux in auxs
                ])
                aux_bbox_loss = sum([
                    batch_bbox_loss(
                        xyhw2xyxy(torch.sigmoid(aux["bboxes"][bi, qi])),
                        q_boxes[bi, ti],
                        cfg=self.cfg
                    ).mean()
                    for aux in auxs
                ])

            sal_loss = torch.zeros_like(obj_loss)  ## 初始化为依赖obj_loss的tensor
            for i in range(n_max + 1):
                q_vis = q_corresponse * q_corresponse.le(i).float()
                q_ans = q_corresponse.eq(i + 1).float()
                sal, _ = self.gaze_shift(
                    q=q,
                    z=zs[gaze_shift_key].flatten(2).transpose(-1, -2),
                    qpe=qpe,
                    zpe=zs_pe[gaze_shift_key].flatten(2).transpose(-1, -2),
                    q_vis=q_vis,
                    bbox=pred_bboxes,
                    size=tuple(zs[gaze_shift_key].shape[2::])
                )
                sal_pos = F.binary_cross_entropy_with_logits(sal, torch.ones_like(sal), reduction="none")
                sal_neg = F.binary_cross_entropy_with_logits(sal, torch.zeros_like(sal), reduction="none")
                sal_ele_loss = q_ans * sal_pos + (1.0 - q_ans) * sal_neg
                sal_ele_loss = torch.where(torch.isnan(sal_ele_loss), torch.zeros_like(sal_ele_loss), sal_ele_loss)
                sal_loss = sal_loss + sal_ele_loss.mean()

            # ====== NaN保护 ======
            mask_loss = torch.nan_to_num(mask_loss, nan=1.0, posinf=1.0, neginf=1.0)
            bbox_loss = torch.nan_to_num(bbox_loss, nan=1.0, posinf=1.0, neginf=1.0)
            obj_loss = torch.nan_to_num(obj_loss, nan=0.1, posinf=0.1, neginf=0.1)
            sal_loss = torch.nan_to_num(sal_loss, nan=0.1, posinf=0.1, neginf=0.1)
            aux_mask_loss = torch.nan_to_num(aux_mask_loss, nan=0.0, posinf=0.0, neginf=0.0)
            aux_bbox_loss = torch.nan_to_num(aux_bbox_loss, nan=0.0, posinf=0.0, neginf=0.0)

            # debugDump
            if np.random.rand() < 0.1:
                k = 5
                mm = pred_masks[bi, qi].sigmoid()[0:k].detach().cpu()
                tt = q_masks[bi, ti].cpu()[0:k]
                ss = pred_objs[bi, qi, 0].sigmoid()[0:k].detach().cpu().tolist()
                oo = [float(calc_iou(m, t)) for m, t in zip(mm, tt)]
                debugDump(
                    output_dir=self.cfg.OUTPUT_DIR,
                    image_name="latest",
                    texts=[ss, oo],
                    lsts=[list(mm), list(tt)],
                    data=None
                )

                        # ====== 检查是否有NaN，如果有就跳过这个batch ======
            if torch.isnan(mask_loss) or torch.isinf(mask_loss) or \
            torch.isnan(bbox_loss) or torch.isinf(bbox_loss) or \
            torch.isnan(obj_loss) or torch.isinf(obj_loss) or \
            torch.isnan(sal_loss) or torch.isinf(sal_loss):
                
                print(f"\n{'='*60}")
                print(f"WARNING: NaN/Inf detected in losses! Skipping this batch.")
                print(f"Batch images:")
                for idx, item in enumerate(batch_dict):
                    print(f"  {item.get('image_name', 'unknown')}")
                print(f"Loss values:")
                print(f"  mask_loss: {mask_loss}")
                print(f"  bbox_loss: {bbox_loss}")
                print(f"  obj_loss: {obj_loss}")
                print(f"  sal_loss: {sal_loss}")
                print(f"{'='*60}\n")
                
                # 返回一个安全的loss dict，但值很小，不影响训练
                return {
                    "mask_loss": torch.tensor(0.01, device=self.device, requires_grad=True),
                    "bbox_loss": torch.tensor(0.01, device=self.device, requires_grad=True),
                    "obj_loss": torch.tensor(0.001, device=self.device, requires_grad=True) * self.cfg.LOSS.CLS_COST,
                    "sal_loss": torch.tensor(0.001, device=self.device, requires_grad=True) * self.cfg.LOSS.SAL_COST,
                    "aux_mask_loss": torch.tensor(0.0, device=self.device, requires_grad=True),
                    "aux_bbox_loss": torch.tensor(0.0, device=self.device, requires_grad=True)
                }

            # 正常返回
            return {
                "mask_loss": mask_loss,
                "bbox_loss": bbox_loss,
                "obj_loss": obj_loss * self.cfg.LOSS.CLS_COST,
                "sal_loss": sal_loss * self.cfg.LOSS.SAL_COST,
                "aux_mask_loss": aux_mask_loss * self.cfg.LOSS.AUX_WEIGHT,
                "aux_bbox_loss": aux_bbox_loss * self.cfg.LOSS.AUX_WEIGHT
            }
        else:
            ## inference - 保持不变
            size = tuple(zs[gaze_shift_key].shape[2::])
            z = zs[gaze_shift_key].flatten(2).transpose(-1, -2)
            zpe = zs_pe[gaze_shift_key].flatten(2).transpose(-1, -2)
            q_vis = torch.zeros_like(pred_objs)
            bs, nq, _ = q.shape
            bs_idx = torch.arange(bs, device=self.device, dtype=torch.long)

            results = [{
                "image_name": x.get("image_name", idx),
                "masks": [],
                "bboxes": [],
                "scores": [],
                "saliency": [],
                "num": 0
            } for idx, x in enumerate(batch_dict)]

            ends_batch = set()
            for i in range(nq):
                sal = self.gaze_shift(q=q, z=z, qpe=qpe, zpe=zpe, q_vis=q_vis, bbox=pred_bboxes, size=size)
                sal_max = torch.argmax(sal[:, :, 0], dim=1).long()
                q_vis[bs_idx, sal_max, 0] = i + 1

                sal_scores = sal[bs_idx, sal_max, 0].sigmoid()
                obj_scores = pred_objs[bs_idx, sal_max, 0].sigmoid()
                the_masks = pred_masks[bs_idx, sal_max, :, :]
                the_bboxes = xyhw2xyxy(pred_bboxes[bs_idx, sal_max, :])

                t_sal = 0.001
                t_obj = 0.001

                for idx in range(bs):
                    obj_score = obj_scores[idx]
                    sal_score = sal_scores[idx]
                    if obj_score < t_obj or sal_score < t_sal:
                        ends_batch.add(idx)
                    if idx in ends_batch: continue

                    hi, wi = batch_dict[idx]["height"], batch_dict[idx]["width"]
                    results[idx]["masks"].append(
                        F.interpolate(the_masks[idx:idx + 1, :, :].unsqueeze(1), size=(hi, wi), mode="bilinear")[
                            0, 0].sigmoid().detach().cpu().gt(.5).float().numpy()
                    )
                    results[idx]["bboxes"].append(
                        (the_bboxes[idx].detach().cpu() * torch.tensor([wi, hi, wi, hi])).tolist()
                    )
                    results[idx]["scores"].append(
                        float(obj_scores[idx].detach().cpu())
                    )
                    results[idx]["saliency"].append(
                        float(sal_scores[idx].detach().cpu())
                    )
                    results[idx]["num"] += 1
                if len(ends_batch) >= bs:
                    break
            return results