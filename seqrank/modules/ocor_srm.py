"""
OCOR-based Sequential Ranking Module (OCOR-SRM)
将原始SRM的self-attn/cross-attn替换为OCOR的object-context prioritization机制
保留iterative selection loop的核心架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable

from ..component import init_weights_
from .registry import GAZE_SHIFT_HEAD


class ObjectContextPrioritization(nn.Module):
    """
    OCOR的核心模块：Object-Context Prioritization
    通过动态调节object和context特征的权重来实现显著性排序
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Object特征处理
        self.object_norm = nn.LayerNorm(embed_dim)
        self.object_proj = nn.Linear(embed_dim, embed_dim)

        # Context特征处理
        self.context_norm = nn.LayerNorm(embed_dim)
        self.context_proj = nn.Linear(embed_dim, embed_dim)

        # Prioritization权重生成器
        self.priority_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),  # [object_weight, context_weight]
            nn.Softmax(dim=-1)
        )

        # Multi-head attention for object-context interaction
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def compute_priority_weights(self, obj_feat, ctx_feat):
        """
        计算object和context的优先级权重
        Args:
            obj_feat: B, nq, C (object features)
            ctx_feat: B, nq, C (aggregated context features)
        Returns:
            weights: B, nq, 2 ([object_weight, context_weight])
        """
        combined = torch.cat([obj_feat, ctx_feat], dim=-1)  # B, nq, 2C
        weights = self.priority_gate(combined)  # B, nq, 2
        return weights

    def multi_head_attention(self, q, k, v, mask=None):
        """
        Multi-head attention mechanism
        """
        B, N, C = q.shape

        # Reshape for multi-head
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B, H, N, D
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # B, H, M, D
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # B, H, M, D

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # B, H, N, M

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B, N, C
        out = self.out_proj(out)

        return out

    def forward(self, queries, context, query_pe=None, context_pe=None, gaze_mask=None):
        """
        Object-Context Prioritization forward pass

        Args:
            queries: B, nq, C (object queries)
            context: B, hw, C (context features from backbone)
            query_pe: B, nq, C (query positional encoding)
            context_pe: B, hw, C (context positional encoding)
            gaze_mask: B, nq, 1 (indicate which objects are already gazed)

        Returns:
            refined_queries: B, nq, C
        """
        # 1. Normalize and project
        obj_feat = self.object_norm(queries)
        ctx_feat = self.context_norm(context)

        obj_proj = self.object_proj(obj_feat)
        ctx_proj = self.context_proj(ctx_feat)

        # 2. Aggregate context for each object query
        # Use attention to get context-aware object features
        q = obj_proj if query_pe is None else obj_proj + query_pe
        k = ctx_proj if context_pe is None else ctx_proj + context_pe
        v = ctx_proj

        # Multi-head attention: queries attend to context
        ctx_aware = self.multi_head_attention(q, k, v)

        # 3. Compute prioritization weights
        # Aggregate context features for priority computation
        ctx_aggregated = torch.mean(ctx_proj, dim=1, keepdim=True).expand(-1, queries.shape[1], -1)
        priority_weights = self.compute_priority_weights(obj_proj, ctx_aggregated)

        # 4. Apply prioritization
        obj_weight = priority_weights[:, :, 0:1]  # B, nq, 1
        ctx_weight = priority_weights[:, :, 1:2]  # B, nq, 1

        # Weighted combination
        prioritized = obj_weight * obj_proj + ctx_weight * ctx_aware

        # 5. Add residual connection
        queries = queries + prioritized

        # 6. Feed-forward network
        queries = queries + self.ffn(self.ffn_norm(queries))

        return queries


class GazeAwarePrioritization(nn.Module):
    """
    Gaze-Aware模块：考虑已经被注视的物体的影响
    这是OCOR中的关键创新，用于sequential ranking
    """
    def __init__(self, embed_dim=256, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim

        # Gaze状态编码
        self.gaze_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # [current_obj, gazed_info]
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Previous gaze aggregation
        self.prev_attn = nn.MultiheadAttention(
            embed_dim, num_heads=8, dropout=dropout, batch_first=True
        )
        self.prev_norm = nn.LayerNorm(embed_dim)

        init_weights_(self)

    def aggregate_previous_gaze(self, queries, gaze_mask):
        """
        聚合之前已经被注视的物体信息
        Args:
            queries: B, nq, C
            gaze_mask: B, nq, 1 (0: not gazed, >0: gaze order)
        Returns:
            gazed_info: B, nq, C
        """
        B, nq, C = queries.shape

        # Create mask for previously gazed objects
        prev_mask = gaze_mask.gt(0).float()  # B, nq, 1

        # Use attention to aggregate previous gaze information
        # Only attend to previously gazed objects
        attn_mask = prev_mask.squeeze(-1).unsqueeze(1).expand(-1, nq, -1)  # B, nq, nq
        attn_mask = (attn_mask == 0)  # True for positions to ignore

        # Self-attention with mask
        gazed_info, _ = self.prev_attn(
            queries, queries, queries,
            key_padding_mask=None,
            attn_mask=attn_mask
        )

        gazed_info = self.prev_norm(queries + gazed_info)

        return gazed_info

    def forward(self, queries, gaze_mask, prev_emb, gaze_emb):
        """
        Args:
            queries: B, nq, C
            gaze_mask: B, nq, 1 (gaze status: 0=not gazed, >0=gaze order)
            prev_emb: 1, 1, C (embedding for previously gazed)
            gaze_emb: 1, 1, C (embedding for currently gazing)
        Returns:
            gaze_aware_queries: B, nq, C
        """
        # Aggregate information from previously gazed objects
        gazed_info = self.aggregate_previous_gaze(queries, gaze_mask)

        # Combine current queries with gazed information
        combined = torch.cat([queries, gazed_info], dim=-1)
        gaze_aware = self.gaze_encoder(combined)

        # Add gaze status embeddings
        prev = (gaze_mask > 0).float()  # B, nq, 1
        gaze = (gaze_mask == gaze_mask.max(dim=1, keepdim=True)[0]).float()  # B, nq, 1

        gaze_aware = gaze_aware + prev * prev_emb + gaze * gaze_emb

        return gaze_aware


class OCORCenterShiftBlock(nn.Module):
    """
    OCOR版本的Center Shift Block
    使用Object-Context Prioritization替代传统的self-attn/cross-attn
    """
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout_attn=0.0, dropout_ffn=0.0):
        super().__init__()

        # Object-Context Prioritization模块
        self.obj_ctx_prioritization = ObjectContextPrioritization(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_attn
        )

        # Gaze-Aware Prioritization模块
        self.gaze_aware = GazeAwarePrioritization(
            embed_dim=embed_dim,
            dropout=dropout_attn
        )

        # Saliency prediction head (与原始SRM保持一致)
        self.saliency_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(embed_dim, 1)
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Output projection (for iterative refinement)
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        init_weights_(self)

    def forward(self, q, qpe, z, zpe, gaze, gaze_emb, prev, prev_emb):
        """
        Args:
            q: B, nq, C (object queries)
            qpe: B, nq, C (query positional encoding)
            z: B, hw, C (context features from backbone)
            zpe: B, hw, C (context positional encoding)
            gaze: B, nq, 1 (current gaze mask)
            gaze_emb: 1, 1, C (gaze embedding)
            prev: B, nq, 1 (previously gazed mask)
            prev_emb: 1, 1, C (previous embedding)
        Returns:
            refined_q: B, nq, C
        """
        q0 = q  # Save for residual

        # 1. Gaze-Aware Processing
        # 考虑已经被注视的物体的影响
        gaze_mask = prev + gaze  # Combine previous and current gaze
        q_gaze = self.gaze_aware(q, gaze_mask, prev_emb, gaze_emb)
        q = self.norm1(q + q_gaze)

        # 2. Object-Context Prioritization
        # 动态平衡object和context特征
        q_prioritized = self.obj_ctx_prioritization(
            queries=q,
            context=z,
            query_pe=qpe,
            context_pe=zpe,
            gaze_mask=gaze_mask
        )
        q = self.norm2(q + q_prioritized)

        # 3. Output projection with residual from q0
        q = self.output_proj(q0 + q)

        return q


@GAZE_SHIFT_HEAD.register()
class OCORSequentialRankingModule(nn.Module):
    """
    基于OCOR的Sequential Ranking Module
    保留原始SRM的iterative selection loop
    使用OCOR的object-context prioritization机制
    """
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024,
                 dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2):
        super().__init__()

        # Gaze状态的embedding (与原始SRM保持一致)
        self.prev_emb = nn.Parameter(torch.randn(embed_dim))
        self.gaze_emb = nn.Parameter(torch.randn(embed_dim))

        # OCOR blocks替代原始的attention blocks
        self.blocks = nn.ModuleList([
            OCORCenterShiftBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dropout_attn=dropout_attn,
                dropout_ffn=dropout_ffn
            ) for _ in range(num_blocks)
        ])

        # Saliency prediction head (与原始SRM保持一致)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # 可选：添加一个confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )

    @classmethod
    def from_config(cls, cfg):
        return {
            "embed_dim":    cfg.MODEL.COMMON.EMBED_DIM,
            "num_heads":    cfg.MODEL.COMMON.NUM_HEADS,
            "dropout_attn": cfg.MODEL.COMMON.DROPOUT_ATTN,
            "hidden_dim":   cfg.MODEL.COMMON.HIDDEN_DIM,
            "dropout_ffn":  cfg.MODEL.COMMON.DROPOUT_FFN,
            "num_blocks":   cfg.MODEL.GAZE_SHIFT_HEAD.NUM_BLOCKS
        }

    def forward(self, q, z, qpe, zpe, q_vis, bbox, size):
        """
        与原始SRM保持相同的接口，保留iterative selection loop

        Args:
            q: B, nq, C (object queries from FOM)
            z: B, hw, C (context features from backbone)
            qpe: B, nq, C (query positional encoding)
            zpe: B, hw, C (context positional encoding)
            q_vis: B, nq, 1 (visibility status: 0 or gaze order 1,2,3...)
            bbox: B, nq, 4 [xyhw] in [0,1] (bounding boxes)
            size: Tuple(h, w) (feature map size)

        Returns:
            saliency: B, nq, 1 (saliency scores, logits)
            confidences: B, nq, 1 (optional confidence scores)
        """
        prev_emb = self.prev_emb[None, None, :]  # 1, 1, C
        gaze_emb = self.gaze_emb[None, None, :]  # 1, 1, C

        # 计算gaze状态 (与原始SRM相同)
        gaze_rank, _ = torch.max(q_vis[:, :, 0], dim=1)  # B
        prev = (q_vis.gt(0.5) * q_vis.le(gaze_rank[:, None, None])).float()  # B, nq, 1
        gaze = (q_vis.gt(0.5) * q_vis.eq(gaze_rank[:, None, None])).float()  # B, nq, 1

        # Iterative refinement through OCOR blocks
        predictions = []
        confidences = []

        for layer in self.blocks:
            # OCOR-based refinement
            q = layer(
                q=q,
                qpe=qpe,
                z=z,
                zpe=zpe,
                gaze=gaze,
                gaze_emb=gaze_emb,
                prev=prev,
                prev_emb=prev_emb
            )

            # Saliency prediction
            sal = self.head(q)  # B, nq, 1
            predictions.append(sal)

            # Optional: predict confidence
            conf = self.confidence_head(q)  # B, nq, 1
            confidences.append(conf)

        # Return最后一层的预测
        out = predictions[-1]
        out_conf = confidences[-1]

        if self.training:
            # Training: return all intermediate predictions for auxiliary loss
            return out, predictions[:-1]
        else:
            # Inference: return final prediction and confidence
            return out, out_conf


# ============ 新增：OCOR-SRM的变体版本 ============

@GAZE_SHIFT_HEAD.register()
class OCORSequentialRankingModuleV2(OCORSequentialRankingModule):
    """
    OCOR-SRM变体版本：添加显式的ranking loss
    """
    @configurable
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024,
                 dropout_attn=0.0, dropout_ffn=0.0, num_blocks=2):
        super().__init__(embed_dim, num_heads, hidden_dim, dropout_attn, dropout_ffn, num_blocks)

        # Ranking比较模块
        self.rank_comparator = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_ffn),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability that obj1 > obj2
        )

    def compute_pairwise_ranking(self, q):
        """
        计算成对的ranking关系
        Args:
            q: B, nq, C
        Returns:
            rank_scores: B, nq, nq (probability that i > j)
        """
        B, nq, C = q.shape

        # Expand for pairwise comparison
        q1 = q.unsqueeze(2).expand(-1, -1, nq, -1)  # B, nq, nq, C
        q2 = q.unsqueeze(1).expand(-1, nq, -1, -1)  # B, nq, nq, C

        # Concatenate pairs
        pairs = torch.cat([q1, q2], dim=-1)  # B, nq, nq, 2C

        # Compute ranking scores
        rank_scores = self.rank_comparator(pairs).squeeze(-1)  # B, nq, nq

        return rank_scores

    def forward(self, q, z, qpe, zpe, q_vis, bbox, size):
        """扩展forward以包含ranking信息"""
        # 调用父类的forward
        sal, aux = super().forward(q, z, qpe, zpe, q_vis, bbox, size)

        if self.training:
            # 计算pairwise ranking scores
            rank_scores = self.compute_pairwise_ranking(q)
            return sal, aux, rank_scores
        else:
            return sal, aux


# ============ 辅助函数：用于可视化和调试 ============

def visualize_prioritization_weights(model, queries, context):
    """
    可视化object-context prioritization的权重分布
    用于调试和分析
    """
    with torch.no_grad():
        # Get the first block
        block = model.blocks[0]
        obj_ctx_module = block.obj_ctx_prioritization

        # Compute priority weights
        obj_feat = obj_ctx_module.object_norm(queries)
        ctx_feat = obj_ctx_module.context_norm(context)

        obj_proj = obj_ctx_module.object_proj(obj_feat)
        ctx_aggregated = torch.mean(ctx_feat, dim=1, keepdim=True).expand(-1, queries.shape[1], -1)

        weights = obj_ctx_module.compute_priority_weights(obj_proj, ctx_aggregated)

        return weights  # B, nq, 2 ([object_weight, context_weight])


if __name__ == "__main__":
    # 测试代码
    print("Testing OCOR-based Sequential Ranking Module...")

    # 创建测试数据
    B, nq, hw, C = 2, 10, 400, 256

    q = torch.randn(B, nq, C)
    z = torch.randn(B, hw, C)
    qpe = torch.randn(B, nq, C)
    zpe = torch.randn(B, hw, C)
    q_vis = torch.zeros(B, nq, 1)
    q_vis[:, 0, 0] = 1  # 第一个物体已经被注视
    bbox = torch.rand(B, nq, 4)
    size = (20, 20)

    # 创建模型
    model = OCORSequentialRankingModule(
        embed_dim=256,
        num_heads=8,
        hidden_dim=1024,
        dropout_attn=0.0,
        dropout_ffn=0.0,
        num_blocks=2
    )

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(q, z, qpe, zpe, q_vis, bbox, size)
        if isinstance(output, tuple):
            sal, conf = output
            print(f"Saliency shape: {sal.shape}")
            print(f"Confidence shape: {conf.shape}")
        else:
            print(f"Output shape: {output.shape}")

    # 测试prioritization weights可视化
    weights = visualize_prioritization_weights(model, q, z)
    print(f"\nPrioritization weights shape: {weights.shape}")
    print(f"Object weights (sample): {weights[0, :3, 0]}")
    print(f"Context weights (sample): {weights[0, :3, 1]}")

    print("\nTest passed!")