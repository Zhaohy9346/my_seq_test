"""
Boundary-Guided FPN for Camouflaged Object Detection
基于边界引导的特征金字塔网络，专门用于伪装目标检测

主要创新点：
1. 边界注意力模块 (Boundary Attention Module)
2. 多尺度特征增强 (Multi-Scale Feature Enhancement)
3. 渐进式特征细化 (Progressive Feature Refinement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.config import configurable
from seqrank.component import init_weights_, LayerNorm2D
from seqrank.neck.registry import NECK_HEAD


class BoundaryAttention(nn.Module):
    """
    边界注意力模块
    通过提取边界特征来增强目标-背景区分能力
    """

    def __init__(self, dim=256):
        super().__init__()
        # Sobel算子用于边界检测
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3))

        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).float().view(1, 1, 3, 3))

        # 边界特征处理
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.GELU(),
            LayerNorm2D(dim)
        )

        # 注意力生成
        self.attention_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )

    def detect_boundaries(self, x):
        """使用Sobel算子检测边界"""
        B, C, H, W = x.shape

        # 对每个通道分别应用Sobel算子
        sobel_x = self.sobel_x.repeat(C, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(C, 1, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1, groups=C)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=C)

        # 计算梯度幅值
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return boundary

    def forward(self, x):
        """
        Args:
            x: B, C, H, W
        Returns:
            enhanced_x: B, C, H, W (边界增强后的特征)
            boundary_attn: B, 1, H, W (边界注意力图)
        """
        # 检测边界
        boundary_feat = self.detect_boundaries(x)

        # 融合原始特征和边界特征
        combined = torch.cat([x, boundary_feat], dim=1)
        boundary_enhanced = self.boundary_conv(combined)

        # 生成边界注意力
        boundary_attn = self.attention_conv(boundary_enhanced)

        # 应用注意力
        enhanced_x = x * boundary_attn + x

        return enhanced_x, boundary_attn


class ContextualModule(nn.Module):
    """
    上下文感知模块
    捕获全局和局部上下文信息
    """

    def __init__(self, dim=256):
        super().__init__()
        # 多尺度空洞卷积
        self.dilated_conv1 = nn.Conv2d(dim, dim // 4, 3, padding=1, dilation=1)
        self.dilated_conv2 = nn.Conv2d(dim, dim // 4, 3, padding=2, dilation=2)
        self.dilated_conv3 = nn.Conv2d(dim, dim // 4, 3, padding=3, dilation=3)
        self.dilated_conv4 = nn.Conv2d(dim, dim // 4, 3, padding=4, dilation=4)

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            LayerNorm2D(dim)
        )

    def forward(self, x):
        """
        Args:
            x: B, C, H, W
        Returns:
            context: B, C, H, W
        """
        d1 = self.dilated_conv1(x)
        d2 = self.dilated_conv2(x)
        d3 = self.dilated_conv3(x)
        d4 = self.dilated_conv4(x)

        # 拼接多尺度特征
        multi_scale = torch.cat([d1, d2, d3, d4], dim=1)
        context = self.fusion(multi_scale)

        return context


class ProgressiveRefinement(nn.Module):
    """
    渐进式特征细化模块
    逐步细化高低层特征的融合
    """

    def __init__(self, dim=256):
        super().__init__()
        # 高层特征处理
        self.high_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            LayerNorm2D(dim)
        )

        # 低层特征处理
        self.low_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            LayerNorm2D(dim)
        )

        # 融合权重
        self.weight_gen = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, 2, 1),
            nn.Softmax(dim=1)
        )

        # 最终融合
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            LayerNorm2D(dim)
        )

    def forward(self, high_feat, low_feat):
        """
        Args:
            high_feat: B, C, h, w (高层特征，分辨率低)
            low_feat: B, C, H, W (低层特征，分辨率高)
        Returns:
            refined: B, C, H, W
        """
        # 上采样高层特征
        high_up = F.interpolate(
            high_feat,
            size=low_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 处理特征
        high_processed = self.high_conv(high_up)
        low_processed = self.low_conv(low_feat)

        # 自适应权重融合
        combined = torch.cat([high_processed, low_processed], dim=1)
        weights = self.weight_gen(combined)  # B, 2, H, W

        # 加权融合
        refined = (
                high_processed * weights[:, 0:1, :, :] +
                low_processed * weights[:, 1:2, :, :]
        )

        # 最终细化
        refined = self.final_conv(refined)

        return refined


class EnhancedFPNLayer(nn.Module):
    """
    增强的FPN层，整合边界引导和上下文感知
    """

    def __init__(self, dim=256):
        super().__init__()
        # 边界注意力
        self.boundary_attn = BoundaryAttention(dim)

        # 上下文模块
        self.context_module = ContextualModule(dim)

        # 渐进式细化
        self.progressive_refine = ProgressiveRefinement(dim)

        # 残差连接
        self.residual_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            LayerNorm2D(dim)
        )

    def forward(self, high_feat, low_feat):
        """
        Args:
            high_feat: B, C, h, w
            low_feat: B, C, H, W
        Returns:
            output: B, C, H, W
        """
        # 1. 低层特征边界增强
        low_enhanced, low_boundary = self.boundary_attn(low_feat)

        # 2. 高层特征上下文增强
        high_context = self.context_module(high_feat)

        # 3. 渐进式融合
        refined = self.progressive_refine(high_context, low_enhanced)

        # 4. 残差连接
        residual = self.residual_conv(low_feat)
        output = refined + residual

        return output


@NECK_HEAD.register()
class BoundaryGuidedFPN(nn.Module):
    """
    边界引导的特征金字塔网络
    专门为伪装目标检测设计
    """

    @configurable
    def __init__(
            self,
            dim=256,
            feat_dims=(192, 384, 768, 1536),  # Swin-L的特征维度
            feat_keys=["res2", "res3", "res4", "res5"]
    ):
        super().__init__()
        self.feat_keys = feat_keys

        # Lateral连接 - 将不同维度统一到dim
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_in, dim, 1),
                nn.GELU(),
                LayerNorm2D(dim)
            ) for d_in in feat_dims
        ])

        # 增强的FPN层
        self.enhanced_fpn_layers = nn.ModuleList([
            EnhancedFPNLayer(dim=dim)
            for _ in range(len(feat_dims) - 1)
        ])

        # 每层的边界注意力（用于输出）
        self.output_boundary_attns = nn.ModuleList([
            BoundaryAttention(dim=dim)
            for _ in range(len(feat_dims))
        ])

        # 最终特征增强
        self.final_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GELU(),
                LayerNorm2D(dim)
            ) for _ in range(len(feat_dims))
        ])

        init_weights_(self.lateral_convs)
        init_weights_(self.final_enhance)

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.MODEL.COMMON.EMBED_DIM,
            "feat_dims": cfg.MODEL.BACKBONE.NUM_FEATURES,
            "feat_keys": cfg.MODEL.BACKBONE.FEATURE_KEYS
        }

    def forward(self, feats):
        """
        Args:
            feats: dict with keys as self.feat_keys
                res2: B, C2, H/4, W/4
                res3: B, C3, H/8, W/8
                res4: B, C4, H/16, W/16
                res5: B, C5, H/32, W/32
        Returns:
            feats: dict with same keys, all with dim channels
        """
        # 1. Lateral连接，统一维度
        lateral_feats = [
            layer(feats[k])
            for layer, k in zip(self.lateral_convs, self.feat_keys)
        ]

        # 2. 自顶向下的特征融合（从高层到低层）
        # 反转顺序：res5 -> res4 -> res3 -> res2
        lateral_feats_reversed = lateral_feats[::-1]  # [res5, res4, res3, res2]

        refined_feats = [lateral_feats_reversed[0]]  # 从最高层开始

        for i, fpn_layer in enumerate(self.enhanced_fpn_layers):
            high_feat = refined_feats[-1]  # 上一层的输出
            low_feat = lateral_feats_reversed[i + 1]  # 下一层的lateral特征

            # 使用增强的FPN层融合
            refined = fpn_layer(high_feat, low_feat)
            refined_feats.append(refined)

        # 3. 恢复原始顺序 [res2, res3, res4, res5]
        refined_feats = refined_feats[::-1]

        # 4. 对每层应用边界增强和最终细化
        output_feats = []
        for i, (feat, boundary_attn, enhance) in enumerate(
                zip(refined_feats, self.output_boundary_attns, self.final_enhance)
        ):
            # 边界增强
            enhanced, _ = boundary_attn(feat)
            # 最终细化
            final = enhance(enhanced)
            output_feats.append(final)

        # 5. 构建输出字典
        output = dict(
            (k, v)
            for k, v in zip(self.feat_keys, output_feats)
        )

        return output


@NECK_HEAD.register()
class HybridBoundaryFPN(nn.Module):
    """
    混合边界引导FPN
    结合原始FPN的简洁性和边界增强的有效性
    适合快速实验和调优
    """

    @configurable
    def __init__(
            self,
            dim=256,
            feat_dims=(192, 384, 768, 1536),
            feat_keys=["res2", "res3", "res4", "res5"],
            use_boundary=True,
            use_context=True
    ):
        super().__init__()
        self.feat_keys = feat_keys
        self.use_boundary = use_boundary
        self.use_context = use_context

        # Lateral连接
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_in, dim, 1),
                nn.GELU(),
                LayerNorm2D(dim)
            ) for d_in in feat_dims
        ])

        # 基础FPN层
        self.base_fpn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.GELU(),
                LayerNorm2D(dim)
            ) for _ in range(len(feat_dims) - 1)
        ])

        # 可选的边界注意力
        if use_boundary:
            self.boundary_attns = nn.ModuleList([
                BoundaryAttention(dim=dim)
                for _ in range(len(feat_dims))
            ])

        # 可选的上下文模块
        if use_context:
            self.context_modules = nn.ModuleList([
                ContextualModule(dim=dim)
                for _ in range(len(feat_dims))
            ])

        init_weights_(self.lateral_convs)
        init_weights_(self.base_fpn_layers)

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.MODEL.COMMON.EMBED_DIM,
            "feat_dims": cfg.MODEL.BACKBONE.NUM_FEATURES,
            "feat_keys": cfg.MODEL.BACKBONE.FEATURE_KEYS,
            "use_boundary": cfg.MODEL.NECK.get("USE_BOUNDARY", True),
            "use_context": cfg.MODEL.NECK.get("USE_CONTEXT", True)
        }

    def forward(self, feats):
        # Lateral连接
        lateral_feats = [
                            layer(feats[k])
                            for layer, k in zip(self.lateral_convs, self.feat_keys)
                        ][::-1]  # 反转为高->低

        # 自顶向下融合
        output_feats = [lateral_feats[0]]
        for i, fpn_layer in enumerate(self.base_fpn_layers):
            high_feat = output_feats[-1]
            low_feat = lateral_feats[i + 1]

            # 上采样并相加
            high_up = F.interpolate(
                high_feat,
                size=low_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            fused = fpn_layer(high_up + low_feat)
            output_feats.append(fused)

        # 恢复顺序
        output_feats = output_feats[::-1]

        # 可选增强
        enhanced_feats = []
        for i, feat in enumerate(output_feats):
            enhanced = feat

            # 边界增强
            if self.use_boundary:
                enhanced, _ = self.boundary_attns[i](enhanced)

            # 上下文增强
            if self.use_context:
                context = self.context_modules[i](enhanced)
                enhanced = enhanced + context

            enhanced_feats.append(enhanced)

        return dict(zip(self.feat_keys, enhanced_feats))