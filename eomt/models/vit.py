# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import Optional
import torch
import torch.nn as nn

import timm
from transformers import AutoModel


class _FusedQKV(nn.Module):
    """Fuses separate Q, K, V linear layers into a single callable matching timm's fused qkv."""

    def __init__(self, query, key, value):
        super().__init__()
        self.query = query
        self.key = key
        self.value = value

    def forward(self, x):
        return torch.cat([self.query(x), self.key(x), self.value(x)], dim=-1)


class _TimmAttentionAdapter(nn.Module):
    """Adapts HF ViT self-attention to timm's attention interface."""

    def __init__(self, hf_self_attention, hf_attention_output):
        super().__init__()
        self.qkv = _FusedQKV(
            hf_self_attention.query, hf_self_attention.key, hf_self_attention.value
        )
        self.num_heads = hf_self_attention.num_attention_heads
        self.head_dim = hf_self_attention.attention_head_size
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.proj = hf_attention_output.dense
        self.proj_drop = hf_attention_output.dropout
        self.attn_drop = nn.Dropout(hf_self_attention.dropout_prob)
        self.fused_attn = True
        self.scale = hf_self_attention.attention_head_size**-0.5


class _DINOv1MLP(nn.Module):
    """Wraps HF ViT intermediate+output layers to match timm's MLP interface."""

    def __init__(self, intermediate, output):
        super().__init__()
        self.fc1 = intermediate.dense
        self.act = intermediate.intermediate_act_fn
        self.fc2 = output.dense
        self.drop = output.dropout

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _InterpolatePosEmbedWrapper(nn.Module):
    """Wraps HF ViT embeddings to always interpolate position embeddings for arbitrary input sizes."""

    def __init__(self, embeddings, patch_size, grid_size):
        super().__init__()
        self._embeddings = embeddings
        self.config = embeddings.config
        self.patch_size = patch_size
        self.grid_size = grid_size

    def forward(self, x, **kwargs):
        return self._embeddings(x, interpolate_pos_encoding=True, **kwargs)


class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        if backbone_name.startswith("timm/"):
            # timm model from HuggingFace Hub
            self.backbone = timm.create_model(
                "hf_hub:" + backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                num_classes=0,
            )
        elif "/" in backbone_name:
            self.backbone = self.transformers_to_timm(
                AutoModel.from_pretrained(
                    backbone_name,
                ),
                img_size,
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=ckpt_path is None,
                img_size=img_size,
                patch_size=patch_size,
                num_classes=0,
            )

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def transformers_to_timm(self, backbone, img_size: tuple[int, int]):
        patch_size = backbone.embeddings.config.patch_size
        grid_size = (
            img_size[0] // patch_size,
            img_size[1] // patch_size,
        )
        backbone.embed_dim = backbone.embeddings.config.hidden_size

        is_dinov1 = backbone.config.model_type == "vit"

        if is_dinov1:
            backbone.num_prefix_tokens = 1  # CLS only, no register tokens
            backbone.blocks = backbone.encoder.layer
            backbone.norm = backbone.layernorm

            backbone.patch_embed = _InterpolatePosEmbedWrapper(
                backbone.embeddings, (patch_size, patch_size), grid_size
            )

            for block in backbone.blocks:
                block.attn = _TimmAttentionAdapter(
                    block.attention.attention, block.attention.output
                )
                block.norm1 = block.layernorm_before
                block.norm2 = block.layernorm_after
                block.mlp = _DINOv1MLP(block.intermediate, block.output)
                block.ls1 = nn.Identity()
                block.ls2 = nn.Identity()
                del (
                    block.attention,
                    block.intermediate,
                    block.output,
                    block.layernorm_before,
                    block.layernorm_after,
                )

            del backbone.encoder, backbone.layernorm, backbone.embeddings
            if hasattr(backbone, "pooler"):
                del backbone.pooler
        else:
            backbone.patch_embed = backbone.embeddings
            backbone.patch_embed.patch_size = (patch_size, patch_size)
            backbone.patch_embed.grid_size = grid_size
            backbone.num_prefix_tokens = (
                backbone.patch_embed.config.num_register_tokens + 1
            )
            backbone.blocks = backbone.layer

            del (
                backbone.patch_embed.mask_token,
                backbone.embeddings,
                backbone.layer,
            )

        return backbone
