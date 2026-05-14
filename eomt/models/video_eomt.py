# ---------------------------------------------------------------
# Video-EoMT: Encoder-only Mask Transformer with Early Temporal
# Fusion (ETF) injected between the L1 (encoder) and L2 (joint
# encoder/decoder) ViT stages. Inherits the entire EoMT mask head,
# query injection, and mask annealing logic; only the forward pass
# is overridden to accept (B, T, 3, H, W) inputs.
# ---------------------------------------------------------------

import torch
import torch.nn as nn

from models.eomt import EoMT
from models.etf import EarlyTemporalFusion


class VideoEoMT(EoMT):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        num_q: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
        etf_heads: int = 8,
        etf_dropout: float = 0.0,
    ):
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            masked_attn_enabled=masked_attn_enabled,
        )
        self.etf = EarlyTemporalFusion(
            d_model=self.encoder.backbone.embed_dim,
            n_heads=etf_heads,
            dropout=etf_dropout,
        )

    def forward(self, x: torch.Tensor):
        """Args:
            x: (B, T, 3, H, W) video clip.

        Returns:
            mask_logits_per_layer: list of (B*T, K, H/4, W/4) tensors.
            class_logits_per_layer: list of (B*T, K, num_classes+1) tensors.
        """
        if x.dim() == 4:
            # Allow plain 2D inference passthrough.
            B, T = x.shape[0], 1
        else:
            B, T = x.shape[0], x.shape[1]
            x = x.reshape(B * T, *x.shape[2:])

        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)
        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        total_blocks = len(self.encoder.backbone.blocks)
        l1 = total_blocks - self.num_blocks
        prefix = self.encoder.backbone.num_prefix_tokens

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []
        etf_done = False

        for i, block in enumerate(self.encoder.backbone.blocks):
            # Inject ETF between L1 and L2 (after the L1 blocks finished, before L2 starts).
            if i == l1 and not etf_done:
                if T > 1:
                    prefix_tokens = x[:, :prefix, :]
                    patch_tokens = x[:, prefix:, :]
                    patch_tokens = self.etf(patch_tokens, B, T)
                    x = torch.cat([prefix_tokens, patch_tokens], dim=1)
                etf_done = True

            # Inject queries at the start of L2 (mirrors EoMT).
            if i == total_blocks - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= total_blocks - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)
            else:
                x = x + attn_out

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)
            else:
                x = x + mlp_out

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return mask_logits_per_layer, class_logits_per_layer
