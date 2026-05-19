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

        # Query hidden states captured from the output of the L2 blocks
        # (after the final norm, before the heads). Used as the relational
        # distillation anchors for CRRCD. Populated on every forward.
        self._captured_decoder_hs: torch.Tensor | None = None

    def _select_queries(
        self,
        query_mode: str,
        injected_queries: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve the object queries fed into the L2 blocks.

        query_mode:
            "student"  -> the model's own learnable queries (self.q).
            "teacher"  -> externally injected teacher queries.
            "general"  -> externally injected random/general queries.
        For "teacher"/"general" the injected tensor is (K, D) or
        (batch_size, K, D); K must equal self.num_q so the existing
        mask-prediction / masked-attention / annealing logic (which keys
        off self.num_q) keeps working unchanged.
        """
        if query_mode == "student":
            return self.q.weight[None, :, :].expand(batch_size, -1, -1)

        if query_mode not in ("teacher", "general"):
            raise ValueError(f"Unknown query_mode '{query_mode}'")
        if injected_queries is None:
            raise ValueError(
                f"query_mode='{query_mode}' requires injected_queries"
            )

        q = injected_queries
        if q.dim() == 2:
            q = q[None, :, :].expand(batch_size, -1, -1)
        assert q.shape[0] == batch_size and q.shape[-2] == self.num_q, (
            f"injected_queries must broadcast to ({batch_size}, {self.num_q}, D); "
            f"got {tuple(injected_queries.shape)}"
        )
        return q.to(dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        query_mode: str = "student",
        injected_queries: torch.Tensor | None = None,
    ):
        """Args:
            x: (B, T, 3, H, W) video clip.
            query_mode: "student" / "teacher" / "general" — see _select_queries.
            injected_queries: (K, D) or (B*T, K, D) queries for the non-student
                modes; ignored when query_mode == "student".

        Returns:
            mask_logits_per_layer: list of (B*T, K, H/4, W/4) tensors.
            class_logits_per_layer: list of (B*T, K, num_classes+1) tensors.

        Side effect:
            self._captured_decoder_hs is set to the (B*T, K, D) query hidden
            states at the output of the L2 blocks.
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

            # Inject queries at the start of L2 (mirrors EoMT). Depending on
            # query_mode these are the student's own queries, or externally
            # injected teacher / general queries.
            if i == total_blocks - self.num_blocks:
                queries = self._select_queries(
                    query_mode, injected_queries, x.shape[0], x.dtype, x.device
                )
                x = torch.cat((queries, x), dim=1)

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

        x_norm = self.encoder.backbone.norm(x)
        self._captured_decoder_hs = x_norm[:, : self.num_q, :]
        mask_logits, class_logits = self._predict(x_norm)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return mask_logits_per_layer, class_logits_per_layer
