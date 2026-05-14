# ---------------------------------------------------------------
# Early Temporal Fusion (ETF) module.
# Lightweight per-token temporal self-attention applied between the
# L1 and L2 ViT stages of Video-EoMT. Initialised as an identity
# mapping (zero out_proj) so a checkpoint trained on single frames
# remains numerically equivalent at iteration 0.
# ---------------------------------------------------------------

import torch
import torch.nn as nn


class EarlyTemporalFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Mix patch tokens across the temporal axis.

        Args:
            x: (B*T, N, D) patch tokens (prefix tokens stripped beforehand).
            B: batch size.
            T: temporal window size.

        Returns:
            Tensor of the same shape (B*T, N, D).
        """
        BT, N, D = x.shape
        assert BT == B * T, f"Expected B*T={B*T}, got {BT}"

        # (B*T, N, D) -> (B, T, N, D) -> (B, N, T, D) -> (B*N, T, D)
        x_seq = x.view(B, T, N, D).permute(0, 2, 1, 3).reshape(B * N, T, D)
        x_n = self.norm(x_seq)
        attn_out, _ = self.attn(x_n, x_n, x_n, need_weights=False)
        x_seq = x_seq + attn_out

        # (B*N, T, D) -> (B, N, T, D) -> (B, T, N, D) -> (B*T, N, D)
        x_out = x_seq.view(B, N, T, D).permute(0, 2, 1, 3).reshape(B * T, N, D)
        return x_out
