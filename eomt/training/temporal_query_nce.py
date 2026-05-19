# ---------------------------------------------------------------
# Temporal per-query InfoNCE / NT-Xent loss for Video-EoMT.
#
# Given the final-block query hidden states captured at L2 output,
# for each clip we treat (slot k at frame t) and (slot k at frame t+1)
# as a positive pair, and (slot k at t, slot k' != k at t+1) as
# negatives. This encourages each query slot to keep a temporally
# stable post-cross-attention representation across the T-frame
# window without constraining pixel-level mask geometry — i.e.
# motion-friendly self-supervised query identity stability.
#
# "Dead" slots (background queries with low foreground confidence in
# every frame of the clip) are masked out: forcing background-noise
# embeddings into temporally-stable clusters would be pure noise.
# ---------------------------------------------------------------

import torch
import torch.nn.functional as F


def temporal_query_nce_loss(
    hs: torch.Tensor,        # (B, T, K, D) query hidden states after L2
    fg_conf: torch.Tensor,   # (B, T, K) per-query foreground confidence
    temperature: float = 0.1,
    alive_threshold: float = 0.2,
    symmetric: bool = True,
) -> torch.Tensor:
    """InfoNCE on per-slot query embeddings across adjacent frames.

    Args:
        hs: final-block query hidden states (B, T, K, D).
        fg_conf: per-query max foreground softmax probability (B, T, K).
        temperature: NT-Xent temperature (smaller -> sharper contrastive).
        alive_threshold: a slot is "alive" if its max fg_conf across the
            T-frame window exceeds this value (per clip, per slot).
        symmetric: if True, also use frame t+1 as anchor against frame t.

    Returns:
        Scalar tensor (zero when T < 2 or no slots are alive).
    """
    if hs.dim() != 4:
        raise ValueError(
            f"hs must be (B, T, K, D), got {tuple(hs.shape)}"
        )
    B, T, K, D = hs.shape
    if T < 2 or K < 2:
        return hs.new_zeros(())

    hs = F.normalize(hs, dim=-1)
    tau = max(float(temperature), 1e-6)

    # Per-clip per-slot "alive" indicator: detached, broadcastable over pairs.
    alive = (fg_conf.amax(dim=1) > float(alive_threshold)).float().detach()  # (B, K)
    alive_sum = alive.sum().clamp(min=1.0)

    target = torch.arange(K, device=hs.device).expand(B, K).reshape(-1)  # (B*K,)

    def _ce_pair(anchor: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        # anchor, candidates: (B, K, D)
        sim = torch.einsum("bkd,bjd->bkj", anchor, candidates) / tau  # (B, K, K)
        ce = F.cross_entropy(sim.reshape(-1, K), target, reduction="none").view(B, K)
        return (ce * alive).sum() / alive_sum

    total = hs.new_zeros(())
    n_pairs = 0
    for t in range(T - 1):
        a, p = hs[:, t], hs[:, t + 1]
        loss_fwd = _ce_pair(a, p)
        if symmetric:
            loss_bwd = _ce_pair(p, a)
            total = total + 0.5 * (loss_fwd + loss_bwd)
        else:
            total = total + loss_fwd
        n_pairs += 1

    return total / max(n_pairs, 1)
