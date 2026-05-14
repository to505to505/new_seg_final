
# Multi-frame count-consistency loss (L_num).
# Encourages temporally coherent object counts across the T frames
# of a video clip. Adapted from rfdetr_video/consistency.py.
# ---------------------------------------------------------------

import torch


def num_consistency_loss(
    pred_logits: torch.Tensor,
    threshold: float,
    soft_temp: float = 0.05,
    drop_no_object: bool = True,
) -> torch.Tensor:
    """Args:
        pred_logits: (B, T, Q, C+1) raw classification logits.
        threshold: confidence above which a query is counted as an object.
        soft_temp: temperature of the soft-count sigmoid surrogate.
        drop_no_object: if True, drop the last (no-object) channel before
            computing per-query foreground probability.

    Returns:
        Scalar tensor (zero when T < 2).
    """
    if pred_logits.dim() != 4:
        raise ValueError(
            f"pred_logits must be (B, T, Q, K), got {tuple(pred_logits.shape)}"
        )
    B, T, Q, _K = pred_logits.shape
    if T < 2:
        return pred_logits.new_zeros(())

    logits = pred_logits[..., :-1] if drop_no_object else pred_logits

    # Per-query foreground probability (max over real classes).
    p = logits.softmax(dim=-1).amax(dim=-1)  # (B, T, Q)

    soft_temp = max(float(soft_temp), 1e-6)
    soft_indicator = torch.sigmoid((p - float(threshold)) / soft_temp)
    n_t = soft_indicator.sum(dim=-1)  # (B, T)

    n_r = n_t.detach().median(dim=1, keepdim=True).values  # (B, 1)

    return (n_t - n_r).abs().mean()
