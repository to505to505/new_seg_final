# ---------------------------------------------------------------
# Knowledge-distillation losses for Video-EoMT.
#
# Instance segmentation only — no bounding-box regression. The
# distillation signal is:
#   * class_kl_loss  — KL-divergence on the per-query class logits.
#   * mask_kd_loss   — Dice + BCE between student / teacher mask logits.
#   * CRRCDLoss      — Cross-Resolution Relational Contrastive
#                      Distillation on the L2 query hidden states.
#
# The CRRCD math is ported verbatim from the legacy RF-DETR codebase
# (rfdetr_temporal/distill/crrcd.py); only the docstring is trimmed.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Specific KD — per-query class / mask distillation, weighted by a per-query
# foreground (or background) weight derived from the teacher.
# ---------------------------------------------------------------------------

def class_kl_loss(
    student_logits: torch.Tensor,   # (B, K, C+1)
    teacher_logits: torch.Tensor,   # (B, K, C+1)
    weight: torch.Tensor,           # (B, K)
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-query KL( p_teacher || p_student ) over the softmax class
    distribution (including the no-object channel), reduced as a
    weighted mean ``Σ w·kl / Σ w`` so the scale is independent of how
    many queries are "alive". Hinton ``T**2`` rescaling is applied."""
    T = max(float(temperature), _EPS)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    log_p_t = F.log_softmax(teacher_logits / T, dim=-1).detach()
    p_t = log_p_t.exp()
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1)          # (B, K)
    w_sum = weight.sum().clamp(min=_EPS)
    return (weight * kl).sum() / w_sum * (T * T)


def mask_kd_loss(
    student_mask_logits: torch.Tensor,   # (B, K, h, w)
    teacher_mask_logits: torch.Tensor,   # (B, K, h, w)
    weight: torch.Tensor,                # (B, K)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft Dice + BCE between student and teacher per-query mask logits.

    The teacher's sigmoid masks are used as soft targets. Both terms are
    reduced per-query then combined as a weighted mean over queries.
    Returns ``(loss_bce, loss_dice)``.
    """
    if student_mask_logits.shape[-2:] != teacher_mask_logits.shape[-2:]:
        teacher_mask_logits = F.interpolate(
            teacher_mask_logits,
            size=student_mask_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    t_prob = teacher_mask_logits.sigmoid().detach()

    bce = F.binary_cross_entropy_with_logits(
        student_mask_logits, t_prob, reduction="none"
    ).flatten(2).mean(dim=-1)                              # (B, K)

    s_flat = student_mask_logits.sigmoid().flatten(2)      # (B, K, h*w)
    t_flat = t_prob.flatten(2)
    num = 2.0 * (s_flat * t_flat).sum(dim=-1)
    den = s_flat.sum(dim=-1) + t_flat.sum(dim=-1)
    dice = 1.0 - (num + 1.0) / (den + 1.0)                 # (B, K)

    w_sum = weight.sum().clamp(min=_EPS)
    loss_bce = (weight * bce).sum() / w_sum
    loss_dice = (weight * dice).sum() / w_sum
    return loss_bce, loss_dice


# ---------------------------------------------------------------------------
# CRRCD — Cross-Resolution Relational Contrastive Distillation.
# Ported from rfdetr_temporal/distill/crrcd.py.
# ---------------------------------------------------------------------------

class _RelationMLP(nn.Module):
    """v = W2 · ReLU(W1 (e_i − e_j))."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        return self.net(diff)


class CRRCDLoss(nn.Module):
    """Cross-Resolution Relational Contrastive Distillation loss.

    Operates on the L2 query hidden states captured from both the frozen
    teacher and the trainable student. Two trainable Feature Relation
    Modules model teacher-teacher and teacher-student slot relations; a
    sigmoid-NCE critic forces the cross relation to mimic the teacher
    reference at matching slot pairs. ``e_t`` is detached so no gradient
    reaches the teacher; ``e_s`` carries the graph back into the student.
    """

    def __init__(
        self,
        hidden_dim: int,
        relation_dim: int,
        frm_hidden_dim: int,
        num_fg: int,
        num_bg: int,
        num_negatives: int,
        temperature: float,
    ):
        super().__init__()
        self.F_t = _RelationMLP(hidden_dim, frm_hidden_dim, relation_dim)
        self.F_ts = _RelationMLP(hidden_dim, frm_hidden_dim, relation_dim)
        self.K_fg = int(num_fg)
        self.K_bg = int(num_bg)
        self.n_neg = int(num_negatives)
        self.tau = float(temperature)

    def forward(
        self,
        teacher_hs: torch.Tensor,   # (B, Q, D)  detached
        student_hs: torch.Tensor,   # (B, Q, D)  with grad
        weights: torch.Tensor,      # (B, Q)     teacher max-fg confidence
    ) -> torch.Tensor:
        assert teacher_hs.shape == student_hs.shape, (
            f"shape mismatch: teacher_hs {tuple(teacher_hs.shape)} vs "
            f"student_hs {tuple(student_hs.shape)}"
        )
        assert weights.shape[:2] == teacher_hs.shape[:2], (
            f"weights {tuple(weights.shape)} must match (B, Q) of hs"
        )

        # Belt & braces — never let any gradient reach the teacher.
        e_t = teacher_hs.detach()
        e_s = student_hs

        B, Q, D = e_t.shape
        K_fg = min(self.K_fg, Q)
        K_bg = min(self.K_bg, Q)
        if K_fg == 0 or K_bg < 2:
            return e_s.new_zeros(())

        # Top-K_fg by foreground weight; bottom-K_bg by foreground weight.
        fg_idx = weights.topk(K_fg, dim=1).indices                      # (B, K_fg)
        bg_idx = weights.topk(K_bg, dim=1, largest=False).indices       # (B, K_bg)

        def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))

        et_fg = gather(e_t, fg_idx)        # (B, K_fg, D)
        et_bg = gather(e_t, bg_idx)        # (B, K_bg, D)
        es_bg = gather(e_s, bg_idx)        # (B, K_bg, D)  — carries grad

        # Pairwise differences  (B, K_fg, K_bg, D).
        diff_t = et_fg.unsqueeze(2) - et_bg.unsqueeze(1)
        diff_ts = et_fg.unsqueeze(2) - es_bg.unsqueeze(1)

        v_t = self.F_t(diff_t)             # (B, K_fg, K_bg, R)
        v_ts = self.F_ts(diff_ts)          # (B, K_fg, K_bg, R)

        v_t_n = F.normalize(v_t, dim=-1)
        v_ts_n = F.normalize(v_ts, dim=-1)

        # Full similarity tensor: (B, K_fg, K_bg, K_bg) — last two dims (j, k).
        sim = torch.einsum("bijd,bikd->bijk", v_t_n, v_ts_n) / max(self.tau, 1e-6)

        device = sim.device
        eye = torch.eye(K_bg, device=device, dtype=torch.bool)
        eye = eye.view(1, 1, K_bg, K_bg)

        # Positive term: diagonal (j == k).
        pos_sim = torch.diagonal(sim, dim1=-2, dim2=-1)                # (B, K_fg, K_bg)
        log_h_pos = F.logsigmoid(pos_sim)

        # Negative term: log(1 - σ(s)) = logsigmoid(-s); zero out diagonal.
        log_one_minus_h_neg = F.logsigmoid(-sim).masked_fill(eye, 0.0)

        if 0 < self.n_neg < (K_bg - 1):
            rand = torch.rand_like(sim).masked_fill(eye, -1.0)
            sel = rand.topk(self.n_neg, dim=-1).indices
            neg_term = log_one_minus_h_neg.gather(-1, sel).sum(dim=-1)
        else:
            neg_term = log_one_minus_h_neg.sum(dim=-1)

        loss = -(log_h_pos.mean() + neg_term.mean())
        return loss
