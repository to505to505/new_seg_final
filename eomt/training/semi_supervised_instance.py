# ---------------------------------------------------------------
# Semi-supervised instance segmentation module.
# Extends MaskClassificationInstance with EMA teacher,
# pseudo-label generation, and confidence-filtered unsupervised loss.
# ---------------------------------------------------------------

from copy import deepcopy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_instance import MaskClassificationInstance
from training.mask_classification_loss import MaskClassificationLoss


class SemiSupervisedInstance(MaskClassificationInstance):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        skeleton_coefficient: float = 0.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        eval_top_k_instances: int = 100,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        # Semi-supervised parameters
        ema_momentum_max: float = 0.996,
        conf_threshold: float = 0.95,
        unsupervised_weight: float = 1.0,
        score_threshold: float = 0.5,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            no_object_coefficient=no_object_coefficient,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            skeleton_coefficient=skeleton_coefficient,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            eval_top_k_instances=eval_top_k_instances,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ema_momentum_max = ema_momentum_max
        self.conf_threshold = conf_threshold
        self.unsupervised_weight = unsupervised_weight
        self.score_threshold = score_threshold

        # EMA teacher: deep copy of student, frozen, eval mode
        self.model_ema = deepcopy(self.network)
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)

        # Unsupervised criterion: same as supervised but no skeleton loss
        self.criterion_unsup = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
            skeleton_coefficient=0.0,
        )

    @torch.no_grad()
    def _update_ema(self):
        ema_ratio = min(1 - 1 / (self.global_step + 1), self.ema_momentum_max)
        for param, param_ema in zip(
            self.network.parameters(), self.model_ema.parameters()
        ):
            param_ema.data.mul_(ema_ratio).add_(param.data, alpha=1 - ema_ratio)
        for buf, buf_ema in zip(self.network.buffers(), self.model_ema.buffers()):
            buf_ema.data.mul_(ema_ratio).add_(buf.data, alpha=1 - ema_ratio)
        self.log("semi/ema_ratio", ema_ratio)

    @torch.no_grad()
    def _generate_pseudo_labels(self, imgs_weak: torch.Tensor) -> List[dict]:
        """Generate instance-level pseudo-labels from the EMA teacher."""
        self.model_ema.eval()
        x = imgs_weak / 255.0
        mask_logits_per_block, class_logits_per_block = self.model_ema(x)

        # Use last (most refined) block predictions
        mask_logits = mask_logits_per_block[-1]  # (B, Q, H/s, W/s)
        class_logits = class_logits_per_block[-1]  # (B, Q, C+1)

        # Upsample mask logits to img_size
        mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")

        B = mask_logits.shape[0]
        class_probs = class_logits.softmax(dim=-1)  # (B, Q, C+1)
        # Exclude no-object class (last class)
        class_scores, class_ids = class_probs[:, :, :-1].max(dim=-1)  # (B, Q)

        mask_probs = mask_logits.sigmoid()  # (B, Q, H, W)
        binary_masks = mask_probs > self.score_threshold  # (B, Q, H, W)

        # Mask score: average sigmoid probability over positive mask region
        mask_scores = (mask_probs * binary_masks.float()).flatten(2).sum(dim=2) / (
            binary_masks.flatten(2).sum(dim=2).float() + 1e-6
        )  # (B, Q)

        # Combined instance confidence
        instance_scores = class_scores * mask_scores  # (B, Q)

        pseudo_targets = []
        total_instances = 0
        for b in range(B):
            keep = instance_scores[b] >= self.conf_threshold
            if not keep.any():
                pseudo_targets.append(
                    {
                        "masks": torch.zeros(
                            0,
                            self.img_size[0],
                            self.img_size[1],
                            device=imgs_weak.device,
                            dtype=torch.float32,
                        ),
                        "labels": torch.zeros(
                            0, device=imgs_weak.device, dtype=torch.long
                        ),
                        "is_crowd": torch.zeros(
                            0, device=imgs_weak.device, dtype=torch.bool
                        ),
                    }
                )
                continue

            masks_b = binary_masks[b, keep].float()  # (N, H, W)
            labels_b = class_ids[b, keep]  # (N,)
            is_crowd_b = torch.zeros(
                masks_b.shape[0], device=imgs_weak.device, dtype=torch.bool
            )
            total_instances += masks_b.shape[0]

            pseudo_targets.append(
                {
                    "masks": masks_b,
                    "labels": labels_b,
                    "is_crowd": is_crowd_b,
                }
            )

        self.log("semi/num_pseudo_instances", float(total_instances) / max(B, 1))
        return pseudo_targets

    def training_step(self, batch, batch_idx):
        batch_l = batch["labeled"]
        batch_u = batch["unlabeled"]

        imgs_l, targets_l = batch_l
        imgs_u_weak, imgs_u_strong = batch_u

        # ---- Supervised forward (same as parent) ----
        mask_logits_per_block, class_logits_per_block = self(imgs_l)

        losses_sup = {}
        for i, (mask_logits, class_logits) in enumerate(
            zip(mask_logits_per_block, class_logits_per_block)
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets_l,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_sup.update(losses)

        loss_supervised = self.criterion.loss_total(losses_sup, self.log)

        # ---- Pseudo-label generation from EMA teacher ----
        pseudo_targets = self._generate_pseudo_labels(imgs_u_weak)

        # ---- Unsupervised forward ----
        has_pseudo = any(t["masks"].shape[0] > 0 for t in pseudo_targets)

        if has_pseudo:
            mask_logits_u_per_block, class_logits_u_per_block = self(imgs_u_strong)

            losses_unsup = {}
            for i, (mask_logits_u, class_logits_u) in enumerate(
                zip(mask_logits_u_per_block, class_logits_u_per_block)
            ):
                losses_u = self.criterion_unsup(
                    masks_queries_logits=mask_logits_u,
                    class_queries_logits=class_logits_u,
                    targets=pseudo_targets,
                )
                block_postfix = self.block_postfix(i)
                losses_u = {
                    f"unsup_{key}{block_postfix}": value
                    for key, value in losses_u.items()
                }
                losses_unsup.update(losses_u)

            loss_unsupervised = self.criterion_unsup.loss_total(
                losses_unsup, self._log_unsup
            )
        else:
            loss_unsupervised = torch.tensor(0.0, device=self.device)

        self.log("semi/loss_supervised", loss_supervised)
        self.log("semi/loss_unsupervised", loss_unsupervised)

        loss_total = loss_supervised + self.unsupervised_weight * loss_unsupervised
        self.log("semi/loss_total", loss_total, prog_bar=True)

        # ---- EMA update ----
        self._update_ema()

        return loss_total

    def _log_unsup(self, name, value, **kwargs):
        """Redirect unsupervised loss logging with a prefix."""
        self.log(name.replace("losses/train_", "losses/unsup_"), value, **kwargs)
