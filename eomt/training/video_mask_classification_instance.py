# ---------------------------------------------------------------
# Video Mask Classification (Instance) training module.
# Supervises the central frame of each clip with the standard EoMT
# instance-segmentation losses, while applying a temporal count-
# consistency loss (L_num) over the full T-frame window so the ETF
# module learns to share information across time.
# ---------------------------------------------------------------

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2.functional import pad

from training.mask_classification_instance import MaskClassificationInstance
from training.num_consistency_loss import num_consistency_loss


class VideoMaskClassificationInstance(MaskClassificationInstance):
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
        connectivity_coefficient: float = 0.0,
        lcr_weight: float = 0.2,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        eval_top_k_instances: int = 100,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        consistency_weight: float = 0.5,
        consistency_threshold: float = 0.3,
        consistency_soft_temp: float = 0.05,
        test_results_filename: str = "video_test_results.txt",
        solo_only: bool = False,
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
            connectivity_coefficient=connectivity_coefficient,
            lcr_weight=lcr_weight,
            mask_thresh=mask_thresh,
            overlap_thresh=overlap_thresh,
            eval_top_k_instances=eval_top_k_instances,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            test_results_filename=test_results_filename,
            solo_only=solo_only,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.consistency_weight = consistency_weight
        self.consistency_threshold = consistency_threshold
        self.consistency_soft_temp = consistency_soft_temp

    def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head):
        # Drop ETF keys from missing list: ETF is a new module that does not
        # exist in 2D EoMT checkpoints; it is initialised to identity (zero
        # out_proj), so absence in the source ckpt is safe.
        # Drop criterion.* from unexpected list: the criterion is constructed
        # only later in MaskClassificationInstance.__init__, after the parent's
        # _load_ckpt call, so it is normal for these keys to look unexpected.
        filtered_missing = [
            k for k in incompatible_keys.missing_keys
            if not k.startswith("network.etf.")
        ]
        filtered_unexpected = [
            k for k in incompatible_keys.unexpected_keys
            if not k.startswith("criterion.")
        ]
        if filtered_missing or filtered_unexpected:
            from torch.nn.modules.module import _IncompatibleKeys
            filtered = _IncompatibleKeys(filtered_missing, filtered_unexpected)
            super()._raise_on_incompatible(filtered, load_ckpt_class_head)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _central_idx(T: int) -> int:
        return T // 2

    def _slice_central(self, tensor_bt: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Reshape (B*T, ...) -> (B, T, ...) and slice central frame -> (B, ...)."""
        rest = tensor_bt.shape[1:]
        return tensor_bt.view(B, T, *rest)[:, self._central_idx(T)]

    def _reshape_bt(self, tensor_bt: torch.Tensor, B: int, T: int) -> torch.Tensor:
        rest = tensor_bt.shape[1:]
        return tensor_bt.view(B, T, *rest)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        imgs, targets = batch  # imgs: (B, T, 3, H, W); targets: list[dict] of length B
        B, T = imgs.shape[0], imgs.shape[1]

        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            # Slice central frame for the standard EoMT instance loss.
            central_mask = self._slice_central(mask_logits, B, T)
            central_class = self._slice_central(class_logits, B, T)

            losses = self.criterion(
                masks_queries_logits=central_mask,
                class_queries_logits=central_class,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        total = self.criterion.loss_total(losses_all_blocks, self.log)

        # L_num on the final block's class logits over all T frames.
        if self.consistency_weight > 0 and T > 1:
            final_class_logits = class_logits_per_block[-1]  # (B*T, K, C+1)
            final_class_logits = self._reshape_bt(final_class_logits, B, T)
            l_num = num_consistency_loss(
                final_class_logits,
                threshold=self.consistency_threshold,
                soft_temp=self.consistency_soft_temp,
            )
            self.log("loss_num", l_num, on_step=True, prog_bar=False)
            total = total + self.consistency_weight * l_num

        return total

    # ------------------------------------------------------------------
    # Evaluation (central-frame only)
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _resize_and_pad_clips(self, clips):
        """Apply the same per-frame resize+pad that 2D eval uses, but
        operate per-clip so each clip becomes a (T, 3, Himg, Wimg) tensor.
        Returns: stacked (B, T, 3, Himg, Wimg) tensor and list of original
        per-clip frame sizes (taken from the central frame).
        """
        from PIL import Image
        import numpy as np

        out, central_sizes = [], []
        for clip in clips:  # clip: (T, 3, H, W)
            T_, _, H, W = clip.shape
            new_h, new_w = self.scale_img_size_instance_panoptic((H, W))
            frames = []
            for t in range(T_):
                img = clip[t]
                pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
                pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
                resized_img = (
                    torch.from_numpy(np.array(pil_img))
                    .permute(2, 0, 1)
                    .to(img.device)
                )
                pad_h = max(0, self.img_size[-2] - resized_img.shape[-2])
                pad_w = max(0, self.img_size[-1] - resized_img.shape[-1])
                padded_img = pad(resized_img, [0, 0, pad_w, pad_h])
                frames.append(padded_img)
            out.append(torch.stack(frames, dim=0))
            central_sizes.append((H, W))
        return torch.stack(out, dim=0), central_sizes

    def eval_step(self, batch, batch_idx=None, log_prefix=None):
        clips, targets = batch  # clips: tuple of (T, 3, H, W); targets: tuple[dict]

        clips_tensor, img_sizes = self._resize_and_pad_clips(clips)
        B, T = clips_tensor.shape[0], clips_tensor.shape[1]

        mask_logits_per_layer, class_logits_per_layer = self(clips_tensor)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            # Keep only central-frame predictions.
            mask_logits = self._slice_central(mask_logits, B, T)  # (B, K, h, w)
            class_logits = self._slice_central(class_logits, B, T)  # (B, K, C+1)

            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )

            preds, targets_ = [], []
            for j in range(len(mask_logits)):
                scores = class_logits[j].softmax(dim=-1)[:, :-1]
                labels = (
                    torch.arange(scores.shape[-1], device=self.device)
                    .unsqueeze(0)
                    .repeat(scores.shape[0], 1)
                    .flatten(0, 1)
                )

                topk_scores, topk_indices = scores.flatten(0, 1).topk(
                    self.eval_top_k_instances, sorted=False
                )
                labels = labels[topk_indices]

                topk_indices = topk_indices // scores.shape[-1]
                mask_logits[j] = mask_logits[j][topk_indices]

                masks = mask_logits[j] > 0
                mask_scores = (
                    mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)
                ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                scores = topk_scores * mask_scores

                preds.append(
                    dict(
                        masks=masks,
                        labels=labels,
                        scores=scores,
                    )
                )
                targets_.append(
                    dict(
                        masks=targets[j]["masks"],
                        labels=targets[j]["labels"],
                        iscrowd=targets[j]["is_crowd"],
                    )
                )

            if not self.solo_only:
                self.update_metrics_instance(preds, targets, i)
                self.update_lcr_instance(preds, i)
            self.update_solo_instance(preds, targets, i)

    def on_test_end(self):
        self._on_eval_end_instance("test")
        self._write_test_results_txt(self.test_results_filename)
