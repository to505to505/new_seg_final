# ---------------------------------------------------------------
# Video Mask Classification (Instance) training module.
# Supervises the central frame of each clip with the standard EoMT
# instance-segmentation losses, while applying a temporal count-
# consistency loss (L_num) over the full T-frame window so the ETF
# module learns to share information across time.
#
# Optionally runs a 3-branch knowledge-distillation step from a frozen
# 2D EoMT teacher (same 518x518 resolution):
#   Branch 1 — standard student forward (detection branch).
#   Branch 2 — specific KD: the student is run with the teacher's
#              object queries injected before the L2 blocks; class KL,
#              mask Dice+BCE and relational CRRCD losses are computed
#              on the central frame, weighted by the teacher's per-query
#              foreground confidence.
#   Branch 3 — general KD: random unlearnable queries are injected into
#              both teacher and student; class KL + mask Dice+BCE are
#              weighted by the teacher's background probability to
#              suppress false-positive masks on the background.
# Instance segmentation only — no bounding-box regression losses.
# ---------------------------------------------------------------

import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2.functional import pad

from training.mask_classification_instance import MaskClassificationInstance
from training.num_consistency_loss import num_consistency_loss
from training.distillation_losses import CRRCDLoss, class_kl_loss, mask_kd_loss
from training.temporal_query_nce import temporal_query_nce_loss


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
        # ---- Knowledge distillation ----
        distill_enabled: bool = False,
        teacher_ckpt_path: Optional[str] = None,
        distill_loss_weight: float = 1.0,
        distill_class_weight: float = 1.0,
        distill_mask_bce_weight: float = 1.0,
        distill_mask_dice_weight: float = 1.0,
        distill_temperature: float = 1.0,
        crrcd_enabled: bool = True,
        crrcd_loss_weight: float = 1.0,
        crrcd_relation_dim: int = 128,
        crrcd_hidden_dim: int = 256,
        crrcd_num_fg: int = 20,
        crrcd_num_bg: int = 20,
        crrcd_num_negatives: int = 0,
        crrcd_temperature: float = 0.5,
        general_kd_enabled: bool = True,
        general_num_queries: int = 100,
        general_loss_weight: float = 1.0,
        # ---- Temporal per-query InfoNCE (self-supervised consistency) ----
        nce_enabled: bool = False,
        nce_loss_weight: float = 0.1,
        nce_temperature: float = 0.1,
        nce_alive_threshold: float = 0.2,
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

        # ---- Knowledge distillation setup ----
        self.distill_enabled = distill_enabled
        self.distill_loss_weight = distill_loss_weight
        self.distill_class_weight = distill_class_weight
        self.distill_mask_bce_weight = distill_mask_bce_weight
        self.distill_mask_dice_weight = distill_mask_dice_weight
        self.distill_temperature = distill_temperature
        self.crrcd_enabled = crrcd_enabled
        self.crrcd_loss_weight = crrcd_loss_weight
        self.general_kd_enabled = general_kd_enabled
        self.general_num_queries = general_num_queries
        self.general_loss_weight = general_loss_weight

        self.nce_enabled = nce_enabled
        self.nce_loss_weight = nce_loss_weight
        self.nce_temperature = nce_temperature
        self.nce_alive_threshold = nce_alive_threshold

        # The teacher is held in a list so it is NOT registered as a
        # submodule: it must stay out of state_dict / named_parameters
        # (frozen, never optimised, never checkpointed).
        self._teacher_holder: list = []
        self.crrcd = None

        if distill_enabled:
            teacher_ckpt = teacher_ckpt_path or ckpt_path
            if teacher_ckpt is None:
                raise ValueError(
                    "distill_enabled=True requires teacher_ckpt_path "
                    "(or ckpt_path) pointing at a 2D EoMT checkpoint"
                )
            self._teacher_holder = [self._build_teacher(teacher_ckpt)]
            if crrcd_enabled:
                embed_dim = self.network.encoder.backbone.embed_dim
                self.crrcd = CRRCDLoss(
                    hidden_dim=embed_dim,
                    relation_dim=crrcd_relation_dim,
                    frm_hidden_dim=crrcd_hidden_dim,
                    num_fg=crrcd_num_fg,
                    num_bg=crrcd_num_bg,
                    num_negatives=crrcd_num_negatives,
                    temperature=crrcd_temperature,
                )

    def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head):
        # Drop ETF keys from missing list: ETF is a new module that does not
        # exist in 2D EoMT checkpoints; it is initialised to identity (zero
        # out_proj), so absence in the source ckpt is safe.
        # Drop criterion.* / crrcd.* from unexpected list: the criterion and
        # the CRRCD module are constructed only after the parent's _load_ckpt
        # call, so it is normal for these keys to look unexpected/missing.
        filtered_missing = [
            k for k in incompatible_keys.missing_keys
            if not k.startswith("network.etf.") and not k.startswith("crrcd.")
        ]
        filtered_unexpected = [
            k for k in incompatible_keys.unexpected_keys
            if not k.startswith("criterion.") and not k.startswith("crrcd.")
        ]
        if filtered_missing or filtered_unexpected:
            from torch.nn.modules.module import _IncompatibleKeys
            filtered = _IncompatibleKeys(filtered_missing, filtered_unexpected)
            super()._raise_on_incompatible(filtered, load_ckpt_class_head)

    # ------------------------------------------------------------------
    # Teacher
    # ------------------------------------------------------------------

    @property
    def teacher(self):
        return self._teacher_holder[0] if self._teacher_holder else None

    def _build_teacher(self, ckpt_path: str) -> nn.Module:
        """Frozen 2D EoMT teacher: a deep copy of the student network with
        the 2D checkpoint weights loaded. Because VideoEoMT processes a
        4D (B, 3, H, W) input as a T=1 pass-through (ETF skipped), the
        teacher behaves exactly like the original 2D EoMT model."""
        teacher = copy.deepcopy(self.network)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        teacher_sd = {
            k[len("network."):]: v
            for k, v in ckpt.items()
            if k.startswith("network.")
        }
        incompatible = teacher.load_state_dict(teacher_sd, strict=False)
        missing = [
            k for k in incompatible.missing_keys if not k.startswith("etf.")
        ]
        if missing:
            raise ValueError(f"Teacher checkpoint missing keys: {missing}")

        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        return teacher

    def _ensure_teacher_device(self):
        teacher = self.teacher
        if teacher is None:
            return
        param = next(teacher.parameters(), None)
        if param is not None and param.device != self.device:
            teacher.to(self.device)

    @torch.compiler.disable
    def _teacher_forward(self, frames, query_mode="student", injected_queries=None):
        """Run the frozen teacher and return (outputs, query_hidden_states).

        `frames` are expected already scaled to [0, 1] (same convention as
        LightningModule.forward). Compiler-disabled because the teacher is
        a plain, uncompiled module held outside the LightningModule graph.
        """
        teacher = self.teacher
        with torch.no_grad():
            out = teacher(
                frames, query_mode=query_mode, injected_queries=injected_queries
            )
            hs = teacher._captured_decoder_hs
        return out, (hs.detach() if hs is not None else None)

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

        # ---- Branch 1: standard student forward (detection branch) ----
        mask_logits_per_block, class_logits_per_block = self(imgs)
        # Capture query hidden states from Branch 1 immediately — the KD
        # branches re-run the student with different queries and would
        # overwrite self.network._captured_decoder_hs.
        student_hs_branch1 = self.network._captured_decoder_hs

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

        # ---- Self-supervised temporal query InfoNCE ----
        if self.nce_enabled and T > 1 and student_hs_branch1 is not None:
            hs_btkd = student_hs_branch1.view(B, T, *student_hs_branch1.shape[1:])
            final_class = class_logits_per_block[-1].view(
                B, T, *class_logits_per_block[-1].shape[1:]
            )
            # Per-query foreground confidence (detached: only gates the loss).
            fg_conf = final_class.softmax(dim=-1)[..., :-1].amax(dim=-1).detach()
            l_nce = temporal_query_nce_loss(
                hs_btkd,
                fg_conf,
                temperature=self.nce_temperature,
                alive_threshold=self.nce_alive_threshold,
            )
            self.log("loss_nce", l_nce, on_step=True, prog_bar=False)
            total = total + self.nce_loss_weight * l_nce

        # ---- Branches 2 & 3: knowledge distillation ----
        if self.distill_enabled:
            total = total + self._distillation_step(imgs, B, T)

        return total

    def _distillation_step(self, imgs: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """3-branch KD from the frozen 2D EoMT teacher. Branch 1 (detection)
        is handled in training_step; this covers Branch 2 (specific KD +
        CRRCD) and Branch 3 (general KD). All KD losses are computed on the
        central frame only."""
        self._ensure_teacher_device()
        central_idx = self._central_idx(T)

        x = imgs / 255.0                       # student input, full clip
        teacher_frames = x[:, central_idx]     # (B, 3, H, W) — same 518x518

        kd_total = imgs.new_zeros(())

        # ===== Branch 2: specific KD + relational distillation =====
        t_out, teacher_hs = self._teacher_forward(teacher_frames)
        t_mask_logits = t_out[0][-1]           # (B, K, h, w)
        t_class_logits = t_out[1][-1]          # (B, K, C+1)

        # Teacher object queries (captured before the L2 blocks) — for EoMT
        # these are the static learnable query embeddings of the teacher.
        teacher_queries = self.teacher.q.weight.detach()

        # Per-query teacher foreground confidence.
        fg_weight = t_class_logits.softmax(dim=-1).amax(dim=-1)  # (B, K)

        # Student KD forward: full clip, teacher queries injected before L2.
        s_mask_pl, s_class_pl = self.network(
            x, query_mode="teacher", injected_queries=teacher_queries
        )
        student_hs = self._slice_central(self.network._captured_decoder_hs, B, T)
        s_mask_central = self._slice_central(s_mask_pl[-1], B, T)
        s_class_central = self._slice_central(s_class_pl[-1], B, T)

        kd_class = class_kl_loss(
            s_class_central, t_class_logits, fg_weight, self.distill_temperature
        )
        kd_mask_bce, kd_mask_dice = mask_kd_loss(
            s_mask_central, t_mask_logits, fg_weight
        )
        self.log("kd/class", kd_class, on_step=True)
        self.log("kd/mask_bce", kd_mask_bce, on_step=True)
        self.log("kd/mask_dice", kd_mask_dice, on_step=True)
        kd_total = kd_total + self.distill_loss_weight * (
            self.distill_class_weight * kd_class
            + self.distill_mask_bce_weight * kd_mask_bce
            + self.distill_mask_dice_weight * kd_mask_dice
        )

        if self.crrcd is not None:
            kd_crrcd = self.crrcd(
                teacher_hs=teacher_hs,
                student_hs=student_hs,
                weights=fg_weight,
            )
            self.log("kd/crrcd", kd_crrcd, on_step=True)
            kd_total = kd_total + self.crrcd_loss_weight * kd_crrcd

        # ===== Branch 3: general KD (background scanning) =====
        if self.general_kd_enabled:
            embed_dim = self.network.encoder.backbone.embed_dim
            # Random, unlearnable queries shared by teacher and student.
            gen_q = torch.randn(
                self.general_num_queries, embed_dim,
                device=imgs.device, dtype=torch.float32,
            )

            tg_out, _ = self._teacher_forward(
                teacher_frames, query_mode="general", injected_queries=gen_q
            )
            tg_mask_logits = tg_out[0][-1]
            tg_class_logits = tg_out[1][-1]
            # Teacher background probability — high where the teacher is
            # confident the slot is background; teaches the student to
            # suppress false-positive masks there.
            bg_weight = tg_class_logits.softmax(dim=-1)[..., -1]  # (B, K)

            sg_mask_pl, sg_class_pl = self.network(
                x, query_mode="general", injected_queries=gen_q
            )
            sg_mask_central = self._slice_central(sg_mask_pl[-1], B, T)
            sg_class_central = self._slice_central(sg_class_pl[-1], B, T)

            gen_class = class_kl_loss(
                sg_class_central, tg_class_logits, bg_weight, self.distill_temperature
            )
            gen_mask_bce, gen_mask_dice = mask_kd_loss(
                sg_mask_central, tg_mask_logits, bg_weight
            )
            self.log("kd/gen_class", gen_class, on_step=True)
            self.log("kd/gen_mask_bce", gen_mask_bce, on_step=True)
            self.log("kd/gen_mask_dice", gen_mask_dice, on_step=True)
            kd_total = kd_total + self.general_loss_weight * (
                gen_class + gen_mask_bce + gen_mask_dice
            )

        return kd_total

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
