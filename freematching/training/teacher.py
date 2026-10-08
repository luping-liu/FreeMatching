from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _prepare_roma_inputs(
    src: torch.Tensor,
    tgt: torch.Tensor,
    h_lr: int,
    w_lr: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int, int]:
    """Resize preserving aspect ratio, then pad to RoMa's square input."""
    _, _, h_in, w_in = src.shape
    scale = h_lr / max(h_in, w_in)
    h_scaled, w_scaled = int(h_in * scale), int(w_in * scale)

    src_scaled = F.interpolate(
        src,
        size=(h_scaled, w_scaled),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    tgt_scaled = F.interpolate(
        tgt,
        size=(h_scaled, w_scaled),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    pad_h, pad_w = h_lr - h_scaled, w_lr - w_scaled
    src_lr = F.pad(src_scaled, (0, pad_w, 0, pad_h), mode="constant", value=0)
    tgt_lr = F.pad(tgt_scaled, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return src_lr, tgt_lr, h_in, w_in, h_scaled, w_scaled, pad_h, pad_w


def _remap_roma_warp_to_unpadded(
    warp: torch.Tensor,
    h_scaled: int,
    w_scaled: int,
    pad_h: int,
    pad_w: int,
    h_lr: int,
    w_lr: int,
) -> torch.Tensor:
    warp = warp[:, :h_scaled, :w_scaled, :]
    if pad_w > 0 or pad_h > 0:
        x_range = 2.0 * w_scaled / w_lr
        y_range = 2.0 * h_scaled / h_lr
        warp[..., 0] = (warp[..., 0] + 1.0) / x_range * 2.0 - 1.0
        warp[..., 1] = (warp[..., 1] + 1.0) / y_range * 2.0 - 1.0
    return warp


def _convert_align_corners_false_to_true(
    grid: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """RoMa grids use align_corners=False; current flow code uses align_corners=True."""
    grid = grid.clone()
    if width > 1:
        grid[:, 0] = grid[:, 0] * (width / (width - 1))
    if height > 1:
        grid[:, 1] = grid[:, 1] * (height / (height - 1))
    return grid.clamp(-1, 1)


def warp_image_with_flow(
    image: torch.Tensor,
    flow_grid: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp an image with a normalized sampling grid shaped [B, 2, H, W]."""
    return F.grid_sample(
        image,
        flow_grid.permute(0, 2, 3, 1).clamp(-1, 1),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def build_identity_flow(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create an identity sampling grid shaped [B, 2, H, W] with align_corners=True convention."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    identity = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    return identity.expand(batch_size, -1, -1, -1).contiguous()


def compose_flow_fields(
    base_flow: torch.Tensor,
    residual_flow: torch.Tensor,
    *,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Compose target->intermediate and intermediate->source grids.

    `base_flow` is the student prediction C0: target coords -> source coords.
    `residual_flow` is the RoMa refinement S: target coords -> intermediate coords.
    Returns C_new = C0(S): target coords -> source coords.
    """
    return F.grid_sample(
        base_flow,
        residual_flow.permute(0, 2, 3, 1).clamp(-1, 1),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).clamp(-1, 1)


class RoMaV2Estimator:
    """
    Shared frozen RoMaV2 wrapper.

    Provides:
    - estimate_confidence(warped_source, target_image)
    - estimate_match(pred_image, source_image)
    """

    def __init__(self, setting: str = "fast"):
        self.model = None
        self.setting = setting
        self._cpu_offload = False

    def _ensure_loaded(self, device: torch.device) -> None:
        if self.model is None:
            print(f">>> [RoMaV2] Loading with setting='{self.setting}'...")
            old_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            from romav2 import RoMaV2

            cfg = RoMaV2.Cfg(setting=self.setting, compile=False)
            self.model = RoMaV2(cfg)
            self.model.to(device)
            self.model.eval()
            self.model.requires_grad_(False)
            torch.set_float32_matmul_precision(old_precision)
            print(f">>> [RoMaV2] Loaded. Resolution: {self.model.H_lr}x{self.model.W_lr}")

    def enable_cpu_offload(self) -> None:
        assert self.model is not None, "Call _ensure_loaded before enable_cpu_offload"
        self.model.to("cpu")
        for param in self.model.parameters():
            param.data = param.data.pin_memory()
        for buffer in self.model.buffers():
            if buffer.is_floating_point():
                buffer.data = buffer.data.pin_memory()
        self._cpu_offload = True
        print(">>> [RoMaV2] CPU offload enabled with pinned memory")

    @torch.no_grad()
    def estimate_confidence(
        self,
        warped_source: torch.Tensor,
        target_image: torch.Tensor,
    ) -> torch.Tensor:
        """Return overlap confidence map [B, 1, H, W] in [0, 1]."""
        self._ensure_loaded(warped_source.device)
        if self._cpu_offload:
            self.model.to(warped_source.device)

        src = warped_source.clamp(0, 1).float()
        tgt = target_image.clamp(0, 1).float()
        h_lr, w_lr = self.model.H_lr, self.model.W_lr
        src_lr, tgt_lr, h_in, w_in, h_scaled, w_scaled, _, _ = _prepare_roma_inputs(src, tgt, h_lr, w_lr)

        old_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            preds = self.model(src_lr, tgt_lr)
        finally:
            torch.set_float32_matmul_precision(old_precision)

        overlap = preds["confidence_AB"].clone()[..., :1].sigmoid()
        overlap = overlap[:, :h_scaled, :w_scaled, :].permute(0, 3, 1, 2)
        overlap = F.interpolate(overlap, size=(h_in, w_in), mode="bilinear", align_corners=False)

        if self._cpu_offload:
            overlap = overlap.clone()
            self.model.to("cpu")
            torch.cuda.empty_cache()

        return overlap

    @torch.no_grad()
    def estimate_match(
        self,
        pred_image: torch.Tensor,
        source_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return direct matching grid aligned to grid_sample(..., align_corners=True).

        Input order:
        - pred_image: coordinates live on this image
        - source_image: sampling happens from this image
        """
        self._ensure_loaded(pred_image.device)
        if self._cpu_offload:
            self.model.to(pred_image.device)

        pred = pred_image.clamp(0, 1).float()
        src = source_image.clamp(0, 1).float()
        h_lr, w_lr = self.model.H_lr, self.model.W_lr
        pred_lr, src_lr, h_in, w_in, h_scaled, w_scaled, pad_h, pad_w = _prepare_roma_inputs(
            pred, src, h_lr, w_lr
        )

        old_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            preds = self.model(pred_lr, src_lr)
        finally:
            torch.set_float32_matmul_precision(old_precision)

        warp_ab = _remap_roma_warp_to_unpadded(
            preds["warp_AB"].clone(), h_scaled, w_scaled, pad_h, pad_w, h_lr, w_lr
        )
        overlap = preds["confidence_AB"].clone()[..., :1].sigmoid()
        overlap = overlap[:, :h_scaled, :w_scaled, :]

        warp_ab = F.interpolate(
            warp_ab.permute(0, 3, 1, 2),
            size=(h_in, w_in),
            mode="bilinear",
            align_corners=False,
        )
        overlap = F.interpolate(
            overlap.permute(0, 3, 1, 2),
            size=(h_in, w_in),
            mode="bilinear",
            align_corners=False,
        )
        warp_ab = _convert_align_corners_false_to_true(warp_ab, h_in, w_in)

        if self._cpu_offload:
            warp_ab = warp_ab.clone()
            overlap = overlap.clone()
            self.model.to("cpu")
            torch.cuda.empty_cache()

        return warp_ab, overlap


_ROMA_CACHE: Dict[str, RoMaV2Estimator] = {}


def get_roma_estimator(setting: str = "fast") -> RoMaV2Estimator:
    if setting not in _ROMA_CACHE:
        _ROMA_CACHE[setting] = RoMaV2Estimator(setting=setting)
    return _ROMA_CACHE[setting]


@torch.no_grad()
def compute_iterative_roma_targets(
    base_flow: torch.Tensor,
    source_image: torch.Tensor,
    target_image: torch.Tensor,
    student_mask_pred: Optional[torch.Tensor] = None,
    *,
    roma_setting: str = "fast",
    roma_conf_thresh: float = 0.2,
    mask_thresh: float = 0.7,
    require_improvement: bool = True,
    improvement_margin: float = 0.0,
    return_masks_meta: bool = False,
    return_debug: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Build self-correction targets with:
      C0 = base_flow
      I_inter = warp(source, C0)
      S = RoMa(target, I_inter)
      C_new = compose(C0, S)

    Returns:
    - new_flow: composed pseudo target flow
    - new_mask: valid region for the pseudo target
    - residual_flow: RoMa-predicted local refinement

    When `return_masks_meta=True`, also returns compact mask-related tensors
    useful for training diagnostics.

    When `return_debug=True`, also returns extra intermediate tensors used for
    visualization / debugging.
    """
    flow_h, flow_w = base_flow.shape[-2:]
    img_h, img_w = source_image.shape[-2:]
    if (flow_h != img_h) or (flow_w != img_w):
        source_resized = F.interpolate(source_image.detach(), size=(flow_h, flow_w), mode="bilinear", align_corners=True)
        target_resized = F.interpolate(target_image.detach(), size=(flow_h, flow_w), mode="bilinear", align_corners=True)
    else:
        source_resized = source_image.detach()
        target_resized = target_image.detach()

    base_flow = base_flow.detach().clamp(-1, 1)
    base_warped = warp_image_with_flow(source_resized.float(), base_flow.float()).to(source_resized.dtype)

    target_01 = target_resized.float().clamp(0, 1)
    inter_01 = base_warped.float().clamp(0, 1)

    residual_grid, roma_conf = get_roma_estimator(setting=roma_setting).estimate_match(
        pred_image=target_01,
        source_image=inter_01,
    )
    identity_flow = build_identity_flow(
        batch_size=base_flow.shape[0],
        height=flow_h,
        width=flow_w,
        device=base_flow.device,
        dtype=base_flow.dtype,
    )
    residual_flow = (residual_grid.to(base_flow.dtype) - identity_flow).detach()
    refined_flow = compose_flow_fields(base_flow.float(), residual_grid.float()).to(base_flow.dtype)

    valid_mask = (roma_conf > roma_conf_thresh).float()

    if student_mask_pred is not None:
        student_mask = ((student_mask_pred[:, :1].detach() * 0.5 + 0.5) > mask_thresh).float()
        valid_mask = valid_mask * student_mask
    else:
        student_mask = torch.ones_like(valid_mask)

    refined_warped = warp_image_with_flow(source_resized.float(), refined_flow.float()).to(source_resized.dtype)
    base_err = (base_warped.float() - target_resized.float()).abs().mean(dim=1, keepdim=True)
    refined_err = (refined_warped.float() - target_resized.float()).abs().mean(dim=1, keepdim=True)
    improvement_mask = (refined_err + improvement_margin < base_err).float()

    if require_improvement:
        valid_mask = valid_mask * improvement_mask

    target_flow = refined_flow.detach() * valid_mask + base_flow.detach() * (1.0 - valid_mask)
    target_mask = valid_mask.detach()

    outputs = {
        "new_flow": target_flow,
        "new_mask": target_mask,
        "residual_flow": residual_flow,
    }

    if return_masks_meta or return_debug:
        outputs.update(
            {
                "student_mask": student_mask.detach(),
                "roma_conf": roma_conf.detach(),
                "improvement_mask": improvement_mask.detach(),
            }
        )

    if return_debug:
        outputs.update(
            {
                "target_flow": target_flow,
                "target_mask": target_mask,
                "valid_mask": valid_mask,
                "base_flow": base_flow,
                "residual_grid": residual_grid.detach(),
                "refined_flow": refined_flow.detach(),
                "base_warped": base_warped.detach(),
                "refined_warped": refined_warped.detach(),
                "target_image": target_resized.detach(),
            }
        )

    return outputs
