import torch
import torchvision.transforms.functional as F


def resize_flow_and_mask(flow, mask, size, scale_x, scale_y):
    flow = F.resize(flow.unsqueeze(0), size, interpolation=F.InterpolationMode.BILINEAR).squeeze(0)
    flow = flow.clone()
    flow[0] *= scale_x
    flow[1] *= scale_y

    mask = F.resize(mask.float().unsqueeze(0).unsqueeze(0), size, interpolation=F.InterpolationMode.NEAREST)
    mask = mask.squeeze(0).squeeze(0) > 0.5
    return flow, mask


def resize_flow_and_mask_weighted(
    flow,
    mask,
    size,
    scale_x,
    scale_y,
    eps=1e-6,
    min_weight=0.5,
    mask_mode="nearest",
):
    finite = torch.isfinite(flow).all(dim=0)
    mask = mask.bool() & finite
    flow = torch.nan_to_num(flow.float(), nan=0.0, posinf=0.0, neginf=0.0)
    flow[:, ~mask] = 0

    mask = mask.float()
    weighted_flow = flow * mask.unsqueeze(0)

    flow_sum = F.resize(weighted_flow.unsqueeze(0), size, interpolation=F.InterpolationMode.BILINEAR).squeeze(0)
    weight = F.resize(mask.unsqueeze(0).unsqueeze(0), size, interpolation=F.InterpolationMode.BILINEAR)
    weight = weight.squeeze(0).squeeze(0)

    flow = flow_sum / weight.clamp_min(eps).unsqueeze(0)
    flow = flow.clone()
    flow[0] *= scale_x
    flow[1] *= scale_y

    if mask_mode == "nearest":
        nearest_mask = F.resize(mask.unsqueeze(0).unsqueeze(0), size, interpolation=F.InterpolationMode.NEAREST)
        out_mask = nearest_mask.squeeze(0).squeeze(0) > 0.5
    elif mask_mode == "weighted":
        out_mask = weight > min_weight
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode}")
    flow[:, ~out_mask] = 0
    return flow, out_mask


def flow_resize_comparison_stats(flow_old, mask_old, flow_weighted, mask_weighted):
    common = mask_old & mask_weighted
    old_only = mask_old & ~mask_weighted
    weighted_only = mask_weighted & ~mask_old

    diff = torch.linalg.vector_norm((flow_old - flow_weighted).float(), dim=0)
    stats = {
        "old_valid": int(mask_old.sum().item()),
        "weighted_valid": int(mask_weighted.sum().item()),
        "common_valid": int(common.sum().item()),
        "old_only_valid": int(old_only.sum().item()),
        "weighted_only_valid": int(weighted_only.sum().item()),
    }

    if common.any():
        common_diff = diff[common]
        stats.update(
            {
                "common_diff_mean": float(common_diff.mean().item()),
                "common_diff_p95": float(common_diff.quantile(0.95).item()),
                "common_diff_max": float(common_diff.max().item()),
                "common_diff_gt_0_1px_frac": float((common_diff > 0.1).float().mean().item()),
                "common_diff_gt_1px_frac": float((common_diff > 1.0).float().mean().item()),
            }
        )
    else:
        stats.update(
            {
                "common_diff_mean": 0.0,
                "common_diff_p95": 0.0,
                "common_diff_max": 0.0,
                "common_diff_gt_0_1px_frac": 0.0,
                "common_diff_gt_1px_frac": 0.0,
            }
        )
    return stats
