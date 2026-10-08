"""Correspondences are absolute normalized coordinates, ordered (x, y)."""
import torch
import torch.nn.functional as F


def warp_source(source, correspondence):
    """Sample source A on the target B grid; both inputs use BCHW layout."""
    return F.grid_sample(source.float(), correspondence.float().permute(0, 2, 3, 1).clamp(-1, 1),
                         mode='bilinear', padding_mode='border', align_corners=True)


def correspondence_to_flow(correspondence, source_size=None):
    """Convert B→A coordinates to pixel displacement on the resized image grid.

    With unequal source/target sizes, displacement uses their native pixel axes.
    """
    _, channels, height, width = correspondence.shape
    if channels != 2:
        raise ValueError('Expected Bx2xHxW correspondence')
    sh, sw = source_size or (height, width)
    scale = correspondence.new_tensor([sw - 1, sh - 1]).view(1, 2, 1, 1)
    y, x = torch.meshgrid(torch.arange(height, device=correspondence.device),
                          torch.arange(width, device=correspondence.device), indexing='ij')
    identity = torch.stack((x, y), dim=0).unsqueeze(0)
    return (correspondence + 1) * scale / 2 - identity
