import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def debug_warping(im_a, im_b, corr_map_normalized, mask, step=0):
    """
    一个临时的调试函数，用于可视化 warp 的结果是否正确。
    """
    # --- 准备数据 ---
    # 只处理批次中的第一个样本进行可视化
    im_a = im_a[0]
    im_b = im_b[0]
    corr_map_normalized = corr_map_normalized[0] # Shape: (2, H, W)
    mask = mask[0] # Shape: (1, H, W)

    # --- 核心 Warp 逻辑 ---
    # F.grid_sample 需要 (N, H, W, 2) 形状的 grid，其中最后一个维度是 (x, y)
    # 我们的 corr_map_normalized 是 (2, H, W)，所以需要 unsqueeze 和 permute
    # (2, H, W) -> (1, 2, H, W) -> (1, H, W, 2)
    sampling_grid = corr_map_normalized.unsqueeze(0).permute(0, 2, 3, 1)

    # Warp im_A。需要给 im_A 增加一个批次维度
    # Coordinate normalization matches align_corners=True.
    warped_a = F.grid_sample(
        im_a.unsqueeze(0), 
        sampling_grid,
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True,
    ).squeeze(0) # 移除批次维度

    # --- 转换为 Numpy 以便可视化 ---
    im_a_vis = im_a.permute(1, 2, 0).cpu().numpy()
    im_b_vis = im_b.permute(1, 2, 0).cpu().numpy()
    warped_a_vis = warped_a.permute(1, 2, 0).cpu().numpy()
    
    # 计算 warp 后的图像与目标图像的差异
    diff_vis = np.abs(warped_a_vis - im_b_vis)
    
    # --- 绘图 ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Warping Debug at Step {step}", fontsize=16)
    
    axes[0].imshow(np.clip(im_a_vis, 0, 1))
    axes[0].set_title("Image A (Source)")
    axes[0].axis('off')

    axes[1].imshow(np.clip(im_b_vis, 0, 1))
    axes[1].set_title("Image B (Target)")
    axes[1].axis('off')

    axes[2].imshow(np.clip(warped_a_vis, 0, 1))
    axes[2].set_title("Warped A using Corr Map")
    axes[2].axis('off')

    axes[3].imshow(np.clip(diff_vis, 0, 1))
    axes[3].set_title("Difference (Warped - B)")
    axes[3].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('tmp.png', dpi=100)
    plt.close(fig)
    print(f"已保存 Warp 调试图像至: tmp.png")

def expand_mask_boundary(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    使用最大池化来膨胀一个二值掩码的边界。

    Args:
        mask (torch.Tensor): 输入的二值掩码张量，形状应为 (N, C, H, W)，
                             其中 N 是批次大小，C 是通道数，H 和 W 是高度和宽度。
                             数据类型应为浮点型（例如 torch.float32）。
        kernel_size (int): 用于膨胀的核大小（正方形）。必须是奇数。
                           核越大，膨胀效果越明显。默认为 3。

    Returns:
        torch.Tensor: 膨胀后的掩码张量，与输入张量形状相同。
    """
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"输入必须是 PyTorch 张量，但得到的是 {type(mask)}")

    if mask.ndim != 4:
        raise ValueError(f"输入张量必须是 4 维 (N, C, H, W)，但得到的是 {mask.ndim} 维")

    if kernel_size % 2 == 0:
        raise ValueError(f"核大小必须是奇数，但得到的是 {kernel_size}")

    # 计算 padding 大小以保持输出尺寸与输入一致
    padding = (kernel_size - 1) // 2

    # 使用 F.max_pool2d 执行膨胀操作
    # 对于二值掩码，局部区域的最大值操作等效于形态学膨胀 [2, 10]
    expanded_mask = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding
    )

    return expanded_mask
