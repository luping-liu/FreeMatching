import torch


def flow_gradient_loss(
    predicted_flow: torch.Tensor, 
    ground_truth_flow: torch.Tensor,
    mask: torch.Tensor = None,
    loss_type: str = 'l2',
) -> torch.Tensor:

    if predicted_flow.shape != ground_truth_flow.shape:
        raise ValueError(
            "Predicted flow and ground truth flow must have the same shape. "
            f"Got {predicted_flow.shape} and {ground_truth_flow.shape}"
        )

    def _calculate_gradients(flow: torch.Tensor):
        """计算水平和垂直方向的梯度"""
        # 垂直梯度 (dy)
        # 形状: (B, 2, H-1, W)
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        
        # 水平梯度 (dx)
        # 形状: (B, 2, H, W-1)
        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        
        return dy, dx

    # 计算预测光流的梯度
    pred_dy, pred_dx = _calculate_gradients(predicted_flow)
    
    # 计算真实光流的梯度
    gt_dy, gt_dx = _calculate_gradients(ground_truth_flow)

    # 计算梯度差异
    diff_dy = pred_dy - gt_dy.detach()
    diff_dx = pred_dx - gt_dx.detach()

    # 应用掩码 (如果提供)
    if mask is not None:
        # 裁剪 mask 以匹配梯度图的尺寸
        mask_dy = (mask[:, :, 1:, :] + mask[:, :, :-1, :]) >= 2.0
        mask_dx = (mask[:, :, :, 1:] + mask[:, :, :, :-1]) >= 2.0
        
        diff_dy = diff_dy * mask_dy.detach()
        diff_dx = diff_dx * mask_dx.detach()
        
        # 归一化因子是掩码中有效像素的数量
        # + 1e-8 避免除以零
        norm_factor = torch.sum(mask_dy) + torch.sum(mask_dx) + 1e-8
    else:
        norm_factor = diff_dy.numel() + diff_dx.numel()

    # 计算损失
    if loss_type == 'l1':
        loss = (torch.abs(diff_dy).mean() + torch.abs(diff_dx).mean()) #/ norm_factor
    elif loss_type == 'l2':
        loss = (torch.square(diff_dy).mean() + torch.square(diff_dx).mean()) #/ norm_factor
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'l1' or 'l2'.")
        
    return loss
