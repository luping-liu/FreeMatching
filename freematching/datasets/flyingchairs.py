# ====================================================================
# flyingchairs.py (Final Version for Standard Layout)
#
# 作者: Gemini
# 描述: 适用于 FlyingChairs 标准文件布局的最终版数据集加载器。
#      本代码要求 'FlyingChairs_train_val.txt' 文件已位于
#      'FlyingChairs_release' 文件夹内部。
#      代码结构与您的 FlyingThings3D 加载器严格保持一致。
# ====================================================================

import os
import os.path as osp
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision.datasets import FlyingChairs as TorchVisionFlyingChairs
from tqdm import tqdm
from freematching.datasets.flow_resize import resize_flow_and_mask_weighted

# ====================================================================
# FlyingChairs 数据集类
# ====================================================================
class FlyingChairsDataset(Dataset):
    """
    用于加载 torchvision 提供的 FlyingChairs 数据集。
    假定数据集遵循官方的标准文件布局。
    """
    def __init__(self, root_dir, split, ht=360, wt=640, use_horizontal_flip_aug=False, normalize=True, **kwargs):
        super().__init__()
        self.root_dir = root_dir # root_dir is the parent of 'FlyingChairs_release'
        self.split = split
        self.ht, self.wt = ht, wt
        
        # 数据增强选项
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.normalize = normalize

        # torchvision 的 FlyingChairs 类要求 root 是 'FlyingChairs_release' 的父目录。
        # Builder 会传入正确的路径，所以这里直接使用即可。
        self.chairs_dataset = TorchVisionFlyingChairs(root=self.root_dir, split=self.split)
        print(f"--- FlyingChairs: 成功初始化 '{self.split}' 数据集，共 {len(self.chairs_dataset)} 个样本 ---")
        
    def __len__(self):
        return len(self.chairs_dataset)

    def __getitem__(self, idx):
        # 1. 从 torchvision 数据集加载原始数据
        # im_A_orig, im_B_orig 是 PIL Image, flow_orig 是 (2, H, W) 的 numpy 数组
        im_A_orig, im_B_orig, flow_orig_np = self.chairs_dataset[idx]
        
        flow_orig = torch.from_numpy(flow_orig_np) # shape (2, H, W)
        w_orig, h_orig = im_A_orig.size

        # 2. 计算有效性掩码 (FOV Mask)
        # 像素坐标网格
        coords_y, coords_x = torch.meshgrid(torch.arange(h_orig), torch.arange(w_orig), indexing="ij")
        base_coords = torch.stack((coords_x, coords_y), dim=0).float() # shape (2, H, W)
        
        # 目标坐标
        target_coords = base_coords + flow_orig
        
        # 检查是否越界
        mask_orig = (target_coords[0] >= 0) & (target_coords[0] < w_orig) & \
                    (target_coords[1] >= 0) & (target_coords[1] < h_orig)

        # 3. 图像变换、缩放和归一化
        im_A = F.resize(im_A_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_B = F.resize(im_B_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_A, im_B = F.to_tensor(im_A), F.to_tensor(im_B)

        if self.normalize:
            im_A = F.normalize(im_A, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            im_B = F.normalize(im_B, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        sx, sy = self.wt / w_orig, self.ht / h_orig
        flow, mask = resize_flow_and_mask_weighted(flow_orig, mask_orig, [self.ht, self.wt], sx, sy)

        # 5. 水平翻转增强
        if self.use_horizontal_flip_aug and random.random() > 0.5:
            im_A, im_B = F.hflip(im_A), F.hflip(im_B)
            mask = F.hflip(mask.unsqueeze(0)).squeeze(0)
            flow = F.hflip(flow)
            flow[0, :, :] *= -1 # 翻转后 x 方向的光流需要取反

        # 6. 返回统一格式的字典
        return {
            'im_A': im_B,
            'im_B': im_A,
            'ground_truth_flow': flow,
            'validity_mask': mask.float(),
        }

# ====================================================================
# FlyingChairs 数据集构建器
# ====================================================================
class FlyingChairsBuilder:
    def __init__(self, data_root):
        self.data_root = data_root
        if not osp.exists(osp.join(self.data_root, 'FlyingChairs_release')):
             raise FileNotFoundError(
                f"在 '{self.data_root}' 目录中未找到 'FlyingChairs_release' 子目录。\n"
                f"请确保 data_root 是 'FlyingChairs_release' 文件夹的【父目录】。"
            )

    def build_dataset(self, split='train', **kwargs):
        print(f"--- FlyingChairs: 正在构建 '{split}' 分割数据集 ---")
        dataset = FlyingChairsDataset(root_dir=self.data_root, split=split, **kwargs)
        
        if len(dataset) == 0:
            print(f"警告: FlyingChairs 数据集 '{split}' 分割中没有找到任何样本。")
            return []
        
        return [dataset] # 统一返回列表形式

# ====================================================================
# 使用示例
# ====================================================================
