# filename: main_dataloader.py (最终完美美化版 - 懒加载并保留Builder结构)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# ScanNet++ Dataset Loader (Ultimate Lazy-Loading Version with Builder)
# Builder initializes instantly by loading only indices.
# The created dataset processes scene data on-the-fly in __getitem__.
# --------------------------------------------------------
import argparse
import json
import os
import os.path as osp
import random
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as nnF
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ScanNetPPLazyDataset(Dataset):
    """
    (最终优化版 - 懒加载)
    表示整个 ScanNet++ 数据集，但在 `__getitem__` 中懒加载场景数据。
    此类由 ScanNetPPBuilder 构建，并接收所有预加载的元数据。
    len(dataset) 返回场景总数。
    """
    def __init__(
        self,
        data_root,
        all_scene_names,
        all_image_names,
        flat_pair_indices,
        pair_index_offsets,
        all_pairs,
        scene_start_offsets,
        all_intrinsics,
        all_trajectories,
        ht=384,
        wt=512,
        use_horizontal_flip_aug=False,
        normalize=True,
        debug=True,
    ):
        # --- 接收来自 Builder 的预加载元数据 ---
        self.data_root = data_root
        self.all_scene_names = all_scene_names
        self.all_image_names = all_image_names
        self.flat_pair_indices = flat_pair_indices
        self.pair_index_offsets = pair_index_offsets
        self.all_pairs = all_pairs
        self.scene_start_offsets = scene_start_offsets
        self.all_intrinsics = all_intrinsics
        self.all_trajectories = all_trajectories
        
        # --- 配置参数 ---
        self.ht = ht
        self.wt = wt
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.normalize = normalize

    def __len__(self):
        """返回数据集中场景的总数。"""
        return len(self.all_scene_names)

    def _load_view(self, scene_path, image_name):
        try:
            image_path = osp.join(scene_path, 'images', image_name + '.jpg')
            image = Image.open(image_path).convert('RGB')
            
            depth_path = osp.join(scene_path, 'depth', image_name + '.png')
            depth_map_u16 = np.array(Image.open(depth_path))
            depth_map = torch.from_numpy(depth_map_u16.astype(np.float32) / 1000.0)
        except Exception as e:
            raise IOError(f"为场景 {osp.basename(scene_path)} 加载 {image_name} 失败, 错误: {e}")
        return image, depth_map

    def scale_intrinsic(self, K, original_w, original_h):
        sx = self.wt / original_w
        sy = self.ht / original_h
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=K.dtype, device=K.device)
        return sK @ K

    def _compute_single_flow_and_mask(self, depth_source, K_source, K_target, T_target_to_source):
        ht, wt = depth_source.shape
        device = depth_source.device
        
        K_source, K_target = K_source.to(device), K_target.to(device)
        T_target_to_source = T_target_to_source.to(device)

        y, x = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wt, device=device), indexing='ij')
        pixel_coords = torch.stack([x, y], dim=-1).float().view(-1, 2)
        
        p_cam_homo = torch.cat([pixel_coords, torch.ones(ht * wt, 1, device=device)], dim=-1)
        depth_flat = depth_source.reshape(-1)
        
        P_cam_source = torch.inverse(K_source) @ p_cam_homo.T * depth_flat.unsqueeze(0)
        
        R = T_target_to_source[:3, :3]
        t = T_target_to_source[:3, 3].unsqueeze(1)
        
        P_cam_target = R @ P_cam_source + t
        depth_in_target = P_cam_target[2, :]
        
        p_cam_target_homo = K_target @ P_cam_target
        p_cam_target_proj = p_cam_target_homo[:2, :] / (p_cam_target_homo[2, :].unsqueeze(0) + 1e-8)
        
        flow = p_cam_target_proj.T - pixel_coords
        flow = flow.reshape(ht, wt, 2)
        
        mask_valid_depth_source = (depth_flat > 1e-4)
        mask_positive_depth_target = (depth_in_target > 1e-4)
        mask_in_bounds_target = (p_cam_target_proj[0, :] >= 0) & (p_cam_target_proj[0, :] < wt) & \
                                (p_cam_target_proj[1, :] >= 0) & (p_cam_target_proj[1, :] < ht)
                                
        validity_mask = (mask_valid_depth_source & mask_positive_depth_target & mask_in_bounds_target).reshape(ht, wt)
        return flow, validity_mask

    def warp_flow(self, flow_to_warp, reference_flow):
        B, C, H, W = flow_to_warp.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=flow_to_warp.device), torch.arange(W, device=flow_to_warp.device), indexing='ij')
        grid = torch.stack((grid_x, grid_y), 2).float()
        
        sampling_grid = grid + reference_flow.permute(1, 2, 0)
        sampling_grid[..., 0] = 2 * sampling_grid[..., 0] / (W - 1) - 1
        sampling_grid[..., 1] = 2 * sampling_grid[..., 1] / (H - 1) - 1
        
        warped_flow = nnF.grid_sample(flow_to_warp, sampling_grid.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True)
        return warped_flow.squeeze(0)

    def compute_bidirectional_flow_and_mask(self, depth_A, depth_B, K1, K2, T_c2w_1, T_c2w_2):
        T_c2w_1, T_c2w_2 = T_c2w_1.float(), T_c2w_2.float()
        T_A_to_B = torch.inverse(T_c2w_2) @ T_c2w_1
        T_B_to_A = torch.inverse(T_A_to_B)
        
        flow_bwd, mask_bwd_geom = self._compute_single_flow_and_mask(depth_B, K2, K1, T_B_to_A)
        flow_fwd, _ = self._compute_single_flow_and_mask(depth_A, K1, K2, T_A_to_B)
        
        flow_bwd_chw = flow_bwd.permute(2, 0, 1)
        flow_fwd_chw = flow_fwd.permute(2, 0, 1)
        
        flow_fwd_warped = self.warp_flow(flow_fwd_chw.unsqueeze(0), flow_bwd_chw)
        
        residual_sq = torch.sum((flow_bwd_chw + flow_fwd_warped)**2, dim=0)
        flow_bwd_mag_sq = torch.sum(flow_bwd_chw**2, dim=0) + 1e-6
        
        consistency_mask = (residual_sq < 0.01 * flow_bwd_mag_sq + 0.5)
        final_mask = mask_bwd_geom & consistency_mask
        
        return flow_bwd, final_mask.float()

    def __getitem__(self, scene_idx):
        while True:
            scene_name = self.all_scene_names[scene_idx]
            scene_path = osp.join(self.data_root, scene_name)
            
            start, end = self.pair_index_offsets[scene_idx], self.pair_index_offsets[scene_idx + 1]
            scene_pair_indices = self.flat_pair_indices[start:end]

            if len(scene_pair_indices) == 0:
                scene_idx = (scene_idx + 1) % len(self)
                continue

            random_pair_global_index = random.choice(scene_pair_indices)
            global_idx1, global_idx2 = self.all_pairs[random_pair_global_index]

            scene_start_offset = self.scene_start_offsets[scene_idx]
            local_idx1, local_idx2 = global_idx1 - scene_start_offset, global_idx2 - scene_start_offset

            if scene_idx + 1 < len(self.scene_start_offsets):
                next_scene_start_offset = self.scene_start_offsets[scene_idx + 1]
            else:
                next_scene_start_offset = len(self.all_image_names)
            
            scene_images = self.all_image_names[scene_start_offset:next_scene_start_offset]
            
            img_name1, img_name2 = scene_images[local_idx1], scene_images[local_idx2]
            
            try:
                im_A_orig, depth_A_orig = self._load_view(scene_path, img_name1)
                im_B_orig, depth_B_orig = self._load_view(scene_path, img_name2)
                
                K1_orig = torch.from_numpy(self.all_intrinsics[global_idx1].copy())
                K2_orig = torch.from_numpy(self.all_intrinsics[global_idx2].copy())
                T_c2w_1 = torch.from_numpy(self.all_trajectories[global_idx1].copy())
                T_c2w_2 = torch.from_numpy(self.all_trajectories[global_idx2].copy())

            except (IOError, FileNotFoundError):
                continue
                
            im_A = F.resize(im_A_orig, [self.ht, self.wt], antialias=True)
            im_B = F.resize(im_B_orig, [self.ht, self.wt], antialias=True)
            depth_A = F.resize(depth_A_orig.unsqueeze(0), [self.ht, self.wt], interpolation=F.InterpolationMode.NEAREST).squeeze(0)
            depth_B = F.resize(depth_B_orig.unsqueeze(0), [self.ht, self.wt], interpolation=F.InterpolationMode.NEAREST).squeeze(0)
            
            im_A_tensor, im_B_tensor = F.to_tensor(im_A), F.to_tensor(im_B)
            
            w_orig, h_orig = im_A_orig.size
            K1, K2 = self.scale_intrinsic(K1_orig, w_orig, h_orig), self.scale_intrinsic(K2_orig, w_orig, h_orig)
            
            im_A_final, im_B_final = im_A_tensor, im_B_tensor
            if self.normalize:
                im_A_final = F.normalize(im_A_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                im_B_final = F.normalize(im_B_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                
            if self.use_horizontal_flip_aug and random.random() > 0.5:
                im_A_final, im_B_final = F.hflip(im_A_final), F.hflip(im_B_final)
                depth_A, depth_B = F.hflip(depth_A), F.hflip(depth_B)
                flip_mat = torch.tensor([[-1, 0, self.wt - 1], [0, 1, 0], [0, 0, 1.]], dtype=torch.float)
                K1, K2 = flip_mat @ K1, flip_mat @ K2
                
            flow, mask = self.compute_bidirectional_flow_and_mask(depth_A, depth_B, K1, K2, T_c2w_1, T_c2w_2)
            
            if mask.mean() < 0.05:
                continue
                
            return {
                'im_A': im_A_final, 'im_B': im_B_final, 
                'ground_truth_flow': flow.permute(2, 0, 1),
                'validity_mask': mask.float(), 
                'im_A_depth': depth_A, 
                'K1': K1, 'K2': K2,
                'T_1to2': torch.inverse(T_c2w_2) @ T_c2w_1
            }


class ScanNetPPBuilder:
    """
    (最终优化版) 根据预处理好的内存映射缓存 (`mmap_cache`) 构建数据集。
    此 Builder 仅加载元数据和索引，然后创建一个实现了懒加载的 Dataset 实例。
    """
    def __init__(self, data_root, cache_dir="mmap_cache"):
        self.data_root = data_root
        self.cache_path = osp.join(self.data_root, cache_dir)
        
        if not osp.exists(self.cache_path):
            raise FileNotFoundError(f"错误: 缓存目录 `mmap_cache` 未在 {self.data_root} 中找到。")
            
        print("正在从内存映射缓存加载元数据 (Builder 初始化)...")
        self._load_metadata_and_indices()
        print("元数据加载完毕。Builder 初始化完成。")

    def _load_metadata_and_indices(self):
        """将所有元数据和 mmap 数组加载到 Builder 实例属性中。"""
        with open(osp.join(self.cache_path, 'meta.json'), 'r') as f:
            meta = json.load(f)
            self.all_scene_names = meta['all_scene_names']
            self.all_image_names = np.array(meta['all_image_names'], dtype=object)
        
        self.flat_pair_indices = np.load(osp.join(self.cache_path, 'flat_pair_indices.npy'), mmap_mode='r')
        self.pair_index_offsets = np.load(osp.join(self.cache_path, 'pair_index_offsets.npy'))

        self.all_pairs = np.load(osp.join(self.cache_path, 'all_pairs.npy'), mmap_mode='r')
        all_sceneids = np.load(osp.join(self.cache_path, 'all_sceneids.npy'), mmap_mode='r')
        self.all_intrinsics = np.load(osp.join(self.cache_path, 'all_intrinsics.npy'), mmap_mode='r')
        self.all_trajectories = np.load(osp.join(self.cache_path, 'all_trajectories.npy'), mmap_mode='r')
        
        _, scene_image_counts = np.unique(all_sceneids, return_counts=True)
        self.scene_start_offsets = np.concatenate(([0], np.cumsum(scene_image_counts[:-1])))

    def build_dataset(self, **kwargs):
        """
        (最终优化版) 构建并返回一个单一的 ScanNetPPLazyDataset 实例。
        此实例包含了所有预加载的元数据和对 mmap 数组的引用。
        """
        print("正在构建懒加载数据集实例...")
        lazy_dataset = ScanNetPPLazyDataset(
            data_root=self.data_root,
            all_scene_names=self.all_scene_names,
            all_image_names=self.all_image_names,
            flat_pair_indices=self.flat_pair_indices,
            pair_index_offsets=self.pair_index_offsets,
            all_pairs=self.all_pairs,
            scene_start_offsets=self.scene_start_offsets,
            all_intrinsics=self.all_intrinsics,
            all_trajectories=self.all_trajectories,
            **kwargs
        )
        return [lazy_dataset, ]


