
import os
import os.path as osp
import random
import json
import pickle
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from tqdm import tqdm
from freematching.datasets.flow_resize import resize_flow_and_mask_weighted

# ====================================================================
# 关键依赖：从 uniflowmatch 引入 HDF5 读取工具
# 请确保 uniflowmatch.datasets.utils.distributed_h5 在您的 Python 环境中可用
# 如果不可用，您需要将该文件 (distributed_h5.py) 复制到您的项目中
# ====================================================================

try:
    from distributed_h5 import DistributedH5Reader
except:
    from freematching.datasets.distributed_h5 import DistributedH5Reader
# try:
# except ImportError:
#     print("="*80)
#     print("错误: 无法导入 'DistributedH5Reader'.")
#     print("请确保 'uniflowmatch' 包已安装，或者将 'distributed_h5.py' 文件")
#     print("从 'uniflowmatch/datasets/utils/' 目录复制到您的项目可访问路径中。")
#     print("="*80)
#     # 为了能让代码在没有安装 uniflowmatch 的情况下至少能被解析，定义一个占位符类
#     class DistributedH5Reader:
#         def __init__(self, *args, **kwargs):
#             raise NotImplementedError("DistributedH5Reader 未能成功导入。")
#     # exit() # 在实际使用中应该退出

# ====================================================================
# FlyingThings3D 数据集类 (新实现)
# ====================================================================
class FlyingThingsDataset(Dataset):
    """
    用于加载经过 uniflowmatch 脚本预处理后的 FlyingThings3D HDF5 数据集。
    """
    def __init__(self, samples, root_dir, split, ht=360, wt=640, use_horizontal_flip_aug=False, normalize=True, debug=True, dstype='clean', aug_left_right=False, aug_clean_final=False, image_only=False, enumerate_image_variants=False, bidirectional=False):
        super().__init__()
        self.samples = samples
        self.root_dir = root_dir
        self.split = split
        self.ht, self.wt = ht, wt
        
        # 数据增强选项
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.normalize = normalize
        self.debug = debug
        self.aug_left_right = aug_left_right
        self.aug_clean_final = aug_clean_final
        self.dstype = dstype
        self.other_dstype = "final" if self.dstype == "clean" else "clean"
        self.image_only = bool(image_only)
        self.enumerate_image_variants = bool(enumerate_image_variants)
        self.bidirectional = bool(bidirectional)
        if self.enumerate_image_variants and not self.image_only:
            raise ValueError("enumerate_image_variants is only supported with image_only=True")
        self.image_variants = [
            ("cleanpass", "left"),
            ("cleanpass", "right"),
            ("finalpass", "left"),
            ("finalpass", "right"),
        ]

        # 每个 worker 进程都需要自己的 HDF5 读取器实例
        self.flow_reader = None
        self.individual_quantity_reader = None
        
    def _init_readers(self):
        """Initialize per-worker HDF5 readers, skipping GT flow when image-only."""
        if self.individual_quantity_reader is None:
            self.individual_quantity_folder = osp.join(self.root_dir, "individual_quantities", self.split)
            self.individual_quantity_reader = DistributedH5Reader(self.individual_quantity_folder)
        if not self.image_only and self.flow_reader is None:
            self.flow_folder = osp.join(self.root_dir, "flow", self.split)
            self.flow_reader = DistributedH5Reader(self.flow_folder)

    def __len__(self):
        if self.image_only and self.enumerate_image_variants:
            direction_factor = 2 if self.bidirectional else 1
            return len(self.samples) * len(self.image_variants) * direction_factor
        return len(self.samples)

    def __getitem__(self, idx):
        # 确保 HDF5 读取器已在当前 worker 中初始化
        self._init_readers()

        if self.image_only and self.enumerate_image_variants:
            variants_per_direction = len(self.image_variants)
            variants_per_pair = variants_per_direction * (2 if self.bidirectional else 1)
            base_index, expanded_index = divmod(idx, variants_per_pair)
            reverse_pair = self.bidirectional and expanded_index >= variants_per_direction
            img_type, direction = self.image_variants[expanded_index % variants_per_direction]
        else:
            base_index = idx
            reverse_pair = False
            direction = "left" if (not self.aug_left_right or (random.random() < 0.5)) else "right"
            use_clean_pass = self.dstype == 'clean'
            if self.aug_clean_final and random.random() > 0.5:
                use_clean_pass = not use_clean_pass
            img_type = "cleanpass" if use_clean_pass else "finalpass"

        sample_info = self.samples[base_index]
        flow_idx = sample_info['flow_idx']
        img0_idx = sample_info['img0_idx']
        img1_idx = sample_info['img1_idx']
        if reverse_pair:
            img0_idx, img1_idx = img1_idx, img0_idx

        # 2. 加载图像 (im_A 和 im_B)
        im_A_orig = self.individual_quantity_reader.read(img0_idx, [f"frame_{img_type}_{direction}"])[f"frame_{img_type}_{direction}"]
        im_B_orig = self.individual_quantity_reader.read(img1_idx, [f"frame_{img_type}_{direction}"])[f"frame_{img_type}_{direction}"]
        
        # 将 numpy 数组转为 PIL Image 以便使用 torchvision.transforms
        # Final-pass frames are stored as RGBA in the current HDF5 export.
        # Convert every variant explicitly so the model always receives RGB.
        im_A_orig = Image.fromarray(im_A_orig.numpy()).convert("RGB")
        im_B_orig = Image.fromarray(im_B_orig.numpy()).convert("RGB")
        
        w_orig, h_orig = im_A_orig.size

        if self.image_only:
            # Strict unlabeled-refinement path: the flow reader is never initialized.
            im_A = F.resize(im_A_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
            im_B = F.resize(im_B_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
            im_A, im_B = F.to_tensor(im_A), F.to_tensor(im_B)
            if self.normalize:
                im_A = F.normalize(im_A, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                im_B = F.normalize(im_B, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if self.use_horizontal_flip_aug and random.random() > 0.5:
                im_A, im_B = F.hflip(im_A), F.hflip(im_B)
            return {
                'im_A': im_A,
                'im_B': im_B,
                'ground_truth_flow': torch.zeros(2, self.ht, self.wt, dtype=torch.float32),
                'validity_mask': torch.ones(self.ht, self.wt, dtype=torch.float32),
            }

        # 3. Supervised path: load backward flow and visibility.
        flow_keys_to_read = [f"{direction}_view1_flow", f"{direction}_view1_non_occluded_mask"]
        flow_data = self.flow_reader.read(flow_idx, flow_keys_to_read)
        
        flow_orig = flow_data[f"{direction}_view1_flow"] # (H, W, 2) 或 (2, H, W)
        mask_orig = flow_data[f"{direction}_view1_non_occluded_mask"] # (H, W)

        # 确保光流形状为 (2, H, W)
        if flow_orig.shape[-1] == 2:
            flow_orig = flow_orig.permute(2, 0, 1)

        # 4. 图像变换、缩放和归一化 (逻辑与之前保持一致)
        im_A = F.resize(im_A_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_B = F.resize(im_B_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_A, im_B = F.to_tensor(im_A), F.to_tensor(im_B)

        if self.normalize:
            im_A = F.normalize(im_A, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            im_B = F.normalize(im_B, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        sx, sy = self.wt / w_orig, self.ht / h_orig
        flow, mask = resize_flow_and_mask_weighted(flow_orig, mask_orig, [self.ht, self.wt], sx, sy)

        # 6. 水平翻转增强
        if self.use_horizontal_flip_aug and random.random() > 0.5:
            im_A, im_B = F.hflip(im_A), F.hflip(im_B)
            mask, flow = F.hflip(mask), F.hflip(flow)
            flow[0, :, :] *= -1

        # 7. 返回统一格式的字典
        # print('flyingthings:', flow.min() == torch.nan, flow.max() == torch.nan)
        return {
            'im_A': im_A,
            'im_B': im_B,
            'ground_truth_flow': flow,
            'validity_mask': mask.float(),
        }

# ====================================================================
# FlyingThings3D 数据集构建器 (新实现)
# ====================================================================
class FlyingThingsBuilder:
    def __init__(self, data_root):
        self.data_root = data_root
        if not osp.exists(self.data_root):
            raise FileNotFoundError(f"FlyingThings3D 预处理数据根目录未找到: {self.data_root}")

    def build_dataset(self, split='train', **kwargs):
        cache_dir = osp.join(self.data_root, '.cache')
        cache_path = osp.join(cache_dir, f"flyingthings_samples_{split}.pkl")
        
        if osp.exists(cache_path):
            print(f"--- FlyingThings3D: 正在从缓存 '{cache_path}' 加载样本列表 ---")
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            print(f"--- FlyingThings3D: 成功从缓存加载 {len(samples)} 个样本 ---")
        else:
            print(f"--- FlyingThings3D: 未找到缓存，正在执行首次索引构建... ---")
            
            # 1. 加载单帧数据的索引文件
            individual_index_path = osp.join(self.data_root, "individual_quantities", split, "index.json")
            if not osp.exists(individual_index_path):
                raise FileNotFoundError(f"未找到单帧索引文件: {individual_index_path}")
            
            ind_df = pd.read_json(individual_index_path, orient='index')
            ind_df['h5_reader_index'] = ind_df.index
            # 创建一个用于快速查找的多级索引
            ind_df_search = ind_df.set_index(["split", "division", "seq", "idx"])

            # 2. 加载光流数据对的索引文件
            flow_index_path = osp.join(self.data_root, "flow", split, "job_df.csv")
            if not osp.exists(flow_index_path):
                raise FileNotFoundError(f"未找到光流索引文件: {flow_index_path}")
            
            flow_df = pd.read_csv(flow_index_path)

            # 3. 构建样本列表
            samples = []
            split_key = "TRAIN" if split == "train" else "TEST"
            
            for flow_idx, row in tqdm(flow_df.iterrows(), total=len(flow_df), desc=f"构建 FlyingThings3D ({split}) 索引"):
                try:
                    # 使用多级索引快速查找对应的单帧数据索引
                    img0_h5_idx = ind_df_search.loc[(split_key, row["division"], row["seq"], row["pair0_id"])]["h5_reader_index"]
                    img1_h5_idx = ind_df_search.loc[(split_key, row["division"], row["seq"], row["pair1_id"])]["h5_reader_index"]
                    
                    samples.append({
                        'flow_idx': int(flow_idx),
                        'img0_idx': int(img0_h5_idx),
                        'img1_idx': int(img1_h5_idx),
                    })
                except KeyError:
                    # print(f"警告: 找不到键 {(split_key, row['division'], row['seq'], row['pair0_id'])} 或 ...pair1_id。跳过此样本。")
                    continue
            
            print(f"--- FlyingThings3D: 索引构建完成，共找到 {len(samples)} 个有效样本。 ---")
            if samples:
                print(f"--- FlyingThings3D: 正在将样本列表保存到缓存文件: '{cache_path}' ---")
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(samples, f)

        if not samples:
            print(f"警告: FlyingThings3D 数据集 '{split}' 分割中没有找到任何样本。")
            return []
            
        dataset = FlyingThingsDataset(samples, root_dir=self.data_root, split=split, **kwargs)
        return [dataset] # 统一返回列表形式

# ====================================================================
# 使用示例
# ====================================================================
