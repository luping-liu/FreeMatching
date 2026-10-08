
import os
import os.path as osp
import random
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from tqdm import tqdm
import pickle
import cv2
from freematching.datasets.flow_resize import resize_flow_and_mask_weighted

# ====================================================================
# Spring 数据集类 (新实现)
# ====================================================================
class SpringDataset(Dataset):
    """
    用于加载经过 uniflowmatch 脚本预处理后的 Spring 数据集。
    """
    def __init__(self, samples, ht=360, wt=640, use_horizontal_flip_aug=False, normalize=True, debug=True, image_only=False):
        super().__init__()
        self.samples = samples
        self.ht, self.wt = ht, wt
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.normalize = normalize
        self.debug = debug
        self.image_only = bool(image_only)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_im(path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, idx):
        sample_paths = self.samples[idx]

        # 1. 加载图像对
        im_A_path = sample_paths['frame1']
        im_B_path = sample_paths['frame2']
        im_A_orig, im_B_orig = self.load_im(im_A_path), self.load_im(im_B_path)

        w_orig, h_orig = im_A_orig.size

        if self.image_only:
            # Strict unlabeled-refinement path: never open the flow .npz.
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

        # 2. Supervised path: load preprocessed flow and visibility.
        flow_data = np.load(sample_paths['flow_npz'])
        flow_orig = torch.from_numpy(flow_data['flow_1'])
        mask_orig = torch.from_numpy(flow_data['non_occluded_mask_1'])

        # 3. 变换和缩放
        im_A = F.resize(im_A_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_B = F.resize(im_B_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_A, im_B = F.to_tensor(im_A), F.to_tensor(im_B)

        if self.normalize:
            im_A = F.normalize(im_A, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            im_B = F.normalize(im_B, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        sx, sy = self.wt / w_orig, self.ht / h_orig
        flow, mask = resize_flow_and_mask_weighted(flow_orig, mask_orig, [self.ht, self.wt], sx, sy)

        # 4. 数据增强 - 水平翻转
        if self.use_horizontal_flip_aug and random.random() > 0.5:
            im_A, im_B = F.hflip(im_A), F.hflip(im_B)
            mask, flow = F.hflip(mask), F.hflip(flow)
            flow[0, :, :] *= -1

        # 5. 返回统一格式的字典
        # print('spring:', flow.min() == torch.nan, flow.max() == torch.nan)
        return {
            'im_A': im_A,
            'im_B': im_B,
            'ground_truth_flow': flow,
            'validity_mask': mask.float(),
        }

# ====================================================================
# Spring 数据集构建器 (新实现)
# ====================================================================
class SpringBuilder:
    def __init__(self, data_root):
        self.data_root = data_root
        if not osp.exists(self.data_root):
            raise FileNotFoundError(f"Spring 数据根目录未找到: {self.data_root}")

    def build_dataset(self, split='train', **kwargs):
        # For strict image-only refinement, deterministically enumerate pairs
        # directly from PNG streams without consulting any flow file/index.
        temporal_gaps = kwargs.pop("temporal_gaps", None)
        bidirectional = bool(kwargs.pop("bidirectional", False))
        image_only = bool(kwargs.get("image_only", False))
        if image_only and temporal_gaps is not None:
            gaps = sorted({int(gap) for gap in temporal_gaps})
            if not gaps or gaps[0] < 1:
                raise ValueError(f"temporal_gaps must contain positive integers, got {temporal_gaps}")
            samples = []
            image_dirs = sorted(glob.glob(osp.join(self.data_root, split, "*", "*", "images")))
            for image_dir in image_dirs:
                frames = sorted(glob.glob(osp.join(image_dir, "*.png")))
                for gap in gaps:
                    for first, second in zip(frames, frames[gap:]):
                        samples.append({"frame1": first, "frame2": second})
                        if bidirectional:
                            samples.append({"frame1": second, "frame2": first})
            print(
                f"--- Spring image-only ({split}): gaps={gaps}, "
                f"bidirectional={bidirectional}, pairs={len(samples)} ---"
            )
            if not samples:
                return []
            return [SpringDataset(samples, **kwargs)]

        # Supervised/default adjacent-pair path backed by the existing cache.
        cache_dir = osp.join(self.data_root, '.cache')
        cache_path = osp.join(cache_dir, f"spring_samples_{split}.pkl")
        
        if osp.exists(cache_path):
            print(f"--- Spring: 正在从缓存 '{cache_path}' 加载样本列表 ---")
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            print(f"--- Spring: 成功从缓存加载 {len(samples)} 个样本 ---")
        else:
            print(f"--- Spring: 未找到缓存，正在对 '{split}' 集执行首次扫描... ---")
            samples = []
            split_dir = osp.join(self.data_root, split)
            if not osp.exists(split_dir):
                print(f"警告: 未找到 '{split}' 数据集目录: {split_dir}")
                return []

            # 扫描所有 .npz 文件，因为每个文件代表一个有效的数据对
            npz_files = sorted(glob.glob(osp.join(split_dir, "*", "*", "flow", "*.npz")))
            
            for npz_path in tqdm(npz_files, desc=f"构建 Spring ({split}) 样本列表"):
                parts = npz_path.split(os.sep)
                seq = parts[-4]
                direction = parts[-3]
                filename = osp.basename(npz_path) # e.g., "0001_0002.npz"
                
                idx_0_str, idx_1_str = filename.replace('.npz', '').split('_')
                
                # 构建此样本所需的所有文件路径
                sample_dict = {
                    'frame1': osp.join(self.data_root, split, seq, direction, 'images', f"{idx_0_str}.png"),
                    'frame2': osp.join(self.data_root, split, seq, direction, 'images', f"{idx_1_str}.png"),
                    'flow_npz': npz_path,
                }

                # 检查所有文件是否存在
                if all(osp.exists(p) for p in sample_dict.values()):
                    samples.append(sample_dict)
            
            print(f"--- Spring: 扫描完成，共找到 {len(samples)} 个有效样本。 ---")
            if samples:
                print(f"--- Spring: 正在将样本列表保存到缓存文件: '{cache_path}' ---")
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(samples, f)

        if not samples:
            print(f"警告: Spring 数据集 '{split}' 分割中没有找到任何样本。")
            return []
            
        dataset = SpringDataset(samples, **kwargs)
        return [dataset] # 统一返回列表形式

# ====================================================================
# 使用示例
# ====================================================================
