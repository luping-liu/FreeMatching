"""
此模块定义了用于从特定 JSON 格式加载图像对的 PyTorch Dataset 类和相应的 Builder。

提供了两种类型的 Dataset:
1.  JsonPairedDataset: 加载单个图像对。
2.  JsonPairedCompositeDataset: 通过随机拼接两个图像对来创建更具挑战性的样本。

Builder 类 (`JsonBuilder`) 提供了统一的接口来构建这些数据集。
所有类都支持分数过滤和缓存功能。
"""

import os
import os.path as osp
import random
import json
from glob import glob
from typing import Optional, List
import hashlib

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms.functional as F
from tqdm import tqdm


class _BaseJsonDataset(Dataset):
    """
    一个内部基类，用于处理通用的 JSON 加载、过滤和缓存逻辑。
    不应直接实例化。
    """
    def __init__(
        self,
        data_root: str,
        image_root_dir: str,
        score_threshold: Optional[float] = None,
    ):
        self.data_root = data_root
        self.image_root = image_root_dir
        self.all_pairs = self._load_and_filter_pairs(score_threshold)

    def _load_and_filter_pairs(self, score_threshold: Optional[float]) -> list:
        """
        扫描目录，加载所有 JSON 文件，并根据分数阈值进行过滤。
        如果存在有效的缓存文件，则直接从缓存加载。
        """
        path_hash = hashlib.md5(self.data_root.encode()).hexdigest()[:8]
        thresh_str = f"{score_threshold:.2f}" if score_threshold is not None else "all"
        cache_filename = f"cache_thresh_{thresh_str}_{path_hash}.json"
        cache_dir = osp.join(self.data_root, ".cache")
        cache_path = osp.join(cache_dir, cache_filename)

        if osp.exists(cache_path):
            print(f"检测到缓存文件，正在从 '{cache_path}' 加载...")
            try:
                with open(cache_path, 'r') as f:
                    filtered_pairs = json.load(f)
                print(f"成功从缓存加载 {len(filtered_pairs)} 个图像对。")
                return filtered_pairs
            except Exception as e:
                print(f"警告: 缓存文件 '{cache_path}' 已损坏 ({e})。将重新生成。")
        
        print("未找到有效缓存，正在从原始 JSON 文件构建...")
        all_pairs_unfiltered = []
        json_files = sorted(glob(osp.join(self.data_root, 'split*.json')))
        if not json_files:
            raise FileNotFoundError(f"在 '{self.data_root}' 目录中未找到 'split*.json' 文件。")

        print(f"正在从 {len(json_files)} 个 JSON 文件中加载元数据...")
        for json_file in tqdm(json_files, desc="加载JSON文件"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                all_pairs_unfiltered.extend(data)
        
        print(f"共找到 {len(all_pairs_unfiltered)} 个原始图像对。")

        if score_threshold is None:
            filtered_pairs = all_pairs_unfiltered
        else:
            print(f"正在使用 score_final >= {score_threshold} 的标准过滤数据...")
            filtered_pairs = [
                item for item in all_pairs_unfiltered
                if item.get('vlm_filter_cot', {}).get('score_final', 0.0) >= score_threshold
            ]
            print(f"过滤后剩余 {len(filtered_pairs)} 个图像对。")

        print(f"正在将过滤后的列表保存到缓存 '{cache_path}'...")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(cache_path, 'w') as f:
                json.dump(filtered_pairs, f)
            print("缓存保存成功。")
        except Exception as e:
            print(f"警告: 无法保存缓存文件 ({e})。")
            
        return filtered_pairs

    def __len__(self) -> int:
        return len(self.all_pairs)

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError("基类不支持 __getitem__。请使用其子类。")


class JsonPairedDataset(_BaseJsonDataset):
    """
    从 JSON 文件加载单个图像对，并生成与 MegadepthScene 兼容的输出。
    """
    def __init__(
        self,
        data_root: str,
        image_root_dir: str,
        ht: int = 384,
        wt: int = 512,
        normalize: bool = True,
        score_threshold: Optional[float] = None,
        debug: bool = True,
    ):
        super().__init__(data_root, image_root_dir, score_threshold)
        self.ht = ht
        self.wt = wt
        self.normalize = normalize

    def __getitem__(self, idx: int) -> dict:
        pair_info = self.all_pairs[idx]
        img_path1 = osp.join(self.image_root, pair_info['img_path1'])
        img_path2 = osp.join(self.image_root, pair_info['img_path2'])

        try:
            im_A_orig = Image.open(img_path1).convert('RGB')
            im_B_orig = Image.open(img_path2).convert('RGB')
        except FileNotFoundError as e:
            print(f"警告: 无法找到图像 {e}。正在尝试加载下一个样本。")
            return self.__getitem__((idx + 1) % len(self))

        im_A = F.resize(im_A_orig, [self.ht, self.wt], antialias=True)
        im_B = F.resize(im_B_orig, [self.ht, self.wt], antialias=True)
        
        im_A_tensor = F.to_tensor(im_A)
        im_B_tensor = F.to_tensor(im_B)
        
        if self.normalize:
            im_A_final = F.normalize(im_A_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            im_B_final = F.normalize(im_B_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            im_A_final = im_A_tensor
            im_B_final = im_B_tensor

        return {
            'im_A': im_B_final,
            'im_B': im_A_final,
            'ground_truth_flow': torch.zeros(2, self.ht, self.wt, dtype=torch.float),
            'validity_mask': torch.ones(self.ht, self.wt, dtype=torch.float),
            'im_A_depth': torch.zeros(self.ht, self.wt, dtype=torch.float),
        }


class JsonPairedCompositeDataset(_BaseJsonDataset):
    """
    通过拼接两对随机图像来创建更难的复合样本，并支持分数过滤。
    """
    def __init__(
        self,
        data_root: str,
        image_root_dir: str,
        ht: int = 384,
        wt: int = 512,
        normalize: bool = True,
        score_threshold: Optional[float] = None,
        debug: bool = True,
    ):
        super().__init__(data_root, image_root_dir, score_threshold)
        self.ht = ht
        self.wt = wt
        self.normalize = normalize
        if len(self.all_pairs) < 2:
            raise ValueError("复合数据集在过滤后至少需要2个样本才能工作。")

    def __getitem__(self, idx: int) -> dict:
        idx1 = idx
        idx2 = random.randint(0, len(self) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(self) - 1)

        pair_info1 = self.all_pairs[idx1]
        pair_info2 = self.all_pairs[idx2]

        try:
            im_A1_orig = Image.open(osp.join(self.image_root, pair_info1['img_path1'])).convert('RGB')
            im_B1_orig = Image.open(osp.join(self.image_root, pair_info1['img_path2'])).convert('RGB')
            im_A2_orig = Image.open(osp.join(self.image_root, pair_info2['img_path1'])).convert('RGB')
            im_B2_orig = Image.open(osp.join(self.image_root, pair_info2['img_path2'])).convert('RGB')
        except FileNotFoundError as e:
            print(f"警告: 加载复合样本时无法找到图像 {e}。正在尝试下一个样本。")
            return self.__getitem__((idx + 1) % len(self))

        half_wt = self.wt // 2
        im_A1 = F.resize(im_A1_orig, [self.ht, half_wt], antialias=True)
        im_B1 = F.resize(im_B1_orig, [self.ht, half_wt], antialias=True)
        im_A2 = F.resize(im_A2_orig, [self.ht, half_wt], antialias=True)
        im_B2 = F.resize(im_B2_orig, [self.ht, half_wt], antialias=True)

        im_A_composite = Image.new('RGB', (self.wt, self.ht))
        im_A_composite.paste(im_A1, (0, 0))
        im_A_composite.paste(im_A2, (half_wt, 0))

        im_B_composite = Image.new('RGB', (self.wt, self.ht))
        if random.random() >= 0:  # easy version
            im_B_composite.paste(im_B1, (0, 0))
            im_B_composite.paste(im_B2, (half_wt, 0))
        else:
            im_B_composite.paste(im_B2, (0, 0))
            im_B_composite.paste(im_B1, (half_wt, 0))

        im_A_tensor = F.to_tensor(im_A_composite)
        im_B_tensor = F.to_tensor(im_B_composite)
        
        if self.normalize:
            im_A_final = F.normalize(im_A_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            im_B_final = F.normalize(im_B_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            im_A_final = im_A_tensor
            im_B_final = im_B_tensor

        return {
            'im_A': im_B_final,
            'im_B': im_A_final,
            'ground_truth_flow': torch.zeros(2, self.ht, self.wt, dtype=torch.float),
            'validity_mask': torch.ones(self.ht, self.wt, dtype=torch.float),
            'im_A_depth': torch.zeros(self.ht, self.wt, dtype=torch.float),
        }


class UNOBuilder:
    """
    为从 JSON 文件加载的数据集提供一个构建器接口。
    """
    def __init__(self, data_root: str, image_root_dir: str = None):
        self.data_root = data_root
        if image_root_dir is None:
            image_root_dir = data_root.replace('labels', 'images_full')
        self.image_root_dir = image_root_dir

    def build_dataset(self, dataset_type: str = 'composite', **kwargs) -> List[Dataset]:
        """
        构建一个数据集实例并将其放入列表中返回。
        """
        if dataset_type == 'simple':
            dataset_class = JsonPairedDataset
        elif dataset_type == 'composite':
            dataset_class = JsonPairedCompositeDataset
        else:
            raise ValueError(f"不支持的数据集类型: '{dataset_type}'。请选择 'simple' 或 'composite'。")
            
        dataset = dataset_class(
            data_root=self.data_root,
            image_root_dir=self.image_root_dir,
            score_threshold=4.0,
            **kwargs
        )
        
        return [dataset]

# ====================================================================
#  测试代码块 (当此文件作为主程序运行时执行)
# ====================================================================
