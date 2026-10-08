# unified_dataset.py

import copy
import os
import random
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, Sampler
from collections import defaultdict


def collate_skip_none(batch):
    """过滤 DatasetWrapper 返回的 None(坏样本), 再用默认 collate 合并。
    ★关键: 若整个 batch 的样本全为 None(如 batch_size 小、连续坏块), 过滤后为空列表,
      default_collate([]) 会抛 IndexError 使该卡 worker 崩溃 -> 多机下拖垮全部 rank。
      这里空 batch 返回 None, 交由训练循环跳过该 step(见 train 脚本 next() 处的 None 检查)。"""
    filtered = [b for b in batch if b is not None]
    if len(filtered) == 0:
        return None
    return torch.utils.data.default_collate(filtered)


# ====================================================================
# 1. 导入所有数据集构建器
# ====================================================================
import importlib
import json

DATA_ROOT = os.environ.get("FREEMATCHING_DATA_ROOT", "data/datasets")
DEFAULT_DATASET_CONFIGS_7 = {'blendedmvs': {'ratio': 0.1, 'path': 'BlendedMVS/blendedmvs_processed_ufm'},
 'scannet': {'ratio': 0.15, 'path': 'scannet++/preprocess_scannetpp_ufm/train'},
 'flyingthings': {'ratio': 0.1, 'path': 'flyingthings/flyingthings_preprocessed_ufm'},
 'spring': {'ratio': 0.05, 'path': 'spring/spring_processed_ufm'},
 'blender': {'ratio': 0.2, 'path': 'objaverse/render_output_all'},
 'video': {'ratio': 0.2, 'path': 'vipe'},
 'pointodyssey': {'ratio': 0.05, 'path': 'pointodyssey'},
 'dynamic_replica': {'ratio': 0.1, 'path': 'DynamicStereo'},
 'flyingchairs': {'ratio': 0.05, 'path': 'flyingchairs'}}
for _cfg in DEFAULT_DATASET_CONFIGS_7.values():
    _cfg["path"] = os.path.join(DATA_ROOT, _cfg["path"])
DEFAULT_DATASET_CONFIGS = DEFAULT_DATASET_CONFIGS_7

def load_dataset_config(path=None):
    if not path:
        return copy.deepcopy(DEFAULT_DATASET_CONFIGS)
    with open(path) as f:
        config = json.load(f)
    for item in config.values():
        item["path"] = os.path.expandvars(os.path.expanduser(item["path"]))
    return config

REQUIRED_KEYS = [
    'im_A', 'im_B', 'ground_truth_flow', 'validity_mask'
]
OPTIONAL_PASSTHROUGH_KEYS = [
    'benchmark_name', 'sample_id', 'sample_dir'
]


# ====================================================================
# 3. 辅助类和自定义采样器 (DatasetWrapper 已修改)
# ====================================================================
class DatasetWrapper(torch.utils.data.Dataset):
    """
    修改后的包装器，它会过滤样本，确保只返回 REQUIRED_KEYS 中定义的键。

    ★坏样本处理(2026-07): self.dataset 是单一类别(如 flyingthings), 某些索引因 HDF5
      filter/解压失败等读不出来。__getitem__ 读失败时, 在【同一个 dataset(即同类别)】内
      随机改取另一个索引重试, 最多重试 MAX_RETRY 次(共 MAX_RETRY+1 次尝试)。这样永远返回
      有效样本, batch 恒满 —— 避免整批 None 导致 collate 崩 / 多机各卡 step 数不齐而 NCCL
      desync 死锁。仅当连续 MAX_RETRY+1 个索引都坏(极罕见)才返回 None 兜底。
    """
    MAX_RETRY = 2   # 额外重试次数; 连同首次共 3 次尝试

    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def _load_one(self, index):
        """加载并过滤单个索引; 失败抛异常(由 __getitem__ 捕获重试)。"""
        original_sample = self.dataset[index]
        filtered_sample = {}
        # ★★★ 核心过滤逻辑: 只复制 REQUIRED_KEYS ★★★
        for key in REQUIRED_KEYS:
            if key in original_sample:
                filtered_sample[key] = original_sample[key]
            else:
                raise KeyError(f"错误: 数据集 '{self.dataset_name}' 的样本索引 {index} 缺少必需的键: '{key}'")
        for key in OPTIONAL_PASSTHROUGH_KEYS:
            filtered_sample[key] = original_sample.get(key, "")
        filtered_sample['dataset_name'] = self.dataset_name
        filtered_sample['index'] = index
        return filtered_sample

    def __getitem__(self, index):
        n = len(self.dataset)
        cur = index
        for attempt in range(self.MAX_RETRY + 1):   # 首次 + MAX_RETRY 次重试
            try:
                return self._load_one(cur)
            except Exception as e:
                if attempt < self.MAX_RETRY and n > 1:
                    # 同类别(同一 dataset)内随机换一个索引重试
                    nxt = random.randint(0, n - 1)
                    if nxt == cur:
                        nxt = (cur + 1) % n
                    print(f"警告：{self.dataset_name} 索引 {cur} 加载失败: {e}。"
                          f"同类别随机重试 {attempt + 1}/{self.MAX_RETRY} -> 索引 {nxt}")
                    cur = nxt
                else:
                    print(f"警告：{self.dataset_name} 索引 {cur} 加载失败且重试 {self.MAX_RETRY} 次均失败: {e}。返回 None。")
                    return None


class DistributedWeightedRandomSampler(Sampler):
    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None, replacement=True, seed=0):
        if num_replicas is None:
            if not dist.is_available(): raise RuntimeError("Requires distributed package")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available(): raise RuntimeError("Requires distributed package")
            rank = dist.get_rank()
        self.dataset, self.weights, self.num_samples = dataset, torch.as_tensor(weights, dtype=torch.double), num_samples
        self.num_replicas, self.rank, self.epoch, self.replacement, self.seed = num_replicas, rank, 0, replacement, seed
        self.num_samples_per_replica = int(torch.ceil(torch.tensor(self.num_samples / self.num_replicas)).item())
        self.total_size = self.num_samples_per_replica * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()
        indices = global_indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)
    
    def __len__(self):
        return self.num_samples_per_replica
    
    def set_epoch(self, epoch):
        # print(f"[DistributedWeightedRandomSampler] 设置 epoch {epoch} 为 rank {self.rank}, num_replicas {self.num_replicas}")
        self.epoch = epoch


# ====================================================================
# 4. 核心构建器类 (已更新)
# ====================================================================
class UnifiedDataloaderBuilder:
    BUILDER_MAP = {'blendedmvs': ('blendedmvs', 'BlendedMVSBuilder'),
     'scannet': ('scannetpp_ufm', 'ScanNetPPBuilder'),
     'flyingthings': ('flyingthings', 'FlyingThingsBuilder'),
     'spring': ('spring', 'SpringBuilder'),
     'blender': ('blender', 'BlenderBuilder'),
     'video': ('videotrack', 'VideoTrajectoryBuilder'),
     'pointodyssey': ('pointodyssey', 'PointOdysseyBuilder'),
     'dynamic_replica': ('dynamic_replica', 'DynamicReplicaSparseBuilder'),
     'flyingchairs': ('flyingchairs', 'FlyingChairsBuilder')}

    def __init__(self, batch_size, num_workers, img_size=(288, 512), num_samples_per_epoch=None, dataset_configs=None, seed=0):  # (360, 640)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.num_samples_per_epoch = num_samples_per_epoch
        self.seed = int(seed)
        # Deep-copy so ratio normalization never mutates the legacy module-level
        # defaults; callers can inject their own dataset configuration.
        self.raw_dataset_configs = copy.deepcopy(dataset_configs if dataset_configs is not None else DEFAULT_DATASET_CONFIGS)
        self.dataset_configs = self._prepare_configs()
        self.all_datasets = []
        self.dataset_info = []
    
    # ... (_prepare_configs, _initialize_datasets, _calculate_weights 方法保持不变) ...
    def _prepare_configs(self):
        active = {k: v for k, v in self.raw_dataset_configs.items() if v.get("ratio", 0) > 0}
        if not active:
            raise ValueError("At least one dataset must have a positive sampling ratio")
        for name, cfg in active.items():
            if name not in self.BUILDER_MAP:
                raise ValueError(f"Unknown dataset: {name}")
            if not cfg.get("path") or not os.path.isdir(cfg["path"]):
                raise FileNotFoundError(f"Dataset {name}: prepare {cfg.get('path')} or set its ratio to zero")
        total = sum(v["ratio"] for v in active.values())
        for cfg in active.values():
            cfg["ratio"] /= total
        return active

    def _initialize_datasets(self):
        ht, wt = self.img_size
        for name, params in self.dataset_configs.items():
            root_path = params.get('path')
            print(f"\n正在加载数据集: {name} (最终比例: {params['ratio']*100:.1f}%) from {root_path}")
            module_name, class_name = self.BUILDER_MAP[name]
            Builder = getattr(importlib.import_module("freematching.datasets." + module_name), class_name)
            builder = Builder(data_root=root_path)
            datasets_list = builder.build_dataset(ht=ht, wt=wt, normalize=False, debug=True)
            if datasets_list:
                for ds in datasets_list:
                    wrapped_ds = DatasetWrapper(ds, name)
                    self.all_datasets.append(wrapped_ds)
                    self.dataset_info.append({'name': name, 'len': len(wrapped_ds)})

    def _calculate_weights(self):
        print("\n--- 正在计算加权采样权重 (已按逻辑数据集分组) ---")
        total_lengths = defaultdict(int)
        grouped_info = defaultdict(list)
        for info in self.dataset_info:
            name, length = info['name'], info['len']
            total_lengths[name] += length
            grouped_info[name].append(info)
        print("\n[统计] 各逻辑数据集的总览:")
        for name, total_len in total_lengths.items():
            num_subsets = len(grouped_info[name])
            print(f" - 逻辑数据集: {name:<12} | 子集数量: {num_subsets:<4} | 总样本数: {total_len}")
        ratio_map = {name: params['ratio'] for name, params in self.dataset_configs.items()}
        sample_weights_map = {}
        print("\n[计算] 各逻辑数据集的单一样本权重:")
        for name, total_len in total_lengths.items():
            if total_len == 0: continue
            ratio = ratio_map.get(name)
            sample_weight = ratio / total_len
            sample_weights_map[name] = sample_weight
            print(f"  - '{name}' 的单一样本权重计算为: {ratio:.2f} / {total_len} = {sample_weight:.10f}")
        weights = []
        for info in self.dataset_info:
            name, length = info['name'], info['len']
            unified_weight = sample_weights_map.get(name, 0)
            weights.extend([unified_weight] * length)
        return weights

    def build(self):
        print("\n--- 开始构建统一数据加载器 ---")
        self._initialize_datasets()
        if not self.all_datasets:
            raise RuntimeError("未能加载任何有效的数据集。")
        master_dataset = ConcatDataset(self.all_datasets)
        print(f"\n--- 主数据集已创建，总计 {len(master_dataset)} 个样本。 ---")
        weights = self._calculate_weights()
        if self.num_samples_per_epoch is not None and self.num_samples_per_epoch > 0:
            num_samples = self.num_samples_per_epoch
            print(f"\n[Epoch Size] 使用用户指定的 epoch 大小: {num_samples} 个样本。")
        else:
            num_samples = len(master_dataset)
            print(f"\n[Epoch Size] 默认使用数据集总长度作为 epoch 大小: {num_samples} 个样本。")
        if len(weights) != len(master_dataset):
             raise RuntimeError("权重列表长度与数据集总长度不匹配!")
        sampler = None
        is_ddp = dist.is_available() and dist.is_initialized()
        if is_ddp:
            sampler = DistributedWeightedRandomSampler(master_dataset, weights, num_samples, seed=self.seed)
        else:
            sampler_generator = torch.Generator().manual_seed(self.seed)
            sampler = WeightedRandomSampler(
                weights, num_samples=num_samples, replacement=True, generator=sampler_generator
            )

        # Use a dedicated generator so worker Python/NumPy/Torch seeds are
        # independent of how many RNG calls each model initialization consumes.
        loader_generator = torch.Generator().manual_seed(self.seed + 1)
        # ★★★ 使用 collate_skip_none: 过滤坏样本(None); 整批全 None 时返回 None(不崩) ★★★
        loader_kwargs = dict(
            dataset=master_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_skip_none,
            generator=loader_generator,
        )
        if self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
        data_loader = DataLoader(**loader_kwargs)
        print("\n--- DataLoader 创建成功 ---")
        return data_loader, sampler

# ====================================================================
# 5. 独立运行的测试脚本 (已更新)
# ====================================================================
