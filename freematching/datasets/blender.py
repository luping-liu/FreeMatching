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
import torchvision.transforms as T # 用于保存图像
from freematching.datasets.flow_resize import resize_flow_and_mask_weighted

# ====================================================================
# 辅助函数 (保持不变)
# ====================================================================
import OpenEXR
import Imath

def read_exr_flow(filename):
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    all_channels = list(header['channels'].keys())
    if 'R' in all_channels and 'G' in all_channels:
        channels_to_read = ['R', 'G']
    else:
        float_channels = [c for c in all_channels if header['channels'][c].type == Imath.PixelType(Imath.PixelType.FLOAT)]
        if len(float_channels) < 2:
            raise ValueError(f"错误: EXR '{filename}' 包含的浮点通道少于2个。")
        channels_to_read = float_channels[:2]
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels_bytes = exr_file.channels(channels_to_read, pixel_type=pt)
    channels_np = [np.frombuffer(c, dtype=np.float32).reshape(size) for c in channels_bytes]
    return np.stack(channels_np, axis=-1)

# ====================================================================
# Blender 数据集类 (核心修改)
# ====================================================================
class BlenderSyntheticDataset(Dataset):
    # ===================== 核心修改 1: 增加新参数 =====================
    def __init__(self, samples, ht=360, wt=640, use_horizontal_flip_aug=False, normalize=True, debug=True, replace_background=True):
        super().__init__()
        self.samples = samples
        self.ht, self.wt = ht, wt
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.normalize = normalize
        self.replace_background = replace_background 
        self.debug = debug

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_im(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def load_mask(path):
        # 直接加载为PIL Image，稍后转为numpy
        return Image.open(path).convert('L')

    @staticmethod
    def load_flow(path):
        flow = read_exr_flow(path)
        flow[..., 1] *= -1.0
        return torch.from_numpy(flow)

    def __getitem__(self, idx):
        sample_paths = self.samples[idx]
        
        im_A_path = sample_paths['frame1']
        im_B_path = sample_paths['frame2']
        flow_path = sample_paths['flow_2_to_1']
        mask_B_path = sample_paths['mask2'] # mask B 对应 im_B
        
        im_A_orig, im_B_orig = self.load_im(im_A_path), self.load_im(im_B_path)
        flow_orig, mask_B_pil = self.load_flow(flow_path), self.load_mask(mask_B_path)

        mask_B_np = (np.array(mask_B_pil) > 0).astype(np.float32)
        if mask_B_np.mean() < 0.05:
            # print(f'too small {mask_B_np.mean()} {mask_B_np.shape} ...')
            return self.__getitem__((idx + 1) % len(self))

        # ===================== 核心修改 2: 实现背景替换逻辑 =====================
        if self.replace_background:
            mask_A_path = sample_paths['mask1'] # 需要 mask A
            mask_A_pil = self.load_mask(mask_A_path)

            # 将PIL图像转为Numpy数组以便进行像素操作
            im_A_np = np.array(im_A_orig)
            im_B_np = np.array(im_B_orig)
            mask_A_np = np.array(mask_A_pil)
            mask_B_np = np.array(mask_B_pil)

            # 背景是mask值为0的区域，将其替换为白色(255, 255, 255)
            im_A_np[mask_A_np == 0] = [255, 255, 255]
            im_B_np[mask_B_np == 0] = [255, 255, 255]

            # 将处理后的Numpy数组转回PIL图像，以便后续流程使用
            im_A_orig = Image.fromarray(im_A_np)
            im_B_orig = Image.fromarray(im_B_np)
        # ===================== 背景替换逻辑结束 =====================

        w_orig, h_orig = im_A_orig.size

        # 变换和缩放 (逻辑不变)
        im_A = F.resize(im_A_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_B = F.resize(im_B_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_A, im_B = F.to_tensor(im_A), F.to_tensor(im_B)
        
        if self.normalize:
            im_A = F.normalize(im_A, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            im_B = F.normalize(im_B, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        sx, sy = self.wt / w_orig, self.ht / h_orig
        mask_orig = F.to_tensor(mask_B_pil).squeeze(0) > 0.5
        flow, mask = resize_flow_and_mask_weighted(
            flow_orig.permute(2, 0, 1), mask_orig, [self.ht, self.wt], sx, sy
        )

        if self.use_horizontal_flip_aug and random.random() > 0.5:
            im_A, im_B = F.hflip(im_A), F.hflip(im_B)
            mask, flow = F.hflip(mask), F.hflip(flow)
            flow[0, :, :] *= -1

        # print('blender:', flow.min() == torch.nan, flow.max() == torch.nan)
        return {
            'im_A': im_A, 'im_B': im_B, 
            'ground_truth_flow': flow, 
            'validity_mask': mask.float(),
        }
    
# ====================================================================
# Blender 数据集构建器 (核心修改)
# ====================================================================
class BlenderBuilder:
    def __init__(self, data_root):
        self.data_root = data_root
        if not osp.exists(self.data_root):
            raise FileNotFoundError(f"数据根目录未找到: {self.data_root}")

    def build_dataset(self, split='all', **kwargs):
        cache_dir = osp.join(self.data_root, '.cache')
        cache_path = osp.join(cache_dir, f"blender_samples_{split}.pkl")
        
        if osp.exists(cache_path):
            print(f"--- Blender: 正在从缓存 '{cache_path}' 加载样本列表 ---")
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            print(f"--- Blender: 成功从缓存加载 {len(samples)} 个样本 ---")
        else:
            print(f"--- Blender: 未找到缓存，正在执行首次扫描... ---")
            samples = []
            frames_dir = osp.join(self.data_root, 'frames')
            masks_dir = osp.join(self.data_root, 'masks')
            flow_dir = osp.join(self.data_root, 'flow_data')

            # 提速: 一次性 listdir 三个目录建 set, 内存里做存在性判断,
            # 取代逐样本 5 次 os.path.exists(共享存储上 ~150万次网络 stat -> 数小时)。
            # 语义与原 all(osp.exists(...)) 完全一致(需 frame1/2 + mask1/2 + flow 五件齐全)。
            print(f"--- Blender: 扫描 frames/masks/flow_data 目录 (listdir, 秒级)... ---")
            frames_set = set(os.listdir(frames_dir)) if osp.isdir(frames_dir) else set()
            masks_set = set(os.listdir(masks_dir)) if osp.isdir(masks_dir) else set()
            flow_set = set(os.listdir(flow_dir)) if osp.isdir(flow_dir) else set()

            # 以 frames 里的 *_0001.webp 为锚枚举 basename
            frame1_names = sorted(n for n in frames_set if n.endswith('_0001.webp'))
            for f1_name in tqdm(frame1_names, desc="构建 Blender 样本列表"):
                basename = f1_name[:-len('_0001.webp')]
                f2_name = f"{basename}_0002.webp"
                m1_name = f"{basename}_0001.png"
                m2_name = f"{basename}_0002.png"
                flow_name = f"{basename}_0002.exr"

                # 五件齐全才收(内存集合查, 等价于原 all(exists))
                if (f2_name in frames_set and m1_name in masks_set
                        and m2_name in masks_set and flow_name in flow_set):
                    samples.append({
                        'frame1': osp.join(frames_dir, f1_name),
                        'frame2': osp.join(frames_dir, f2_name),
                        'mask1': osp.join(masks_dir, m1_name),
                        'mask2': osp.join(masks_dir, m2_name),
                        'flow_2_to_1': osp.join(flow_dir, flow_name),
                    })

            print(f"--- Blender: 扫描完成，共找到 {len(samples)} 个有效样本。 ---")
            if samples:
                print(f"--- Blender: 正在将样本列表保存到缓存文件: '{cache_path}' ---")
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(samples, f)

        if not samples:
            return None # 如果没有样本，返回None
            
        dataset = BlenderSyntheticDataset(samples, **kwargs)
        return [dataset] # 统一返回列表

# ====================================================================
# 使用示例 (修改了目录结构说明)
# ====================================================================
