import os
import os.path as osp
import pickle
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from tqdm import tqdm
from freematching.datasets.flow_resize import resize_flow_and_mask_weighted

# BlendedMVS 预处理脚本依赖于 OpenCV 来读取 .exr 文件
# 确保已安装: pip install opencv-python
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

# ====================================================================
# 关键实现：从深度和相机位姿计算光流 (与之前版本保持一致)
# ====================================================================
def get_flow_from_depth_and_poses(depth0, K0, T_world_from_cam0, K1, T_world_from_cam1, ht, wt):
    """
    根据深度图和相机参数计算从图像0到图像1的光流。
    """
    # 计算从相机0到相机1的相对变换
    T_cam1_from_world = torch.inverse(T_world_from_cam1)
    T_cam1_from_cam0 = torch.matmul(T_cam1_from_world, T_world_from_cam0)

    # 创建像素坐标网格
    y_coords, x_coords = torch.meshgrid(torch.arange(ht), torch.arange(wt), indexing='ij')
    coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=0).float()
    coords = coords.view(3, -1) # (3, H*W)

    # 将深度图展平
    depth_vec = depth0.view(-1) # (H*W)

    # 反投影到相机0坐标系下的3D点
    inv_K0 = torch.inverse(K0)
    cam0_points = torch.matmul(inv_K0, coords) * depth_vec # (3, H*W)

    # 变换到相机1坐标系
    cam0_points_hom = torch.cat([cam0_points, torch.ones(1, ht * wt)], dim=0) # (4, H*W)
    cam1_points_hom = torch.matmul(T_cam1_from_cam0, cam0_points_hom)
    cam1_points = cam1_points_hom[:3, :] # (3, H*W)

    # 投影到图像1的像素平面
    cam1_z_raw = cam1_points[2, :]
    cam1_z = cam1_z_raw.clamp(min=1e-8)
    
    projected_coords = torch.matmul(K1, cam1_points)
    projected_coords = projected_coords[:2, :] / cam1_z # (2, H*W)

    # 计算光流
    flow_vec = projected_coords - coords[:2, :]
    flow = flow_vec.view(2, ht, wt)

    # 生成有效性掩码
    valid_depth_mask = (depth_vec > 0).view(ht, wt)
    valid_z_mask = (cam1_z_raw > 0).view(ht, wt)
    valid_x_mask = (projected_coords[0, :] >= 0) & (projected_coords[0, :] < wt)
    valid_y_mask = (projected_coords[1, :] >= 0) & (projected_coords[1, :] < ht)
    valid_proj_mask = (valid_x_mask & valid_y_mask).view(ht, wt)

    mask = valid_depth_mask & valid_z_mask & valid_proj_mask
    flow[:, ~mask] = 0.0

    return flow, mask

# ====================================================================
# BlendedMVS 数据集类 (与之前版本基本一致)
# ====================================================================
class BlendedMVSDataset(Dataset):
    """
    用于加载经过预处理后的 BlendedMVS 数据集。
    该类会动态计算光流作为真值。
    """
    def __init__(self, samples, root_dir, ht=360, wt=640, use_horizontal_flip_aug=False, normalize=True, debug=True):
        super().__init__()
        self.samples = samples
        self.root_dir = root_dir
        self.ht, self.wt = ht, wt
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.normalize = normalize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        seq_path = sample_info['seq_path']
        img0_name = sample_info['img0_name']
        img1_name = sample_info['img1_name']

        base_path0 = osp.join(seq_path, img0_name)
        base_path1 = osp.join(seq_path, img1_name)

        # 1. 加载图像、深度和相机参数
        im_A_orig = Image.open(base_path0 + ".jpg").convert("RGB")
        im_B_orig = Image.open(base_path1 + ".jpg").convert("RGB")
        depth_orig = cv2.imread(base_path0 + ".exr", cv2.IMREAD_ANYDEPTH)
        cam_params0 = np.load(base_path0 + ".npz")
        cam_params1 = np.load(base_path1 + ".npz")
        
        w_orig, h_orig = im_A_orig.size
        
        # 2. 将 numpy 数据转为 torch tensor
        depth_orig = torch.from_numpy(depth_orig.astype(np.float32)).float()
        K0 = torch.from_numpy(cam_params0['intrinsics']).float()
        R0 = torch.from_numpy(cam_params0['R_cam2world']).float()
        t0 = torch.from_numpy(cam_params0['t_cam2world']).float().view(3, 1)
        K1 = torch.from_numpy(cam_params1['intrinsics']).float()
        R1 = torch.from_numpy(cam_params1['R_cam2world']).float()
        t1 = torch.from_numpy(cam_params1['t_cam2world']).float().view(3, 1)

        # 3. 构建 4x4 位姿矩阵 (T_world_from_cam)
        T_world_from_cam0 = torch.cat([torch.cat([R0, t0], dim=1), torch.tensor([[0., 0., 0., 1.]])], dim=0)
        T_world_from_cam1 = torch.cat([torch.cat([R1, t1], dim=1), torch.tensor([[0., 0., 0., 1.]])], dim=0)

        # 4. 动态计算光流和掩码
        flow_orig, mask_orig = get_flow_from_depth_and_poses(
            depth_orig, K0, T_world_from_cam0, K1, T_world_from_cam1, h_orig, w_orig
        )

        # 5. 图像变换、缩放和归一化
        im_A = F.resize(im_A_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_B = F.resize(im_B_orig, [self.ht, self.wt], interpolation=F.InterpolationMode.BILINEAR)
        im_A, im_B = F.to_tensor(im_A), F.to_tensor(im_B)

        if self.normalize:
            im_A = F.normalize(im_A, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            im_B = F.normalize(im_B, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        sx, sy = self.wt / w_orig, self.ht / h_orig
        flow, mask = resize_flow_and_mask_weighted(flow_orig, mask_orig, [self.ht, self.wt], sx, sy)
        
        # 7. 水平翻转增强
        if self.use_horizontal_flip_aug and random.random() > 0.5:
            im_A, im_B = F.hflip(im_A), F.hflip(im_B)
            mask, flow = F.hflip(mask), F.hflip(flow)
            flow[0, :, :] *= -1

        return {
            'im_A': im_B,
            'im_B': im_A,
            'ground_truth_flow': flow.nan_to_num(0, 0, 0),
            'validity_mask': mask.float(),
        }

# ====================================================================
# 新增：用于验证数据完整性的辅助模块 (对齐官方实现)
# ====================================================================
class BlendedMVSInfo:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sequences = sorted([d for d in os.listdir(root_dir) if osp.isdir(osp.join(root_dir, d)) and d != '.cache'])

# ====================================================================
# BlendedMVS 数据集构建器 (已对齐官方实现)
# ====================================================================
class BlendedMVSBuilder:
    def __init__(self, data_root, pair_metadata_path=None):
        self.data_root = data_root
        self.pair_metadata_path = pair_metadata_path
        if not osp.exists(self.data_root):
            raise FileNotFoundError(f"BlendedMVS 预处理数据根目录未找到: {self.data_root}")
        if self.pair_metadata_path is None:
            self.pair_metadata_path = self.data_root.replace("blendedmvs_processed_ufm", "blendedmvs_pairs.npy")
        if not osp.exists(self.pair_metadata_path):
            raise FileNotFoundError(f"BlendedMVS 配对元数据文件未找到: {self.pair_metadata_path}")
        self.blendedmvs_info = BlendedMVSInfo(self.data_root)

    def build_dataset(self, split='train', **kwargs):
        cache_dir = osp.join(self.data_root, '.cache')
        cache_path = osp.join(cache_dir, f"blendedmvs_samples_{split}.pkl")

        if osp.exists(cache_path):
            print(f"--- BlendedMVS ({split}): 正在从缓存 '{cache_path}' 加载样本列表 ---")
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            print(f"--- BlendedMVS ({split}): 成功从缓存加载 {len(samples)} 个样本 ---")
        else:
            print(f"--- BlendedMVS ({split}): 未找到缓存，正在根据 '{self.pair_metadata_path}' 执行首次索引构建... ---")
            
            all_pairs = np.load(self.pair_metadata_path)
            
            # 1. 根据官方逻辑选择训练集或验证集
            if split == 'train':
                # 90% 的场景用于训练
                selection_mask = (all_pairs["seq_low"] % 10) > 0
            elif split == 'val':
                # 10% 的场景用于验证
                selection_mask = (all_pairs["seq_low"] % 10) == 0
            else:
                raise ValueError(f"未知的 split '{split}', 必须是 'train' 或 'val'")
            
            # 2. 筛选出在磁盘上真实存在的序列
            valid_seq_low_ids = {int(seq[8:], 16) for seq in self.blendedmvs_info.sequences}
            exist_mask = np.isin(all_pairs["seq_low"], list(valid_seq_low_ids))
            
            final_mask = selection_mask & exist_mask
            pairs = all_pairs[final_mask]
            
            # 3. 构建样本列表
            samples = []
            for pair in tqdm(pairs, desc=f"构建 BlendedMVS ({split}) 索引"):
                seqh, seql, img1_idx, img2_idx, _ = pair
                seq_name = f"{seqh:08x}{seql:016x}"
                seq_path = osp.join(self.data_root, seq_name)

                # 确保序列目录存在
                if not osp.isdir(seq_path):
                    continue

                samples.append({
                    'seq_path': seq_path,
                    'img0_name': f"{img1_idx:08d}",
                    'img1_name': f"{img2_idx:08d}",
                })

            print(f"--- BlendedMVS ({split}): 索引构建完成，共找到 {len(samples)} 个有效样本对。 ---")
            if samples:
                print(f"--- BlendedMVS ({split}): 正在将样本列表保存到缓存文件: '{cache_path}' ---")
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(samples, f)

        if not samples:
            print(f"警告: BlendedMVS 数据集 '{split}' 分割中没有找到任何样本。")
            return []
            
        dataset = BlendedMVSDataset(samples, root_dir=self.data_root, **kwargs)
        return [dataset]

# ====================================================================
# 使用示例
# ====================================================================
