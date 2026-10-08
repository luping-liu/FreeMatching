import os
import glob
import json
import random
import numpy
import torch
import cv2
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# --- 您可能需要安装这个库: pip install tifffile ---
import tifffile
# --- 新增导入，用于图像和张量的缩放 ---
import torchvision.transforms.functional as F


# ====================================================================
# 最终优化版：修复了分辨率缩放问题
# ====================================================================

class VideoTrajectoryDataset(Dataset):
    """
    高效且健壮的视频数据集。
    本版本已修复光流和置信度图的分辨率缩放问题，使其能够正确处理自定义的 ht 和 wt。
    """
    def __init__(self, video_base_paths, virtual_epoch_len=50000, ht=360, wt=640, downsample_ratio=8, confidence_threshold=0.1, normalize=True, debug=True):
        self.video_base_paths = video_base_paths
        self.virtual_epoch_len = virtual_epoch_len
        self.ht, self.wt = ht, wt
        self.target_size_cv2 = (wt, ht)  # (W, H) for cv2
        self.target_size_torch = [ht, wt] # (H, W) for torch transforms
        self.downsample_ratio = downsample_ratio
        self.confidence_threshold = confidence_threshold
        self.normalize = normalize
        self.debug = debug
        
        # 元数据缓存
        self._metadata_cache = {}

    def __len__(self):
        return self.virtual_epoch_len

    def _get_metadata(self, sup_dir):
        # ... (此函数无需修改) ...
        if sup_dir in self._metadata_cache:
            return self._metadata_cache[sup_dir]
        metadata_path = os.path.join(sup_dir, 'metadata.json')
        if not os.path.exists(metadata_path): return None
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self._metadata_cache[sup_dir] = metadata
        return metadata

    def _dequantize_traj(self, traj_q, min_val, max_val):
        # ... (此函数无需修改) ...
        traj_norm = traj_q.astype(numpy.float32) / 32767.0
        return min_val + (traj_norm + 1.0) * 0.5 * (max_val - min_val)

    def _dequantize_conf(self, conf_q):
        # ... (此函数无需修改) ...
        return conf_q.astype(numpy.float32) / 255.0

    def __getitem__(self, idx):
        while True:
            try:
                # 1. 随机选择视频 (逻辑不变)
                video_base_path = random.choice(self.video_base_paths)
                sup_dir = video_base_path + '__supp'
                video_mp4_path = video_base_path + '.mp4'
                
                # 2. 动态检查并选择采样方向 (逻辑不变)
                fwd_tiffs = glob.glob(os.path.join(sup_dir, 'fwd-*.tiff'))
                bwd_tiffs = glob.glob(os.path.join(sup_dir, 'bwd-*.tiff'))
                options = []
                if len(fwd_tiffs) > 4: options.append(True)
                if len(bwd_tiffs) > 4: options.append(False)
                if not options: continue
                is_forward = random.choice(options)
                
                # 3. 随机选择轨迹文件 (逻辑不变)
                tiff_list = fwd_tiffs if is_forward else bwd_tiffs
                prefix = 'fwd' if is_forward else 'bwd'
                tiff_frame_index = random.randint(2, len(tiff_list) - 2)
                selected_tiff_path = os.path.join(sup_dir, f'{prefix}-{tiff_frame_index:03d}.tiff')
                
                # 4. 加载元数据和原始轨迹/置信度数据 (逻辑不变)
                metadata = self._get_metadata(sup_dir)
                if metadata is None: continue

                min_val, max_val = (metadata['traj_maps_e_min'], metadata['traj_maps_e_max']) if is_forward else (metadata['bwd_traj_maps_e_min'], metadata['bwd_traj_maps_e_max'])
                
                traj_x_abs = self._dequantize_traj(tifffile.imread(selected_tiff_path, key=0), min_val, max_val)
                traj_y_abs = self._dequantize_traj(tifffile.imread(selected_tiff_path, key=1), min_val, max_val)
                conf = self._dequantize_conf(tifffile.imread(selected_tiff_path, key=2))

                # ===================== 核心修复：光流和置信度的缩放逻辑 =====================
                h_orig, w_orig = conf.shape

                # 将 numpy 数组转为 tensor 以便使用 F.resize
                # unsqueeze(0) 增加一个通道维度
                traj_x_tensor = torch.from_numpy(traj_x_abs).float().unsqueeze(0)
                traj_y_tensor = torch.from_numpy(traj_y_abs).float().unsqueeze(0)
                conf_tensor = torch.from_numpy(conf).float().unsqueeze(0)

                # 使用 BILINEAR 插值进行空间缩放
                traj_x_resized = F.resize(traj_x_tensor.unsqueeze(0), self.target_size_torch, interpolation=F.InterpolationMode.BILINEAR).squeeze()
                traj_y_resized = F.resize(traj_y_tensor.unsqueeze(0), self.target_size_torch, interpolation=F.InterpolationMode.BILINEAR).squeeze()
                conf_resized = F.resize(conf_tensor.unsqueeze(0), self.target_size_torch, interpolation=F.InterpolationMode.BILINEAR).squeeze()

                # 计算缩放比例并调整轨迹（光流）幅度
                sx = self.wt / w_orig
                sy = self.ht / h_orig
                traj_x_scaled = traj_x_resized * sx
                traj_y_scaled = traj_y_resized * sy

                # 创建目标尺寸的网格并计算最终光流
                grid_y, grid_x = torch.meshgrid(torch.arange(self.ht), torch.arange(self.wt), indexing='ij')
                flow_x = traj_x_scaled - grid_x.float()
                flow_y = traj_y_scaled - grid_y.float()
                ground_truth_flow = torch.stack([flow_x, flow_y], dim=0)

                # 创建有效性掩码，并增加通道维度 (1, H, W)
                validity_mask = (conf_resized > self.confidence_threshold).float().unsqueeze(0)
                # ===================== 修复结束 =====================

                # 5. 计算源/目标帧索引 (逻辑不变)
                cap = cv2.VideoCapture(video_mp4_path)
                if not cap.isOpened(): continue
                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                base_idx = tiff_frame_index * self.downsample_ratio
                
                if is_forward:
                    source_video_idx, target_video_idx = base_idx, 0
                else: # is_backward
                    source_video_idx = total_video_frames - 1 - base_idx
                    target_video_idx = total_video_frames - 1

                if not (0 <= source_video_idx < total_video_frames and 0 <= target_video_idx < total_video_frames):
                    cap.release()
                    continue

                # 6. 加载视频帧 (逻辑不变)
                cap.set(cv2.CAP_PROP_POS_FRAMES, source_video_idx)
                ret1, source_frame_bgr = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_video_idx)
                ret2, target_frame_bgr = cap.read()
                cap.release()
                if not ret1 or not ret2: continue
                
                # 7. 预处理图像 (现在与缩放后的光流匹配)
                im_A = torch.from_numpy(cv2.cvtColor(cv2.resize(source_frame_bgr, self.target_size_cv2), cv2.COLOR_BGR2RGB)).permute(2,0,1).float() / 255.0
                im_B = torch.from_numpy(cv2.cvtColor(cv2.resize(target_frame_bgr, self.target_size_cv2), cv2.COLOR_BGR2RGB)).permute(2,0,1).float() / 255.0
                
                # print('videotrack:', ground_truth_flow.min() == torch.nan, ground_truth_flow.max() == torch.nan)
                return {
                    'im_A': im_A, 'im_B': im_B,
                    'ground_truth_flow': ground_truth_flow, 'validity_mask': validity_mask[0].float(),
                }
            except Exception as e:
                # 健壮性处理：如果发生任何错误，打印信息并尝试下一次随机抽样
                print(f"警告: 处理视频 {video_base_path} (idx {idx}) 时出错: {e}。正在重试...")
                continue


# ====================================================================
# 构建器 (无需修改)
# ====================================================================
class VideoTrajectoryBuilder:
    def __init__(self, data_root):
        self.root_dir = data_root

    def build_dataset(self, **kwargs):
        index_cache_file = os.path.join(self.root_dir, 'video_index_cache.txt')
        if os.path.exists(index_cache_file):
            print('--- 从缓存加载视频列表... ---')
            with open(index_cache_file, 'r') as f: video_base_paths = [line.strip() for line in f]
        else:
            print('--- 未找到缓存，正在扫描视频... ---')
            video_base_paths = [d.replace('__supp/metadata.json', '') for d in glob.glob(os.path.join(self.root_dir, '*__supp', 'metadata.json'))]
            print(f'--- 找到 {len(video_base_paths)} 个视频，正在缓存列表... ---')
            with open(index_cache_file, 'w') as f:
                for path in video_base_paths: f.write(f"{path}\n")
        
        if not video_base_paths:
            print(f"警告: 在目录 {self.root_dir} 中未找到任何有效的视频。")
            return []

        print(f'--- 数据集已初始化，包含 {len(video_base_paths)} 个视频。 ---')
        
        dataset = VideoTrajectoryDataset(video_base_paths, **kwargs)
        return [dataset]


# ====================================================================
# Test Code
# ====================================================================
