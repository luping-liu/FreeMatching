import os
import os.path as osp
import re
import pickle
import random
import collections
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import cv2
from PIL import Image
from tqdm import tqdm

# ====================================================================
# 1. IO Utils
# ====================================================================
def read_flow_dynamic_replica(file_name):
    """ 读取光流 (uint16 container -> float16) """
    if not os.path.exists(file_name): return None
    try:
        with Image.open(file_name) as f:
            raw_data = np.array(f, dtype=np.uint16)
            flow_raw = np.frombuffer(raw_data, dtype=np.float16).astype(np.float32)
            w_raw, h_raw = f.size
            flow_raw = flow_raw.reshape((h_raw, w_raw))
            real_w = flow_raw.shape[1] // 2
            u = flow_raw[:, :real_w]
            v = flow_raw[:, real_w:]
            return np.stack([u, v], axis=-1)
    except: return None

def read_mask_dynamic_replica(file_name):
    """ 读取 Mask """
    if not os.path.exists(file_name): return None
    try:
        mask_im = Image.open(file_name)
        mask_np = np.array(mask_im)
        if mask_np.ndim == 3: mask_np = mask_np[:, :, 0]
        return (mask_np > 0)
    except: return None

def read_traj_dynamic_replica(pth_path):
    """
    读取 Trajectory 数据
    返回: coords (N, 2), vis (N,)
    """
    if not os.path.exists(pth_path): return None
    try:
        data = torch.load(pth_path, map_location='cpu', weights_only=True) 
        
        # 获取 2D 坐标
        coords = None
        if "traj_2d" in data:
            coords = data["traj_2d"]
            if coords.shape[-1] == 3: coords = coords[:, :2]
        elif "points" in data:
            coords = data["points"]
        
        if coords is None: return None

        # 获取可见性 Mask
        vis = None
        if "verts_inds_vis" in data:
            vis = data["verts_inds_vis"]
            vis = (vis > 0)
        
        # 转 numpy
        if isinstance(coords, torch.Tensor): coords = coords.numpy()
        if isinstance(vis, torch.Tensor): vis = vis.numpy()
        
        # 如果没有 vis 信息，默认全可见
        if vis is None:
            vis = np.ones(coords.shape[0], dtype=bool)
            
        return coords.astype(np.float32), vis.astype(bool)
    except: return None

# ====================================================================
# 2. Dataset Class
# ====================================================================
class DynamicReplicaDataset(data.Dataset):
    def __init__(self, samples, seq_meta, ht=384, wt=512, mode='dense', 
                 interval_range=1, augment=False, normalize=True, debug=True, cache_size=200):
        self.samples = samples
        self.seq_meta = seq_meta
        self.ht = ht
        self.wt = wt
        self.mode = mode
        self.augment = augment
        self.normalize = normalize
        
        if isinstance(interval_range, (list, tuple)):
            self.min_gap, self.max_gap = interval_range[0], interval_range[1]
            self.dynamic_interval = True
        else:
            self.min_gap, self.max_gap = interval_range, interval_range
            self.dynamic_interval = False
            
        self.cache_size = cache_size
        self.data_cache = collections.OrderedDict()

    def _get_cached_data(self, path, loader_func):
        if path in self.data_cache:
            self.data_cache.move_to_end(path)
            return self.data_cache[path]
        
        data = loader_func(path)
        
        if len(self.data_cache) >= self.cache_size:
            self.data_cache.popitem(last=False)
        
        if data is not None:
            self.data_cache[path] = data
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_idx = self.samples[idx]
        seq_name = sample_idx['seq_name']
        idx0 = sample_idx['idx0'] # 在 images 列表中的索引
        
        meta = self.seq_meta[seq_name]
        img_list = meta['images']
        img_dir = meta['img_dir']
        indices = meta['indices'] # 【重要】获取真实的帧号列表
        
        # 1. Time Interval
        seq_len = len(img_list)
        gap = self.min_gap
        direction = 1 
        
        if self.mode == 'sparse' and self.dynamic_interval:
            can_go_fwd = (idx0 + self.min_gap < seq_len)
            can_go_bwd = (idx0 - self.min_gap >= 0)
            directions = []
            if can_go_fwd: directions.append(1)
            if can_go_bwd: directions.append(-1)
            if not directions: return self.__getitem__((idx + 1) % len(self))
            direction = random.choice(directions)
            max_step = min(self.max_gap, seq_len - 1 - idx0 if direction == 1 else idx0)
            if max_step < self.min_gap: return self.__getitem__((idx + 1) % len(self))
            gap = random.randint(self.min_gap, max_step)
        
        idx1 = idx0 + (direction * gap)
        
        # 2. Load Images
        path0 = os.path.join(img_dir, img_list[idx0])
        path1 = os.path.join(img_dir, img_list[idx1])

        try:
            im_A_orig = Image.open(path0).convert("RGB")
            im_B_orig = Image.open(path1).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self))

        w_orig, h_orig = im_A_orig.size

        # Output containers
        flow_out = torch.zeros((2, self.ht, self.wt), dtype=torch.float32)
        mask_out = torch.zeros((self.ht, self.wt), dtype=torch.float32)

        sx = self.wt / w_orig
        sy = self.ht / h_orig
        scale_vec = np.array([sx, sy], dtype=np.float32)

        # ==================================================================
        # BACKWARD FLOW CONSTRUCTION (Target B -> Source A)
        # ==================================================================

        # --- DENSE MODE ---
        if self.mode == 'dense':
            flow_path = sample_idx.get('flow_path')
            mask_path = sample_idx.get('mask_path')
            if flow_path:
                flow_np = self._get_cached_data(flow_path, read_flow_dynamic_replica)
                if flow_np is not None:
                    mask_np = None
                    if mask_path: mask_np = self._get_cached_data(mask_path, read_mask_dynamic_replica)
                    if mask_np is None: mask_np = np.ones((h_orig, w_orig), dtype=bool)

                    ys, xs = np.nonzero(mask_np) 
                    if len(ys) > 0:
                        flow_vals = flow_np[ys, xs] 
                        src_xs_scaled = xs * sx
                        src_ys_scaled = ys * sy
                        flow_u_scaled = flow_vals[:, 0] * sx
                        flow_v_scaled = flow_vals[:, 1] * sy
                        target_xs = (src_xs_scaled + flow_u_scaled).round().astype(np.int64)
                        target_ys = (src_ys_scaled + flow_v_scaled).round().astype(np.int64)
                        bwd_u = -flow_u_scaled
                        bwd_v = -flow_v_scaled

                        valid_bounds = (target_xs >= 0) & (target_xs < self.wt) & \
                                       (target_ys >= 0) & (target_ys < self.ht)
                        target_xs, target_ys = target_xs[valid_bounds], target_ys[valid_bounds]
                        bwd_u, bwd_v = bwd_u[valid_bounds], bwd_v[valid_bounds]

                        if len(target_xs) > 0:
                            flow_out[0, target_ys, target_xs] = torch.from_numpy(bwd_u)
                            flow_out[1, target_ys, target_xs] = torch.from_numpy(bwd_v)
                            mask_out[target_ys, target_xs] = 1.0

        # --- SPARSE MODE (Fixed Name Matching) ---
        elif self.mode == 'sparse':
            traj_dir = meta['traj_dir']
            
            # ★★★ 核心修复：使用 indices 中的真实帧号 ★★★
            # 之前的错误：使用 img_list[idx] 的文件名来推断
            # 现在的逻辑：直接拿 indices[idx] (整数) -> 000000.pth
            frame_num0 = indices[idx0]
            frame_num1 = indices[idx1]
            
            t0_path = os.path.join(traj_dir, f"{frame_num0:06d}.pth")
            t1_path = os.path.join(traj_dir, f"{frame_num1:06d}.pth")
            
            data0 = self._get_cached_data(t0_path, read_traj_dynamic_replica)
            data1 = self._get_cached_data(t1_path, read_traj_dynamic_replica)

            if data0 is not None and data1 is not None:
                pts0, vis0 = data0 # (coords, vis_mask)
                pts1, vis1 = data1
                
                if len(pts0) == len(pts1):
                    # 1. 只有在两帧都可见的点才有效
                    # vis0 & vis1 确保点在 A 和 B 中都是可见的（没有被遮挡）
                    common_vis = vis0 & vis1
                    
                    if np.any(common_vis):
                        p0_valid = pts0[common_vis] # Source (A)
                        p1_valid = pts1[common_vis] # Target (B)

                        # 2. 图像范围检查 (Source A)
                        in_img_a = (p0_valid[:, 0] >= 0) & (p0_valid[:, 0] < w_orig) & \
                                   (p0_valid[:, 1] >= 0) & (p0_valid[:, 1] < h_orig)
                        
                        p0_valid = p0_valid[in_img_a]
                        p1_valid = p1_valid[in_img_a]

                        if len(p0_valid) > 0:
                            # 3. 坐标缩放
                            p0_scaled = p0_valid * scale_vec
                            p1_scaled = p1_valid * scale_vec

                            # 4. Backward Flow = P_source - P_target
                            flow_vecs = (p0_scaled - p1_scaled).astype(np.float32)

                            # 5. 投影到 B 网格
                            px = np.round(p1_scaled[:, 0]).astype(int)
                            py = np.round(p1_scaled[:, 1]).astype(int)

                            # 6. 边界检查 (Image B)
                            in_bounds = (px >= 0) & (px < self.wt) & (py >= 0) & (py < self.ht)
                            px, py = px[in_bounds], py[in_bounds]
                            flow_vecs = flow_vecs[in_bounds]

                            # 7. 填充
                            if len(px) > 0:
                                flow_out[0, py, px] = torch.from_numpy(flow_vecs[:, 0])
                                flow_out[1, py, px] = torch.from_numpy(flow_vecs[:, 1])
                                mask_out[py, px] = 1.0

        # 3. Transforms
        im_A = TF.resize(im_A_orig, [self.ht, self.wt], interpolation=TF.InterpolationMode.BILINEAR)
        im_B = TF.resize(im_B_orig, [self.ht, self.wt], interpolation=TF.InterpolationMode.BILINEAR)
        im_A, im_B = TF.to_tensor(im_A), TF.to_tensor(im_B)

        if self.normalize:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im_A = TF.normalize(im_A, mean, std)
            im_B = TF.normalize(im_B, mean, std)

        if self.augment and random.random() > 0.5:
            im_A, im_B = TF.hflip(im_A), TF.hflip(im_B)
            flow_out = TF.hflip(flow_out)
            mask_out = TF.hflip(mask_out)
            flow_out[0, :, :] *= -1 

        return {
            'im_A': im_A, 'im_B': im_B, 
            'ground_truth_flow': flow_out.nan_to_num(0), 
            'validity_mask': mask_out,  #.unsqueeze(0),
            'seq_name': seq_name
        }

# ====================================================================
# 3. Builder Class
# ====================================================================
class DynamicReplicaBuilder:
    def __init__(self, data_root):
        self.data_root = data_root
        if not os.path.exists(data_root):
            raise ValueError(f"Data root not found: {data_root}")

    def _get_frame_index(self, filename):
        stem = os.path.splitext(filename)[0]
        matches = re.findall(r'\d+', stem)
        if matches: return int(matches[-1])
        return None

    def build_dataset(self, split='train', mode='dense', time_interval=1, use_cache=True, save_cache=True, **kwargs):
        if isinstance(time_interval, (list, tuple)):
            min_gap, max_gap = time_interval[0], time_interval[1]
            interval_str = f"range_{min_gap}_{max_gap}"
        else:
            min_gap, max_gap = time_interval, time_interval
            interval_str = f"int_{min_gap}"

        split_dir_name = 'valid' if split in ['val', 'valid'] else 'train'
        split_path = os.path.join(self.data_root, split_dir_name)
        # 更改 Cache 名称以强制刷新
        cache_file = os.path.join(self.data_root, f".cache_dr_{split_dir_name}_{mode}_{interval_str}_Left_FixedNumTraj.pkl")
        
        samples, seq_meta = [], {}
        loaded = False

        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data_dict = pickle.load(f)
                    samples, seq_meta = data_dict['samples'], data_dict['seq_meta']
                loaded = True
                print(f"--- Cache loaded: {len(samples)} samples ---")
            except: loaded = False
        
        if not loaded:
            print(f"--- Indexing {split} (Mode: {mode}, LEFT VIEW ONLY) ---")
            if not os.path.exists(split_path): return []
            
            all_seqs = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
            seq_names = [s for s in all_seqs if 'left' in s.lower()]
            print(f"Found {len(all_seqs)} total seqs, filtering to {len(seq_names)} LEFT VIEW seqs.")
            
            for seq_name in tqdm(seq_names, desc="Indexing"):
                seq_dir = os.path.join(split_path, seq_name)
                img_dir = os.path.join(seq_dir, 'images')
                if not os.path.exists(img_dir): continue
                
                img_idx_map = {}
                for f in os.listdir(img_dir):
                    if f.endswith(('.png', '.jpg')):
                        idx = self._get_frame_index(f)
                        if idx is not None: img_idx_map[idx] = f
                if not img_idx_map: continue
                
                valid_indices = set(img_idx_map.keys())
                flow_dir, mask_dir, traj_dir = None, None, None
                flow_idx_map, mask_idx_map = {}, {}

                if mode == 'dense':
                    flow_dir = os.path.join(seq_dir, 'flow_forward')
                    mask_dir = os.path.join(seq_dir, 'flow_forward_mask')
                    if not os.path.exists(flow_dir): continue
                    for f in os.listdir(flow_dir):
                        if f.endswith(('.png', '.pfm', '.flo')):
                            idx = self._get_frame_index(f)
                            if idx is not None: flow_idx_map[idx] = f
                    valid_indices = valid_indices & set(flow_idx_map.keys())
                    if os.path.exists(mask_dir):
                        for f in os.listdir(mask_dir):
                            if f.endswith('.png'):
                                idx = self._get_frame_index(f)
                                if idx is not None: mask_idx_map[idx] = f
                
                elif mode == 'sparse':
                    traj_dir = os.path.join(seq_dir, 'trajectories')
                    if not os.path.exists(traj_dir): continue

                sorted_indices = sorted(list(valid_indices))
                if len(sorted_indices) < min_gap + 1: continue
                
                seq_images = [img_idx_map[idx] for idx in sorted_indices]
                seq_meta[seq_name] = {
                    'images': seq_images, 
                    'img_dir': img_dir,
                    'flow_dir': flow_dir, 'mask_dir': mask_dir, 'traj_dir': traj_dir,
                    'flow_map': flow_idx_map, 'mask_map': mask_idx_map,
                    'indices': sorted_indices # ★★★ 保存真实帧号 ★★★
                }

                limit = len(seq_images) - min_gap if mode == 'dense' else len(seq_images)
                for i in range(limit):
                    frame_idx = sorted_indices[i]
                    sample = {'seq_name': seq_name, 'idx0': i}
                    if mode == 'dense':
                        sample['flow_path'] = os.path.join(flow_dir, flow_idx_map[frame_idx])
                        if frame_idx in mask_idx_map: sample['mask_path'] = os.path.join(mask_dir, mask_idx_map[frame_idx])
                        else: sample['mask_path'] = None
                    samples.append(sample)

            if len(samples) > 0 and save_cache:
                with open(cache_file, 'wb') as f: pickle.dump({'samples': samples, 'seq_meta': seq_meta}, f)

        return [DynamicReplicaDataset(samples, seq_meta, mode=mode, interval_range=time_interval, cache_size=200, **kwargs), ]


# ====================================================================
# 4. Visualization
# ====================================================================
def visualize_correspondence(batch, output_path='vis_final.png'):
    images_a, images_b = batch['im_A'], batch['im_B']
    flows, masks = batch['ground_truth_flow'], batch['validity_mask']
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    vis_images = []

    batch_size = images_a.shape[0]
    if len(images_a.shape) == 3: 
        batch_size = 1
        images_a, images_b = images_a.unsqueeze(0), images_b.unsqueeze(0)
        flows, masks = flows.unsqueeze(0), masks.unsqueeze(0)

    for i in range(batch_size):
        def tensor_to_bgr(t):
            t = t.cpu() * std + mean
            img_np = t.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_a_bgr = tensor_to_bgr(images_a[i])
        img_b_bgr = tensor_to_bgr(images_b[i])
        flow = flows[i].cpu().permute(1, 2, 0).numpy()
        mask = masks[i].cpu().squeeze().numpy()
        
        valid_coords = np.argwhere(mask > 0)
        step = max(1, len(valid_coords) // 100)
        
        for j, (y, x) in enumerate(valid_coords[::step]):
            p1 = (int(x), int(y)) 
            u, v = flow[y, x, 0], flow[y, x, 1]
            p0 = (int(round(x + u)), int(round(y + v)))
            color = colors[j % len(colors)]
            cv2.circle(img_b_bgr, p1, 4, color, -1)
            cv2.circle(img_a_bgr, p0, 4, color, -1)

        combined = np.hstack((img_a_bgr, img_b_bgr))
        vis_images.append(torch.from_numpy(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)).permute(2, 0, 1))

    grid = vutils.make_grid(vis_images, nrow=1, normalize=False)
    vutils.save_image(grid.float() / 255.0, output_path)
    print(f"Saved visualization to {output_path}")


class DynamicReplicaDenseBuilder(DynamicReplicaBuilder):
    def build_dataset(self, split='train', mode='dense', time_interval=1, **kwargs):
        return super().build_dataset(split, 'dense', 1, **kwargs)
    

class DynamicReplicaSparseBuilder(DynamicReplicaBuilder):
    def build_dataset(self, split='train', mode='sparse', time_interval=[10, 40], **kwargs):
        return super().build_dataset(split, 'sparse', [10, 40], **kwargs)


