import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
import random
import time
import glob
from tqdm import tqdm
import argparse
import pickle
import multiprocessing

# ====================================================================
# UTILS BUNDLE (为简洁省略，请在实际运行时粘贴您提供的完整utils代码)
# ====================================================================
def farthest_point_sample_py(xyz, npoint):
    N, C = xyz.shape
    if N <= npoint: return np.arange(N)
    inds = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N, dtype=np.int32)
    for i in range(npoint):
        inds[i] = farthest
        centroid = xyz[farthest, :].reshape(1, C)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return inds

# ====================================================================
# 基础数据集类 (PointOdysseyDataset - 保持不变)
# ====================================================================
class PointOdysseyDataset(Dataset):
    # ... (与上一版完全相同，无需修改)
    def __init__(self, dataset_location, dset='train', use_augs=False, S=20, N=4096, 
                 strides=[1,2,4], clip_step=2, resize_size=(720, 1280), crop_size=(720, 1280), **kwargs):
        self.S, self.N = S, N
        self.use_augs = use_augs
        self.resize_size, self.crop_size = resize_size, crop_size
        self.rgb_paths, self.annotation_paths, self.full_idxs = [], [], []
        sequences = sorted([p for p in glob.glob(os.path.join(dataset_location, dset, "*")) if os.path.isdir(p)])
        for seq in sequences:
            rgb_path, anno_path = os.path.join(seq, 'rgbs'), os.path.join(seq, 'anno.npz')
            if os.path.isfile(anno_path):
                num_frames = len(glob.glob(os.path.join(rgb_path, '*.jpg')))
                for stride in strides:
                    for ii in range(0, num_frames - S*stride + 1, clip_step):
                        full_idx = ii + np.arange(S) * stride
                        if full_idx[-1] < num_frames:
                            self.rgb_paths.append([os.path.join(rgb_path, f'rgb_{idx:05d}.jpg') for idx in full_idx])
                            self.annotation_paths.append(anno_path)
                            self.full_idxs.append(full_idx)
    def __len__(self): return len(self.rgb_paths)
    def __getitem__(self, index):
        try:
            full_idx, anno = self.full_idxs[index], np.load(self.annotation_paths[index], allow_pickle=True)
            trajs, visibs, valids = [anno[k][full_idx].astype(np.float32) for k in ['trajs_2d', 'visibs', 'valids']]
        except Exception: return None, False
        trajs[np.isinf(trajs) | np.isnan(trajs)] = 0
        vis_and_val_initial = valids * visibs
        vis0 = vis_and_val_initial[0] > 0
        trajs, visibs, valids = trajs[:,vis0], visibs[:,vis0], valids[:,vis0]
        vis_and_val_filtered = valids * visibs
        val_ok = np.sum(vis_and_val_filtered, axis=0) >= max(np.sqrt(self.S), 2)
        trajs, visibs, valids = trajs[:, val_ok], visibs[:, val_ok], valids[:, val_ok]
        H, W = self.crop_size
        trajs_norm = trajs / max(H,W) if max(H,W) > 0 else trajs
        vel = trajs_norm[1:] - trajs_norm[:-1]
        accel = vel[1:] - vel[:-1]
        jerk = accel[1:] - accel[:-1]
        mot_ok = (np.max(np.linalg.norm(vel, axis=-1), axis=0) < 0.4) & \
                 (np.max(np.linalg.norm(accel, axis=-1), axis=0) < 0.3) & \
                 (np.max(np.linalg.norm(jerk, axis=-1), axis=0) < 0.1)
        trajs, visibs, valids = trajs[:, mot_ok], visibs[:, mot_ok], valids[:, mot_ok]
        N = trajs.shape[1]
        if N < self.N // 2: return None, False
        if N > self.N:
            xym = np.concatenate([trajs[0], np.mean(trajs[1:]-trajs[:-1],axis=0)], axis=-1)
            inds = farthest_point_sample_py(xym, self.N)
            trajs, visibs, valids = trajs[:,inds], visibs[:,inds], valids[:,inds]
        current_N = trajs.shape[1]
        if current_N < self.N:
            pad_n = self.N - current_N
            trajs = np.pad(trajs, ((0,0),(0,pad_n),(0,0)),'constant')
            visibs = np.pad(visibs, ((0,0),(0,pad_n)),'constant')
            valids = np.pad(valids, ((0,0),(0,pad_n)),'constant')
        sample={'trajs':trajs,'visibs':visibs,'valids':valids} # return numpy arrays
        return sample, True

# ====================================================================
# 【新】最终数据集类 (MemMap版本 - 已简化)
# ====================================================================
class PointOdysseyFinalDataset(Dataset):
    def __init__(self, dataset_location, dset='train', ht=288, wt=512, use_augs=False, S=20, normalize=True, **kwargs):
        self.ht, self.wt, self.S, self.use_augs, self.normalize = ht, wt, S, use_augs, normalize
        print(f"Initializing PointOdyssey Final Dataloader (MemMap, S={S}, Res: {ht}x{wt})...")
        cache_dir = osp.join(dataset_location, '.cache')
        metadata_path = osp.join(cache_dir, f'precomputed_metadata_S{S}_{dset}.pkl')
        if not osp.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件未找到: {metadata_path}\n请先使用 '--task preprocess' 模式运行此脚本。")
        with open(metadata_path, 'rb') as f: self.metadata = pickle.load(f)
        self.S, self.N, self.rgb_paths = self.metadata['S'], self.metadata['N'], self.metadata['rgb_paths']
        self.resize_size, self.crop_size = (int(ht*1.15),int(wt*1.15)), (ht,wt)
        self.trajs_mmap = np.memmap(self.metadata['trajs_path'], dtype=np.float32, mode='r', shape=self.metadata['trajs_shape'])
        self.visibs_mmap = np.memmap(self.metadata['visibs_path'], dtype=np.float32, mode='r', shape=self.metadata['visibs_shape'])
        self.valids_mmap = np.memmap(self.metadata['valids_path'], dtype=np.float32, mode='r', shape=self.metadata['valids_shape'])
        print(f"成功加载元数据并映射数据文件。共 {len(self.rgb_paths)} 个有效剪辑可用。")

    def __len__(self): return len(self.rgb_paths)
    def __getitem__(self, index):
        idx0, idx1 = random.sample(range(self.S), 2)
        try:
            rgb_path0, rgb_path1 = self.rgb_paths[index][idx0], self.rgb_paths[index][idx1]
            im_0, im_1 = np.array(Image.open(rgb_path0).convert('RGB')), np.array(Image.open(rgb_path1).convert('RGB'))
            rgbs = [im_0, im_1]
            trajs = self.trajs_mmap[index][[idx0, idx1]]
            visibs = self.visibs_mmap[index][[idx0, idx1]]
            valids = self.valids_mmap[index][[idx0, idx1]]
        except Exception: return self.__getitem__(random.randint(0, len(self)-1))
        if self.use_augs: rgbs, trajs = self.apply_spatial_augs(rgbs, trajs)
        else: rgbs, trajs = self.just_resize(rgbs, trajs)
        im_A,im_B = torch.from_numpy(rgbs[0].copy()).permute(2,0,1), torch.from_numpy(rgbs[1].copy()).permute(2,0,1)
        im_A,im_B=[im.float()/255.0 for im in [im_A,im_B]]
        if self.normalize:
            im_A,im_B=[TF.normalize(im,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]) for im in [im_A,im_B]]
        flow,mask=torch.zeros((2,self.ht,self.wt),dtype=torch.float32),torch.zeros((self.ht,self.wt),dtype=torch.bool)
        c0,c1,v0,v1,d0,d1=[torch.from_numpy(x) for x in [trajs[0],trajs[1],visibs[0],visibs[1],valids[0],valids[1]]]
        valid_mask=(v0>0)&(d0>0)&(v1>0)&(d1>0); vc0,vc1=c0[valid_mask],c1[valid_mask]
        if vc1.shape[0]>0:
            flow_vecs=vc0-vc1; px,py=torch.round(vc1[:,0]).long(),torch.round(vc1[:,1]).long()
            coord_mask=(px>=0)&(px<self.wt)&(py>=0)&(py<self.ht)
            px,py,flow_vecs=px[coord_mask],py[coord_mask],flow_vecs[coord_mask]
            flow[0,py,px],flow[1,py,px]=flow_vecs[:,0],flow_vecs[:,1]; mask[py,px]=True
        return {'im_A':im_A,'im_B':im_B,'ground_truth_flow':flow,'validity_mask':mask.float()}
    def just_resize(self,rgbs,trajs):
        H_new,W_new=self.crop_size; sx,sy=W_new/rgbs[0].shape[1],H_new/rgbs[0].shape[0]
        rgbs=[cv2.resize(r,(W_new,H_new),interpolation=cv2.INTER_LINEAR) for r in rgbs]
        trajs[...,0]*=sx; trajs[...,1]*=sy; return rgbs,trajs
    def apply_spatial_augs(self,rgbs,trajs):
        H_resize,W_resize=self.resize_size; sx,sy=W_resize/rgbs[0].shape[1],H_resize/rgbs[0].shape[0]
        rgbs=[cv2.resize(r,(W_resize,H_resize),interpolation=cv2.INTER_LINEAR) for r in rgbs]
        trajs[...,0]*=sx; trajs[...,1]*=sy; H_crop,W_crop=self.crop_size
        y0=np.random.randint(0,H_resize-H_crop+1) if H_resize>H_crop else 0
        x0=np.random.randint(0,W_resize-W_crop+1) if W_resize>W_crop else 0
        rgbs=[r[y0:y0+H_crop,x0:x0+W_crop] for r in rgbs]; trajs[...,0]-=x0; trajs[...,1]-=y0
        return rgbs,trajs

# ====================================================================
# 数据集构建器 (保持不变)
# ====================================================================
class PointOdysseyBuilder:
    # ... (与上一版完全相同，无需修改)
    def __init__(self, data_root):
        self.data_root = data_root
        if not osp.exists(self.data_root): raise FileNotFoundError(f"数据集根目录未找到: {self.data_root}")
    def build_dataset(self, split='train', **kwargs):
        print(f"--- PointOdyssey Builder: 正在构建 '{split}' 数据集 ---")
        dataset = PointOdysseyFinalDataset(self.data_root, dset=split, **kwargs)
        return [dataset]
        
# ====================================================================
# 【新】多进程预处理工作函数
# ====================================================================
def process_chunk(args_tuple):
    worker_id, chunk_indices, dataset_args, tmp_dir = args_tuple
    base_dataset = PointOdysseyDataset(**dataset_args)
    
    valid_trajs, valid_visibs, valid_valids = [], [], []
    valid_original_indices = []

    # 【修改点】根据 worker_id 决定是否创建进度条
    iterable = chunk_indices
    if worker_id == 0:
        # 只为 worker 0 创建进度条，并放在第 1 行 (主进度条下方)
        iterable = tqdm(chunk_indices, desc="Worker 0 进度", position=1, leave=False)

    for index in iterable:
        sample, gotit = base_dataset[index]
        if gotit and sample is not None:
            valid_trajs.append(sample['trajs'])
            valid_visibs.append(sample['visibs'])
            valid_valids.append(sample['valids'])
            valid_original_indices.append(index)
    
    if valid_trajs:
        np.save(osp.join(tmp_dir, f'trajs_{worker_id}.npy'), np.stack(valid_trajs))
        np.save(osp.join(tmp_dir, f'visibs_{worker_id}.npy'), np.stack(valid_visibs))
        np.save(osp.join(tmp_dir, f'valids_{worker_id}.npy'), np.stack(valid_valids))
        np.save(osp.join(tmp_dir, f'indices_{worker_id}.npy'), np.array(valid_original_indices, dtype=np.int32))

    return True

# ====================================================================
# 【已修正】多进程预处理主函数 - 为主进度条指定位置
# ====================================================================
def preprocess_point_odyssey_multiprocess(dataset_location, dset='train', S=20, N=4096, num_workers=None):
    print("="*60 + f"\n开始预计算 Point Odyssey '{dset}' (S={S}) - [多进程模式]\n" + "="*60)
    
    if num_workers is None:
        num_workers = os.cpu_count()
    print(f"将使用 {num_workers} 个 worker 进程。")

    cache_dir = osp.join(dataset_location, '.cache')
    tmp_dir = osp.join(cache_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    metadata_path = osp.join(cache_dir, f'precomputed_metadata_S{S}_{dset}.pkl')
    
    if osp.exists(metadata_path):
        if input(f"警告: 缓存文件 '{metadata_path}' 已存在。\n强制重新计算请输入 'yes': ").lower() != 'yes':
            print("跳过预计算。"); return

    dataset_args = {'dataset_location': dataset_location, 'dset': dset, 'S': S, 'N': N}
    base_dataset_info = PointOdysseyDataset(**dataset_args)
    num_clips = len(base_dataset_info)
    if num_clips == 0: print("未找到任何剪辑。"); return
    
    indices = np.arange(num_clips)
    chunks = np.array_split(indices, num_workers)
    
    args_list = [(i, chunks[i], dataset_args, tmp_dir) for i in range(num_workers)]

    print("阶段 1/3: 并行处理数据块...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        # 【修改点】为主进度条添加 position=0
        main_pbar = tqdm(pool.imap_unordered(process_chunk, args_list), 
                         total=len(args_list), 
                         desc="主进程分块进度", 
                         position=0)
        list(main_pbar) # 使用 list() 来消费迭代器并等待所有任务完成

    print("\n\n阶段 2/3: 合并临时文件...") # 添加换行符以避免覆盖 worker 进度条
    # ... (合并和清理的逻辑保持不变)
    all_valid_trajs, all_valid_visibs, all_valid_valids, all_valid_indices = [], [], [], []
    for i in tqdm(range(num_workers), desc="合并中"):
        if osp.exists(osp.join(tmp_dir, f'indices_{i}.npy')):
            all_valid_indices.append(np.load(osp.join(tmp_dir, f'indices_{i}.npy')))
            all_valid_trajs.append(np.load(osp.join(tmp_dir, f'trajs_{i}.npy')))
            all_valid_visibs.append(np.load(osp.join(tmp_dir, f'visibs_{i}.npy')))
            all_valid_valids.append(np.load(osp.join(tmp_dir, f'valids_{i}.npy')))
    
    if not all_valid_indices: print("错误：未能处理任何有效数据。"); return

    final_indices = np.concatenate(all_valid_indices)
    final_trajs = np.concatenate(all_valid_trajs)
    final_visibs = np.concatenate(all_valid_visibs)
    final_valids = np.concatenate(all_valid_valids)

    # 排序以保证原始rgb_paths和最终数据的对应关系
    sort_order = np.argsort(final_indices)
    final_indices = final_indices[sort_order]
    final_trajs = final_trajs[sort_order]
    final_visibs = final_visibs[sort_order]
    final_valids = final_valids[sort_order]
    
    num_valid_clips = len(final_indices)
    trajs_shape = (num_valid_clips, S, N, 2)
    visibs_shape = (num_valid_clips, S, N)
    
    trajs_mmap_path = osp.join(cache_dir, f'trajs_S{S}_{dset}.mmap')
    visibs_mmap_path = osp.join(cache_dir, f'visibs_S{S}_{dset}.mmap')
    valids_mmap_path = osp.join(cache_dir, f'valids_S{S}_{dset}.mmap')
    
    trajs_mmap = np.memmap(trajs_mmap_path, dtype=np.float32, mode='w+', shape=trajs_shape)
    visibs_mmap = np.memmap(visibs_mmap_path, dtype=np.float32, mode='w+', shape=visibs_shape)
    valids_mmap = np.memmap(valids_mmap_path, dtype=np.float32, mode='w+', shape=visibs_shape)

    trajs_mmap[:] = final_trajs; visibs_mmap[:] = final_visibs; valids_mmap[:] = final_valids
    
    metadata = {
        'S': S, 'N': N,
        'rgb_paths': [base_dataset_info.rgb_paths[i] for i in final_indices], # 只保存有效剪辑的路径
        'trajs_path': trajs_mmap_path, 'trajs_shape': trajs_shape,
        'visibs_path': visibs_mmap_path, 'visibs_shape': visibs_shape,
        'valids_path': valids_mmap_path, 'valids_shape': visibs_shape,
    }
    with open(metadata_path, 'wb') as f: pickle.dump(metadata, f)
    
    print("阶段 3/3: 清理临时文件...")
    # ... (清理逻辑保持不变)
    for f in glob.glob(os.path.join(tmp_dir, '*.npy')):
        os.remove(f)
    os.rmdir(tmp_dir)
    
    print("\n预计算完成！缓存已保存在:", cache_dir)

# ====================================================================
# 主执行函数
# ====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train', choices=['preprocess', 'train'])
    parser.add_argument('--data_root', default="/path/to/your/point_odyssey_v1.2")
    parser.add_argument('--s_frames', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=None, help="预处理时使用的进程数，默认为CPU核心数。")
    args = parser.parse_args()

    if not osp.exists(args.data_root) or "path/to" in args.data_root:
        print("="*60 + f"\n错误: 请通过 --data_root 参数指定您的实际数据集路径！\n" + "="*60); return

    if args.task == 'preprocess':
        preprocess_point_odyssey_multiprocess(args.data_root, dset='train', S=args.s_frames, num_workers=args.num_workers)

    elif args.task == 'train':
        builder = PointOdysseyBuilder(data_root=args.data_root)
        train_datasets = builder.build_dataset(split='train', ht=288, wt=512, use_augs=True, S=args.s_frames)
        train_loader = DataLoader(train_datasets[0], batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
        print(f"\n--- 使用最终 MemMap 版加载器 (分辨率 {train_datasets[0].ht}x{train_datasets[0].wt}) 进行速度测试 ---")
        start_time = time.time()
        num_batches_to_test = 20
        data_iter = iter(train_loader)
        for i in range(num_batches_to_test):
            batch_start_time = time.time()
            batch = next(data_iter)
            batch_end_time = time.time()
            print(f"Batch {i+1}/{num_batches_to_test} 加载成功, 耗时: {batch_end_time - batch_start_time:.4f} 秒")
        total_time = time.time() - start_time
        print(f"\n加载 {num_batches_to_test} 个批次的平均时间: {total_time / num_batches_to_test:.4f} 秒/批次")

if __name__ == '__main__':
    main()
