#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# build_scannetpp_mmap_cache.py
#
# 重建丢失的 scannet++ mmap_cache 生成器 (原 step1_reaggregate_metadata.py +
# step2_preprocess_for_mmap.py, 已据 scannetpp_ufm.py 的读取格式反推重写).
#
# 作用: 把 UFM preprocess_scannetpp.py 的输出(每个场景一个 scene_metadata.npz)
#       再聚合(reaggregate)并摊平(flatten)成 LongAlign loader 需要的 mmap_cache/.
#
# 为什么不直接用 UFM 的 all_metadata.npz:
#   UFM preprocess 并行跑(world_size>1)时, 每个 rank 末尾的 concat 只覆盖自己
#   那一片场景, all_metadata.npz 会被最后结束的 rank 覆盖成残缺版 —— 所以必须
#   从每个场景的 scene_metadata.npz 重新聚合(这正是 "reaggregate" 的由来).
#
# loader (scannetpp_ufm.py) 读取的 7 个产物:
#   mmap_cache/meta.json                 -> {all_scene_names:[...], all_image_names:[...]}
#   mmap_cache/all_sceneids.npy          (N_img,)      每张图属于第几个场景
#   mmap_cache/all_intrinsics.npy        (N_img,3,3)   每张图内参(opencv, 已去畸变+rescale)
#   mmap_cache/all_trajectories.npy      (N_img,4,4)   每张图 cam2world
#   mmap_cache/all_pairs.npy             (N_pair,2)    全局图像索引对
#   mmap_cache/flat_pair_indices.npy     (N_pair,)     按场景分组后的 pair 行索引
#   mmap_cache/pair_index_offsets.npy    (N_scene+1,)  每个场景在 flat_pair_indices 的切片边界
#
# 用法:
#   # 真正生成(UFM 预处理完成后跑):
#   python build_scannetpp_mmap_cache.py \
#       --ufm_output_dir /.../scannet++/preprocess_scannetpp_ufm/train
#   # 自检(不需要真实数据, 合成假场景验证索引自洽):
#   python build_scannetpp_mmap_cache.py --self_test
# ============================================================================
import argparse
import json
import os
import os.path as osp

import numpy as np


# ----------------------------------------------------------------------------
# step1: reaggregate —— 扫描每个场景的 scene_metadata.npz, 拼成全局数组
# ----------------------------------------------------------------------------
def reaggregate(ufm_output_dir, verbose=True):
    """
    遍历 ufm_output_dir/<scene>/scene_metadata.npz, 复刻 UFM 末尾的全局 concat
    (但覆盖所有场景, 不受并行 rank 残缺 all_metadata.npz 影响).

    每个 scene_metadata.npz 含:
        trajectories (n,4,4) cam2world
        intrinsics   (n,3,3) opencv, 去畸变+rescale 后
        images       (n,)    图像名(无扩展名, 如 'DSC01752')
        pairs        (e,2)   场景内 *局部* 索引对 (无全局偏移!)

    返回 dict, 含全局摊平数组 + 每个场景的 pair 数(用于切片偏移).
    """
    if not osp.isdir(ufm_output_dir):
        raise FileNotFoundError(f"UFM 输出目录不存在: {ufm_output_dir}")

    # 与 UFM 一致: 场景按名字排序, 保证确定性 & 与 all_metadata.npz 顺序一致
    scene_dirs = sorted(
        d for d in os.listdir(ufm_output_dir)
        if osp.isfile(osp.join(ufm_output_dir, d, "scene_metadata.npz"))
    )
    if not scene_dirs:
        raise RuntimeError(
            f"{ufm_output_dir} 下没找到任何 <scene>/scene_metadata.npz —— "
            f"UFM preprocess_scannetpp.py 跑完了吗?"
        )

    all_scene_names = []
    all_image_names = []          # 拼接的图像名(全局)
    all_sceneids = []             # 每张图的场景序号
    all_intrinsics = []           # (n,3,3) 列表, 后面 concat
    all_trajectories = []         # (n,4,4)
    all_pairs = []                # (e,2) 全局索引对
    pair_counts = []              # 每个场景的 pair 数(按场景顺序)

    img_offset = 0
    n_skip = 0
    for scene_idx, scene_id in enumerate(scene_dirs):
        meta_path = osp.join(ufm_output_dir, scene_id, "scene_metadata.npz")
        with np.load(meta_path, allow_pickle=True) as d:
            traj = d["trajectories"]
            intr = d["intrinsics"]
            imgs = d["images"]
            pairs = d["pairs"]

        num_imgs = int(imgs.shape[0])
        if num_imgs == 0:
            n_skip += 1
            continue  # 没有图像的场景跳过(否则会破坏 sceneid<->scene 对齐)

        # --- 校验形状自洽 ---
        assert intr.shape[0] == num_imgs, f"{scene_id}: intrinsics 数 {intr.shape[0]} != images 数 {num_imgs}"
        assert traj.shape[0] == num_imgs, f"{scene_id}: trajectories 数 {traj.shape[0]} != images 数 {num_imgs}"

        # --- pairs: 局部索引 -> 全局索引(加 img_offset), 复刻 UFM 行 413 ---
        pairs = np.asarray(pairs)
        if pairs.size == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        else:
            pairs = pairs.reshape(-1, 2).astype(np.int64)
            # pairs 是局部索引, 必须落在 [0, num_imgs)
            assert pairs.max() < num_imgs and pairs.min() >= 0, \
                f"{scene_id}: pair 索引越界 (max={pairs.max()}, num_imgs={num_imgs})"
            pairs = pairs + img_offset

        all_scene_names.append(str(scene_id))
        all_image_names.extend([str(x) for x in imgs.tolist()])
        all_sceneids.extend([len(all_scene_names) - 1] * num_imgs)
        all_intrinsics.append(intr.astype(np.float32))
        all_trajectories.append(traj.astype(np.float32))
        all_pairs.append(pairs)
        pair_counts.append(int(pairs.shape[0]))

        img_offset += num_imgs

    if not all_scene_names:
        raise RuntimeError("聚合后没有任何有效场景(都没有图像?)")

    agg = {
        "all_scene_names": all_scene_names,                                  # list[str], len=S
        "all_image_names": all_image_names,                                  # list[str], len=N_img
        "all_sceneids": np.asarray(all_sceneids, dtype=np.int64),            # (N_img,)
        "all_intrinsics": np.concatenate(all_intrinsics, axis=0),            # (N_img,3,3)
        "all_trajectories": np.concatenate(all_trajectories, axis=0),        # (N_img,4,4)
        "all_pairs": (np.concatenate(all_pairs, axis=0)
                      if all_pairs else np.zeros((0, 2), dtype=np.int64)),   # (N_pair,2)
        "pair_counts": pair_counts,                                          # list[int], len=S
    }
    if verbose:
        print(f"[reaggregate] 场景 {len(all_scene_names)} 个 (跳过空场景 {n_skip}) | "
              f"图像 {len(all_image_names)} | pair {agg['all_pairs'].shape[0]}")
    return agg


# ----------------------------------------------------------------------------
# step2: preprocess_for_mmap —— 写出 loader 直接 mmap 的 7 个文件
# ----------------------------------------------------------------------------
def write_mmap_cache(agg, cache_dir, verbose=True):
    os.makedirs(cache_dir, exist_ok=True)

    n_pair = int(agg["all_pairs"].shape[0])
    pair_counts = agg["pair_counts"]

    # pairs 是按场景顺序 concat 的 -> 天然按场景分组连续.
    # flat_pair_indices = 0..n_pair-1 (恒等), pair_index_offsets = 每场景 pair 数的前缀和.
    flat_pair_indices = np.arange(n_pair, dtype=np.int64)
    pair_index_offsets = np.concatenate(
        ([0], np.cumsum(np.asarray(pair_counts, dtype=np.int64)))
    ).astype(np.int64)

    # --- 自洽性兜底校验: flat_pair_indices[off[s]:off[s+1]] 必须全属于场景 s ---
    sceneids = agg["all_sceneids"]
    pairs = agg["all_pairs"]
    if n_pair > 0:
        pair_scene_of_a = sceneids[pairs[:, 0]]
        pair_scene_of_b = sceneids[pairs[:, 1]]
        assert np.array_equal(pair_scene_of_a, pair_scene_of_b), \
            "存在跨场景的 pair —— 不应该发生(UFM 只在场景内配对)"
        for s in range(len(pair_counts)):
            seg = flat_pair_indices[pair_index_offsets[s]:pair_index_offsets[s + 1]]
            if seg.size:
                assert np.all(pair_scene_of_a[seg] == s), f"场景 {s} 的 pair 分组错位"

    # --- 写文件 ---
    np.save(osp.join(cache_dir, "all_sceneids.npy"), sceneids)
    np.save(osp.join(cache_dir, "all_intrinsics.npy"), agg["all_intrinsics"].astype(np.float32))
    np.save(osp.join(cache_dir, "all_trajectories.npy"), agg["all_trajectories"].astype(np.float32))
    np.save(osp.join(cache_dir, "all_pairs.npy"), pairs.astype(np.int64))
    np.save(osp.join(cache_dir, "flat_pair_indices.npy"), flat_pair_indices)
    np.save(osp.join(cache_dir, "pair_index_offsets.npy"), pair_index_offsets)
    with open(osp.join(cache_dir, "meta.json"), "w") as f:
        json.dump(
            {"all_scene_names": agg["all_scene_names"],
             "all_image_names": agg["all_image_names"]},
            f,
        )

    if verbose:
        print(f"[write_mmap_cache] 已写出 7 个文件 -> {cache_dir}")
        print(f"    all_scene_names : {len(agg['all_scene_names'])}")
        print(f"    all_image_names : {len(agg['all_image_names'])}")
        print(f"    all_sceneids    : {sceneids.shape}")
        print(f"    all_intrinsics  : {agg['all_intrinsics'].shape}")
        print(f"    all_trajectories: {agg['all_trajectories'].shape}")
        print(f"    all_pairs       : {pairs.shape}")
        print(f"    flat_pair_indices : {flat_pair_indices.shape}")
        print(f"    pair_index_offsets: {pair_index_offsets.shape} (= n_scene+1)")


# ----------------------------------------------------------------------------
# 复刻 loader 的读取 + 索引数学, 用于自检(验证 mmap_cache 能被正确消费)
# ----------------------------------------------------------------------------
def verify_cache_against_loader_logic(cache_dir, original=None, n_probe=200, verbose=True):
    """模拟 scannetpp_ufm.py 的 Builder/Dataset 索引路径, 确认产物自洽."""
    with open(osp.join(cache_dir, "meta.json")) as f:
        meta = json.load(f)
    all_scene_names = meta["all_scene_names"]
    all_image_names = np.array(meta["all_image_names"], dtype=object)

    flat_pair_indices = np.load(osp.join(cache_dir, "flat_pair_indices.npy"))
    pair_index_offsets = np.load(osp.join(cache_dir, "pair_index_offsets.npy"))
    all_pairs = np.load(osp.join(cache_dir, "all_pairs.npy"))
    all_sceneids = np.load(osp.join(cache_dir, "all_sceneids.npy"))
    all_intrinsics = np.load(osp.join(cache_dir, "all_intrinsics.npy"))
    all_trajectories = np.load(osp.join(cache_dir, "all_trajectories.npy"))

    # loader 的 scene_start_offsets 推导方式(一字不差复刻)
    _, scene_image_counts = np.unique(all_sceneids, return_counts=True)
    scene_start_offsets = np.concatenate(([0], np.cumsum(scene_image_counts[:-1])))

    assert len(scene_start_offsets) == len(all_scene_names), \
        f"scene_start_offsets({len(scene_start_offsets)}) != all_scene_names({len(all_scene_names)})"
    assert len(pair_index_offsets) == len(all_scene_names) + 1, \
        "pair_index_offsets 长度应为 n_scene+1"
    assert all_intrinsics.shape[0] == len(all_image_names)
    assert all_trajectories.shape[0] == len(all_image_names)

    rng = np.random.default_rng(0)
    n_scene = len(all_scene_names)
    checked = 0
    for _ in range(n_probe):
        scene_idx = int(rng.integers(n_scene))
        start, end = pair_index_offsets[scene_idx], pair_index_offsets[scene_idx + 1]
        if end <= start:
            continue
        p = int(flat_pair_indices[rng.integers(start, end)])
        g1, g2 = all_pairs[p]
        # 两端图像都必须属于该场景
        assert all_sceneids[g1] == scene_idx and all_sceneids[g2] == scene_idx, \
            f"pair {p} 的端点不属于场景 {scene_idx}"
        # local idx + scene_images 切片 应能取回图像名
        sso = scene_start_offsets[scene_idx]
        nxt = scene_start_offsets[scene_idx + 1] if scene_idx + 1 < n_scene else len(all_image_names)
        scene_images = all_image_names[sso:nxt]
        l1, l2 = g1 - sso, g2 - sso
        assert 0 <= l1 < len(scene_images) and 0 <= l2 < len(scene_images), "局部索引越界"
        _ = scene_images[l1], scene_images[l2]
        # 内参/外参可索引且形状正确
        assert all_intrinsics[g1].shape == (3, 3) and all_trajectories[g1].shape == (4, 4)

        # 若提供了 original(自检合成数据), 进一步比对内容一致
        if original is not None:
            so = original["scene_offsets"][scene_idx]
            assert str(scene_images[l1]) == str(original["images"][scene_idx][l1])
            assert np.allclose(all_intrinsics[g1], original["intrinsics"][scene_idx][l1])
            assert np.allclose(all_trajectories[g2], original["trajectories"][scene_idx][l2])
            _ = so
        checked += 1

    if verbose:
        print(f"[verify] 索引自洽校验通过 (随机抽查 {checked} 个 pair, 覆盖 {n_scene} 场景)")
    return True


# ----------------------------------------------------------------------------
# 自检: 合成假场景 -> 写 scene_metadata.npz -> 跑全流程 -> 校验
# ----------------------------------------------------------------------------
def self_test():
    import tempfile
    print("=" * 70)
    print("self_test: 合成 4 个假场景, 验证 reaggregate + write + loader 索引数学")
    print("=" * 70)
    rng = np.random.default_rng(42)
    tmp = tempfile.mkdtemp(prefix="snpp_mmap_selftest_")
    ufm_out = osp.join(tmp, "train")
    os.makedirs(ufm_out, exist_ok=True)

    orig = {"images": [], "intrinsics": [], "trajectories": [], "scene_offsets": []}
    scene_names = ["0a76e06478", "0b031f3119", "1a8e0d78c0", "zz_emptyish"]
    off = 0
    for si, sc in enumerate(scene_names):
        n = int(rng.integers(3, 8))
        imgs = np.array([f"DSC{si:02d}{k:03d}" for k in range(n)], dtype="<U16")
        intr = rng.random((n, 3, 3)).astype(np.float32)
        traj = rng.random((n, 4, 4)).astype(np.float32)
        # 随机生成局部无向边(去重, i<j)
        edges = set()
        for _ in range(n * 2):
            a, b = sorted(rng.choice(n, 2, replace=False).tolist())
            if a != b:
                edges.add((a, b))
        pairs = np.array(sorted(edges), dtype=np.int64) if edges else np.zeros((0, 2), np.int64)
        os.makedirs(osp.join(ufm_out, sc), exist_ok=True)
        np.savez(osp.join(ufm_out, sc, "scene_metadata.npz"),
                 trajectories=traj, intrinsics=intr, images=imgs, pairs=pairs)
        orig["images"].append(imgs)
        orig["intrinsics"].append(intr)
        orig["trajectories"].append(traj)
        orig["scene_offsets"].append(off)
        off += n

    agg = reaggregate(ufm_out)
    cache_dir = osp.join(ufm_out, "mmap_cache")
    write_mmap_cache(agg, cache_dir)
    verify_cache_against_loader_logic(cache_dir, original=orig, n_probe=500)
    print(f"\n✅ self_test PASS. (临时目录: {tmp})")
    print("   说明: 真实运行时把 --ufm_output_dir 指向 UFM 预处理输出即可.")


def main():
    ap = argparse.ArgumentParser(description="重建 scannet++ mmap_cache (reaggregate + flatten)")
    ap.add_argument("--ufm_output_dir", type=str, default=None,
                    help="UFM preprocess_scannetpp.py 的输出目录(含各 <scene>/scene_metadata.npz), "
                         "通常是 .../preprocess_scannetpp_ufm/train")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="mmap_cache 输出目录, 默认 <ufm_output_dir>/mmap_cache")
    ap.add_argument("--self_test", action="store_true", help="不读真实数据, 合成假场景自检索引数学")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.ufm_output_dir:
        ap.error("需要 --ufm_output_dir (或用 --self_test)")
    cache_dir = args.cache_dir or osp.join(args.ufm_output_dir, "mmap_cache")

    agg = reaggregate(args.ufm_output_dir)
    write_mmap_cache(agg, cache_dir)
    verify_cache_against_loader_logic(cache_dir)
    print(f"\n✅ 完成. loader 可直接读取: {cache_dir}")
    print("   下一步: 把 unified_loader.py 的 scannet path 指向该 mmap_cache 的父目录")
    print(f"           ({osp.dirname(cache_dir)})")


if __name__ == "__main__":
    main()
