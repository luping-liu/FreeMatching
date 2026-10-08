import argparse
import os
import os.path as osp
import re
import sys
import time

import cv2
import numpy as np
import PIL.Image as Image
import trimesh
import trimesh.exchange.ply
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


# ============================================================================
# 渲染后端抽象: cuda(nvdiffrast, 生产默认) / osmesa(pyrender, 仅对照测速用)
# 两个后端吃完全相同的 (mesh, cam_to_world, K, W, H), 返回 (H,W) float32 米 深度.
# ============================================================================
class _CudaBackend:
    name = "cuda(nvdiffrast)"
    def __init__(self, verbose=False):
        from render_depth_nvdiffrast import DepthRendererCuda
        self.r = DepthRendererCuda()
        self.r.calibrate(verbose=verbose)   # 锁定 flip_v(应为 False)
    def set_mesh(self, mesh):
        self.r.set_mesh(np.asarray(mesh.vertices, np.float32),
                        np.asarray(mesh.faces, np.int32))
    def render(self, c2w, K, W, H, znear, zfar):
        return self.r.render(np.asarray(c2w, np.float64), np.asarray(K, np.float64),
                             W, H, znear=znear, zfar=zfar)


class _OsmesaBackend:
    name = "osmesa(pyrender)"
    def __init__(self, verbose=False):
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        import pyrender
        self.pyrender = pyrender
        self.scene = None
        self.renderer = None   # 延迟到首帧(osmesa 不能 0x0 建上下文)
    def set_mesh(self, mesh):
        pr = self.pyrender
        pmesh = pr.Mesh.from_trimesh(mesh, smooth=False)
        self.scene = pr.Scene()
        self.scene.add(pmesh)
    def render(self, c2w, K, W, H, znear, zfar):
        pr = self.pyrender
        if self.renderer is None:
            self.renderer = pr.OffscreenRenderer(W, H)
        self.renderer.viewport_width, self.renderer.viewport_height = W, H
        K = np.asarray(K, np.float64)
        cam = pr.camera.IntrinsicsCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                                         znear=znear, zfar=zfar)
        node = self.scene.add(cam, pose=np.asarray(c2w, np.float64) @ OPENGL_TO_OPENCV)
        depth = self.renderer.render(self.scene, flags=pr.RenderFlags.DEPTH_ONLY)
        self.scene.remove_node(node)
        return depth


def make_backend(name, verbose=False):
    return _CudaBackend(verbose) if name == "cuda" else _OsmesaBackend(verbose)


inv = np.linalg.inv
REGEXPR_DSLR = re.compile(r"^DSC(?P<frameid>\d+).JPG$")
REGEXPR_IPHONE = re.compile(r"frame_(?P<frameid>\d+).jpg$")

# ---- 内参约定转换(geometry.py 逐字) ----
def colmap_to_opencv_intrinsics(K):
    K = K.copy(); K[0, 2] -= 0.5; K[1, 2] -= 0.5; return K

def opencv_to_colmap_intrinsics(K):
    K = K.copy(); K[0, 2] += 0.5; K[1, 2] += 0.5; return K


# ---- cropping.py 逐字 ----
try:
    _lanczos = Image.Resampling.LANCZOS
    _bicubic = Image.Resampling.BICUBIC
except AttributeError:
    _lanczos = Image.LANCZOS
    _bicubic = Image.BICUBIC


class ImageList:
    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            self.images.append(image)
    def __len__(self): return len(self.images)
    def to_pil(self): return tuple(self.images) if len(self.images) > 1 else self.images[0]
    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes); return sizes[0]
    def resize(self, *a, **k): return ImageList(self._dispatch("resize", *a, **k))
    def _dispatch(self, func, *a, **k): return [getattr(im, func)(*a, **k) for im in self.images]


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution,
                          scaling=1, offset_factor=0.5, offset=None):
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins
    out = opencv_to_colmap_intrinsics(input_camera_matrix)
    out[:2, :] *= scaling
    out[:2, 2] -= offset
    return colmap_to_opencv_intrinsics(out)


def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True):
    image = ImageList(image)
    input_resolution = np.array(image.size)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        assert tuple(depthmap.shape[:2]) == image.size[::-1]
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)
    image = image.resize(tuple(output_resolution), resample=_lanczos if scale_final < 1 else _bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final, fy=scale_final,
                              interpolation=cv2.INTER_NEAREST)
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)
    return image.to_pil(), depthmap, camera_intrinsics


# ---- preprocess_scannetpp.py 逐字 ----
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannetpp_dir", required=True,
                        help="含 data/<scene>/{dslr,scans} 的 raw 根 (= $NEWROOT/scannet++/raw)")
    parser.add_argument("--precomputed_anymap_pairs", required=True,
                        help="snpp_aggregated_metadata_train.npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_resolution", default=980, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--backend", choices=["cuda", "osmesa"], default="cuda",
                        help="渲染后端: cuda=nvdiffrast(生产), osmesa=pyrender(仅对照测速)")
    parser.add_argument("--max_scenes", default=0, type=int,
                        help=">0 时只处理前 N 个场景(测速/小试用)")
    parser.add_argument("--timing", action="store_true", help="打印分段计时(渲染 vs IO 等)")
    return parser


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)


def get_frame_number(name, cam_type="dslr"):
    name = name.split("_")[-1]
    regex_expr = REGEXPR_DSLR if cam_type == "dslr" else REGEXPR_IPHONE
    return re.match(regex_expr, name)["frameid"]


def load_sfm(sfm_dir, cam_type="dslr"):
    with open(osp.join(sfm_dir, "cameras.txt"), "r") as f:
        raw = f.read().splitlines()[3:]
    intrinsics = {}
    for camera in raw:
        camera = camera.split(" ")
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]
    with open(os.path.join(sfm_dir, "images.txt"), "r") as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith("#")]
    img_idx = {}; img_infos = {}
    for image, points in zip(raw[0::2], raw[1::2]):
        image = image.split(" ")
        idx = image[0]; img_name = image[-1]
        assert img_name not in img_idx, "duplicate db image: " + img_name
        img_idx[img_name] = idx
        img_infos[idx] = dict(
            intrinsics=intrinsics[int(image[-2])],
            path=img_name,
            frame_id=get_frame_number(img_name, cam_type),
            cam_to_world=pose_from_qwxyz_txyz(image[1:-2]),
        )
    return img_idx, img_infos, None, None


def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]
    width = int(intrinsics[1]); height = int(intrinsics[2])
    fx = intrinsics[3]; fy = intrinsics[4]; cx = intrinsics[5]; cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])
    K = np.zeros([3, 3]); K[0, 0] = fx; K[0, 2] = cx; K[1, 1] = fy; K[1, 2] = cy; K[2, 2] = 1
    K = colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion, (width, height), np.eye(3), balance=0.0)
        new_K[0, 2] = width / 2.0; new_K[1, 2] = height / 2.0
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 1, (width, height), True)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
    undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    new_K = opencv_to_colmap_intrinsics(new_K)
    return width, height, new_K, undistorted_image, undistorted_mask


def adjacency_list_to_array(adj_list):
    edges = set()
    for node, neighbors in adj_list.items():
        for neighbor in neighbors:
            edges.add(tuple(sorted((node, neighbor))))
    return np.array(list(edges))


def process_scenes(root, pairs_npz, output_dir, target_resolution, rank=0, world_size=1,
                   backend="cuda", max_scenes=0, timing=False):
    os.makedirs(output_dir, exist_ok=True)

    pairs_data = np.load(pairs_npz, allow_pickle=True)
    pairs_data = {k.split("/")[-1]: v for k, v in pairs_data.items()}
    pairs_keys_list = sorted(list(pairs_data.keys()))
    pairs_keys_list = np.array_split(pairs_keys_list, world_size)[rank]
    if max_scenes > 0:
        pairs_keys_list = pairs_keys_list[:max_scenes]

    znear = 0.05
    zfar = 20.0

    # === 渲染后端(cuda=nvdiffrast / osmesa=pyrender), 唯一变量 ===
    renderer = make_backend(backend, verbose=(rank == 0))
    if rank == 0:
        print(f"[rank0] 渲染后端: {renderer.name}")

    t_render = 0.0
    n_frames = 0
    t_scene_start = time.time()

    for scene_id in tqdm(pairs_keys_list, position=0, desc=f"[rank{rank}] scenes"):
        data_dir = os.path.join(root, "data", scene_id)
        dir_dslr = os.path.join(data_dir, "dslr")
        dir_scans = os.path.join(data_dir, "scans")

        scene_pairs = pairs_data[scene_id]
        adjacency_list = scene_pairs.item()["adjacency_list"]
        file_names_original = scene_pairs.item()["file_names"]

        output_dir_scene = os.path.join(output_dir, scene_id)
        output_dir_scene_rgb = os.path.join(output_dir_scene, "images")
        output_dir_scene_depth = os.path.join(output_dir_scene, "depth")

        # 幂等: 已完成(scene_metadata.npz) 或 已知坏场景(.skipped) 都跳过.
        # 注意: .cuda_failed 不跳过 —— CUDA 错误可能只是当时 context 被别的场景污染,
        # 换次运行(干净 context)可能就好了, 给它重试机会.
        if osp.exists(osp.join(output_dir_scene, "scene_metadata.npz")):
            continue
        if osp.exists(osp.join(output_dir_scene, ".skipped")):
            continue
        os.makedirs(output_dir_scene_rgb, exist_ok=True)
        os.makedirs(output_dir_scene_depth, exist_ok=True)

        ply_path = os.path.join(dir_scans, "mesh_aligned_0.05.ply")
        sfm_dir_dslr = os.path.join(dir_dslr, "colmap")
        rgb_dir_dslr = os.path.join(dir_dslr, "resized_images")
        mask_dir_dslr = os.path.join(dir_dslr, "resized_anon_masks")

        try:
            with open(ply_path, "rb") as f:
                mesh_kwargs = trimesh.exchange.ply.load_ply(f)
            mesh_scene = trimesh.Trimesh(**mesh_kwargs)
            img_idx_dslr, img_infos_dslr, _, _ = load_sfm(sfm_dir_dslr, cam_type="dslr")
        except FileNotFoundError:
            print(f"Skipping {scene_id} as colmap reconstruction not found")
            continue

        # === 把 mesh 送进渲染后端(cuda: 顶点/面上 GPU; osmesa: 建 pyrender.Scene) ===
        renderer.set_mesh(mesh_scene)

        # img_idx: colmap 图名 -> 索引. colmap key 形如 'DSC04469.JPG'.
        # 建多种 key 容错: 去扩展名 + 末段去前缀, 兼容 file_names 的写法.
        img_idx = {}
        for k, v in img_idx_dslr.items():
            stem = k.split(".")[0]                 # DSC04469.JPG -> DSC04469
            img_idx[stem] = v
            img_idx[stem.split("_")[-1]] = v
        img_infos = img_infos_dslr

        # file_names 可能带场景前缀(159/514 个场景如 'd792f7be_DSC00017')也可能纯名.
        # 规整成 colmap 能查到的 key: 去扩展名 + 去前缀(取末段 DSCxxxxx).
        def _to_key(x):
            return str(x).split(".")[0].split("_")[-1]
        selection_keys = [_to_key(x) for x in file_names_original]
        selection_cam = [str(x).split(".")[0] for x in file_names_original]

        rgb_dir = rgb_dir_dslr
        mask_dir = mask_dir_dslr

        # 严格保持与 file_names_original 的【位置对齐】(pairs 索引指向该顺序, 不能丢条目).
        # 任一图缺失/坏 -> 跳过整个场景(不产出错位 metadata). CUDA 错误记 marker 防重试死循环.
        try:
            intrinsics_undistorted = []
            trajectories = []
            for key, imgname in tqdm(list(zip(selection_keys, selection_cam)),
                                     position=1, leave=False):
                if key not in img_idx:
                    raise KeyError(f"{imgname} 不在 colmap")
                imgidx = img_idx[key]
                img_infos_idx = img_infos[imgidx]
                rgb = np.array(Image.open(os.path.join(rgb_dir, img_infos_idx["path"])))
                mask = np.array(Image.open(os.path.join(mask_dir, img_infos_idx["path"][:-3] + "png")))

                _, _, K, rgb, mask = undistort_images(img_infos_idx["intrinsics"], rgb, mask)

                intrinsics = colmap_to_opencv_intrinsics(K)
                image, mask, intrinsics = rescale_image_depthmap(
                    rgb, mask, intrinsics, (target_resolution, target_resolution * 3.0 / 4))
                W, H = image.size

                rgb_outpath = os.path.join(output_dir_scene_rgb, img_infos_idx["path"][:-3] + "jpg")
                depth_outpath = os.path.join(output_dir_scene_depth, img_infos_idx["path"][:-3] + "png")

                # === 渲深度: 后端统一接口(cuda 内部 opencv 约定已验证等价 pyrender) ===
                K_np = np.asarray(intrinsics, np.float64)
                c2w = np.asarray(img_infos_idx["cam_to_world"], np.float64)
                _t = time.time()
                depth = renderer.render(c2w, K_np, W, H, znear=znear, zfar=zfar)
                t_render += time.time() - _t
                n_frames += 1

                depth = (depth * 1000).astype("uint16")
                depth[mask < 255] = 0
                image.save(rgb_outpath)
                Image.fromarray(depth).save(depth_outpath)

                intrinsics_undistorted.append(intrinsics)
                trajectories.append(img_infos_idx["cam_to_world"])
        except Exception as e:
            msg = str(e)
            if "Cuda error" in msg or "cudaStream" in msg:
                # CUDA context 已污染, 该进程后续全废. 记 marker 防重试死循环, 再上抛让进程退出.
                open(osp.join(output_dir_scene, ".cuda_failed"), "w").write(msg[:200])
                print(f"  [{scene_id}] ⚠️ CUDA 错误, 记 marker 并退出进程(重跑会跳过此场景): {msg[:80]}")
                raise
            # 非 CUDA(缺图/坏图): 跳过整个场景(保持 pairs 索引对齐), 记 marker
            open(osp.join(output_dir_scene, ".skipped"), "w").write(f"{type(e).__name__}: {msg[:200]}")
            print(f"  [{scene_id}] 跳过整个场景: {type(e).__name__}: {msg[:100]}")
            continue

        intrinsics_undistorted = np.stack(intrinsics_undistorted, axis=0)
        trajectories = np.stack(trajectories, axis=0)
        pairs = adjacency_list_to_array(adjacency_list)

        np.savez(
            osp.join(output_dir_scene, "scene_metadata.npz"),
            trajectories=trajectories,
            intrinsics=intrinsics_undistorted,
            images=selection_cam,
            pairs=pairs,
        )

    wall = time.time() - t_scene_start
    print(f"[rank{rank}] all scenes done | 后端={renderer.name} | "
          f"帧数={n_frames} | 总墙钟={wall:.1f}s")
    if n_frames > 0:
        print(f"[rank{rank}] 纯渲染累计={t_render:.1f}s | "
              f"渲染均速={t_render/n_frames*1000:.1f}ms/帧 | "
              f"端到端均速={wall/n_frames*1000:.1f}ms/帧(含 IO/去畸变/存图)")


if __name__ == "__main__":
    args = get_parser().parse_args()
    process_scenes(
        args.scannetpp_dir,
        args.precomputed_anymap_pairs,
        args.output_dir,
        args.target_resolution,
        args.rank,
        args.world_size,
        backend=args.backend,
        max_scenes=args.max_scenes,
        timing=args.timing,
    )
