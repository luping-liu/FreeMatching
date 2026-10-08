#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# render_depth_nvdiffrast.py
#
#   用 nvdiffrast 的 CUDA 光栅化后端渲染深度图, 替代 UFM preprocess_scannetpp.py
#   里的 pyrender(OpenGL/EGL). 目的: 在 compute-only 容器(无 libEGL_nvidia)上
#   也能用 GPU 渲深度 —— 原理同 blender Cycles-CUDA: 只吃 CUDA compute, 不吃 OpenGL.
#
#   语义严格对齐 pyrender(UFM 的金标准):
#     - 相机: opencv 约定 (x右 y下 z前/朝场景), 内参 fx,fy,cx,cy
#     - 位姿: 输入 cam_to_world (4x4, opencv), 内部取 inv 得 world2cam
#     - 深度: 沿相机光轴的正距离(view-space Z), 单位米; 背景=0
#     - 遮挡: 近的覆盖远的 (z-buffer 保留最小 view Z)
#     - 行序: row 0 = 图像顶部 (与 PIL/pyrender 一致) —— 由自检自动锁定 FLIP_V
#
#   用法:
#     python render_depth_nvdiffrast.py --self_test      # 合成场景自校准+自检(必跑)
#   作为库:
#     from render_depth_nvdiffrast import DepthRendererCuda
#     r = DepthRendererCuda()
#     r.set_mesh(verts_np, faces_np)                      # world 坐标
#     depth = r.render(cam_to_world, K, W, H)             # -> (H,W) float32 米, bg=0
# ============================================================================
import argparse
import numpy as np
import torch
import nvdiffrast.torch as dr

ZNEAR_DEFAULT = 0.05      # UFM/scannetpp render.yml 默认
ZFAR_DEFAULT = 20.0


class DepthRendererCuda:
    def __init__(self, device="cuda", flip_v=True):
        """
        flip_v: 是否对输出做垂直翻转以匹配"row 0 = 顶部"的图像约定.
                默认 True (经自检确认的正确值); calibrate() 会复核并按需覆盖.
        """
        self.device = torch.device(device)
        self.glctx = dr.RasterizeCudaContext(device=self.device)
        self.flip_v = flip_v
        self.verts = None     # (V,3) float32 cuda, world
        self.faces = None     # (F,3) int32 cuda

    def set_mesh(self, vertices, faces):
        self.verts = torch.as_tensor(np.ascontiguousarray(vertices),
                                     dtype=torch.float32, device=self.device)
        self.faces = torch.as_tensor(np.ascontiguousarray(faces),
                                     dtype=torch.int32, device=self.device)

    @torch.no_grad()
    def render(self, cam_to_world, K, W, H, znear=ZNEAR_DEFAULT, zfar=ZFAR_DEFAULT):
        assert self.verts is not None, "先 set_mesh()"
        dev = self.device
        c2w = torch.as_tensor(cam_to_world, dtype=torch.float32, device=dev)
        w2c = torch.inverse(c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # world -> opencv camera (x右 y下 z前)
        v_cam = self.verts @ R.T + t                      # (V,3)
        X, Y, Z = v_cam[:, 0], v_cam[:, 1], v_cam[:, 2]

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # opencv 像素 -> NDC, 再乘 w(=Z) 得 clip-space 齐次坐标 (nvdiffrast 期望 clip)
        #   u_pix = fx*X/Z + cx (列, 0..W) ; v_pix = fy*Y/Z + cy (行, 0..H, 顶部=0)
        #   x_ndc = 2*u_pix/W - 1 ; y_ndc = 2*v_pix/H - 1
        eps = 1e-8
        Zc = Z.clamp_min(eps)
        xc = (2.0 * fx * X + (2.0 * cx - W) * Z) / W
        yc = (2.0 * fy * Y + (2.0 * cy - H) * Z) / H
        # 深度测试用的 z_ndc: Z∈[znear,zfar]->[-1,1] 单调增 => 近的 z_ndc 小 => 近的覆盖远的
        zc = ((zfar + znear) / (zfar - znear)) * Z - (2.0 * zfar * znear / (zfar - znear))
        wc = Z
        pos_clip = torch.stack([xc, yc, zc, wc], dim=-1).unsqueeze(0).contiguous()  # (1,V,4)

        rast, _ = dr.rasterize(self.glctx, pos_clip, self.faces, resolution=[H, W])  # (1,H,W,4)

        # 深度 = 插值 view-space Z (正距离, 米)
        depth_attr = Zc.reshape(1, -1, 1).contiguous()            # (1,V,1)
        depth, _ = dr.interpolate(depth_attr, rast, self.faces)   # (1,H,W,1)
        depth = depth[0, ..., 0]
        mask = rast[0, ..., 3] > 0                                # triangle_id>0 = 命中
        depth = torch.where(mask, depth, torch.zeros_like(depth))

        if self.flip_v:
            depth = torch.flip(depth, dims=[0])
        return depth.float().cpu().numpy()

    # ----- 自校准: 用"上近下远"非对称合成场景自动决定 flip_v -----
    @torch.no_grad()
    def calibrate(self, verbose=True):
        # 相机看 +Z, fx=fy=200, 256x256, 主点居中
        W = H = 256
        K = np.array([[200., 0, 128.], [0, 200., 128.], [0, 0, 1.]])
        c2w = np.eye(4)
        # 上方四边形(opencv -Y=上方, 投到图像上半), 近 z=1.5
        # 下方四边形(+Y=下方, 投到图像下半), 远 z=2.5
        verts = np.array([
            # top quad (Y in [-0.4,-0.02]) z=1.5
            [-0.5, -0.40, 1.5], [0.5, -0.40, 1.5], [0.5, -0.02, 1.5], [-0.5, -0.02, 1.5],
            # bottom quad (Y in [0.02,0.4]) z=2.5
            [-0.5,  0.02, 2.5], [0.5,  0.02, 2.5], [0.5,  0.40, 2.5], [-0.5,  0.40, 2.5],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
        self.set_mesh(verts, faces)

        best = None
        for fv in (False, True):
            self.flip_v = fv
            d = self.render(c2w, K, W, H)
            top = d[:H // 2][d[:H // 2] > 0]
            bot = d[H // 2:][d[H // 2:] > 0]
            if top.size == 0 or bot.size == 0:
                continue
            top_med, bot_med = float(np.median(top)), float(np.median(bot))
            # 正确: 上半近(≈1.5), 下半远(≈2.5)
            ok = (abs(top_med - 1.5) < 0.1) and (abs(bot_med - 2.5) < 0.1)
            if verbose:
                print(f"  [calibrate] flip_v={fv}: 上半深度中位={top_med:.3f} 下半={bot_med:.3f} "
                      f"{'<= 正确' if ok else ''}")
            if ok:
                best = fv
        if best is None:
            raise RuntimeError("自校准失败: 两种 flip 都对不上 上近下远 期望, 渲染数学有问题")
        self.flip_v = best
        if verbose:
            print(f"  [calibrate] 锁定 flip_v = {best}")
        return best


def _approx(name, got, exp, tol, results):
    ok = abs(got - exp) <= tol
    results.append(ok)
    print(f"    {'✅' if ok else '❌'} {name}: 实测={got:.4f} 期望≈{exp} (tol {tol})")


def self_test():
    print("=" * 72)
    print("render_depth_nvdiffrast 自检: 合成解析场景, 验证深度值/方向/遮挡/mask")
    print("=" * 72)
    r = DepthRendererCuda()

    print("\n[1] 自校准 flip_v (上近下远 非对称场景)")
    r.calibrate()

    results = []
    W = H = 256
    K = np.array([[200., 0, 128.], [0, 200., 128.], [0, 0, 1.]])
    c2w = np.eye(4)

    print("\n[2] 深度值/单位: 正对 z=2.0m 的大平面, 期望全前景深度≈2.0")
    big = 2.0
    verts = np.array([[-big, -big, 2.0], [big, -big, 2.0], [big, big, 2.0], [-big, big, 2.0]],
                     dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    r.set_mesh(verts, faces)
    d = r.render(c2w, K, W, H)
    cov = float((d > 0).mean())
    _approx("中心深度(米)", float(d[H // 2, W // 2]), 2.0, 0.01, results)
    _approx("全图覆盖率", cov, 1.0, 0.02, results)

    print("\n[3] 遮挡 + 水平方向: 远平面z=3覆盖全图, 近平面z=1只盖左半(X<0)")
    print("     opencv x右 => X<0 投到图像左半 => 左半应=1, 右半应=3")
    far = np.array([[-3, -3, 3.0], [3, -3, 3.0], [3, 3, 3.0], [-3, 3, 3.0]], dtype=np.float32)
    near = np.array([[-3, -3, 1.0], [0, -3, 1.0], [0, 3, 1.0], [-3, 3, 1.0]], dtype=np.float32)
    verts = np.concatenate([far, near], 0)
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
    r.set_mesh(verts, faces)
    d = r.render(c2w, K, W, H)
    left_med = float(np.median(d[:, :W // 2 - 10][d[:, :W // 2 - 10] > 0]))
    right_med = float(np.median(d[:, W // 2 + 10:][d[:, W // 2 + 10:] > 0]))
    _approx("左半深度(近,遮挡赢)", left_med, 1.0, 0.05, results)
    _approx("右半深度(远)", right_med, 3.0, 0.05, results)

    print("\n[4] mask: 小三角只占中心, 四角应为背景(深度=0)")
    verts = np.array([[-0.2, -0.2, 2.0], [0.2, -0.2, 2.0], [0.0, 0.2, 2.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    r.set_mesh(verts, faces)
    d = r.render(c2w, K, W, H)
    corner = max(float(d[0, 0]), float(d[0, -1]), float(d[-1, 0]), float(d[-1, -1]))
    center_hit = float(d[H // 2, W // 2]) > 1.0
    _approx("四角背景=0", corner, 0.0, 1e-6, results)
    results.append(center_hit)
    print(f"    {'✅' if center_hit else '❌'} 中心命中(深度>1): {center_hit}")

    print("\n" + "=" * 72)
    if all(results):
        print(f"✅ 全部 {len(results)} 项通过. 渲染器数学正确, flip_v={r.flip_v}.")
        print("   下一步: osmesa 渲真实 scannet++ mesh 做 ref, 逐像素对齐确认 (verify 脚本).")
    else:
        print(f"❌ {sum(results)}/{len(results)} 通过. 把输出贴回, 据失败项修正.")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        ap.print_help()
