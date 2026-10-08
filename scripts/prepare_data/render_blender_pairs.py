# coding: utf-8
import bpy
import os
import random
import math
import sys
import traceback
from mathutils import Vector
from mathutils import geometry
from bpy_extras.object_utils import world_to_camera_view

# 尝试导入 tqdm 库，如果失败则提供一个虚拟版本以保证代码可运行
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterator, *args, **kwargs):
        print("警告: tqdm 库未安装，无法显示进度条。请为Blender的Python环境安装tqdm。")
        return iterator

# --- 配置项 (与您的版本完全一致) ---
LAYOUT_SPACING_FACTOR = 1.0
FRAME2_ROTATION_DELTA_DEGREES_MIN = 30.0
FRAME2_ROTATION_DELTA_DEGREES_MAX = 50.0
MAX_TILT_DEGREES = 30.0
ASPECT_RATIO_THRESHOLD = 3.0
CROP_TARGET_RATIO = 3.0
FRAME2_SCALE_VARIATION_MIN = 0.8
FRAME2_SCALE_VARIATION_MAX = 1.2
MIN_RELATIVE_SCALE = 0.8
MAX_RELATIVE_SCALE = 1.2
NORMALIZED_DIAGONAL = 2.0
PROJECTED_AREA_RATIO_THRESHOLD = 0.2

# --- 多GPU并行化所需的核心功能 ---
# 优化版: OPTIX优先 + 修复GPU选择bug + 返回所用backend(供降噪器选择)
ACTIVE_BACKEND = 'CUDA'  # 全局, setup_gpu 里更新
_REJECT_CACHE = set()    # #7: 跨样本缓存导入失败/不合格的 glb, 避免重复导入

def setup_gpu(device_index):
    """选择并激活指定物理GPU。修复#3: 按 type 过滤(OPTIX优先,回退CUDA),
    避免按混排索引误选到 CPU 条目导致静默回退 CPU 渲染。"""
    global ACTIVE_BACKEND
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        backend = None
        for cand in ('OPTIX', 'CUDA'):   # #2: OPTIX 优先(H20更快)
            try:
                prefs.compute_device_type = cand
                prefs.get_devices()
                if [d for d in prefs.devices if d.type == cand]:
                    backend = cand; break
            except Exception:
                continue
        if backend is None:
            print("错误: 无可用 OPTIX/CUDA 设备, 回退 CPU。")
            bpy.context.scene.cycles.device = 'CPU'; return False
        ACTIVE_BACKEND = backend
        gpus = [d for d in prefs.devices if d.type == backend]
        for d in prefs.devices: d.use = False     # 先全关(含CPU条目)
        phys = device_index % len(gpus)           # 防越界
        gpus[phys].use = True
        bpy.context.scene.cycles.device = 'GPU'
        print(f"成功激活 [{backend}] GPU {phys}: {gpus[phys].name} (共 {len(gpus)} 卡)")
        return True
    except Exception as e:
        print(f"设置GPU时发生错误: {e}")
        bpy.context.scene.cycles.device = 'CPU'; return False

# --- 核心功能函数 (与您的版本完全一致) ---

def get_all_glb_files(search_dir):
    if not os.path.isdir(search_dir):
        print(f"错误：指定的根目录 '{search_dir}' 不存在。")
        return None
    all_files = [os.path.join(dp, f) for dp, dn, fs in os.walk(search_dir) for f in fs if f.endswith('.glb')]
    if len(all_files) < 2:
        print(f"错误：在 '{search_dir}' 中需要至少两个 .glb 文件。")
        return None
    return all_files

def cleanup_scene():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
    for coll in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)

def get_hierarchy_bounds(obj):
    bpy.context.view_layer.update()
    min_c, max_c = Vector((float('inf'),)*3), Vector((float('-inf'),)*3)
    mesh_objs = [obj] if obj.type == 'MESH' else [c for c in obj.children_recursive if c.type == 'MESH']
    if not any(c.bound_box for c in mesh_objs):
        return (Vector((0,0,0)), obj.matrix_world.translation)
    
    depsgraph = bpy.context.evaluated_depsgraph_get()
    has_verts = False
    for child in mesh_objs:
        if not child.bound_box: continue
        eval_child = child.evaluated_get(depsgraph)
        world_matrix = eval_child.matrix_world
        for corner in [world_matrix @ Vector(c) for c in child.bound_box]:
            min_c = Vector(map(min, zip(min_c, corner)))
            max_c = Vector(map(max, zip(max_c, corner)))
            has_verts = True
    if not has_verts:
        return (Vector((0,0,0)), obj.matrix_world.translation)

    return (max_c - min_c, (min_c + max_c) / 2.0)

def normalize_shape_if_exaggerated(parent_obj):
    orig_loc, orig_rot, orig_scale = parent_obj.location.copy(), parent_obj.rotation_euler.copy(), parent_obj.scale.copy()
    parent_obj.location, parent_obj.rotation_euler, parent_obj.scale = (0,0,0), (0,0,0), (1,1,1)
    dims, _ = get_hierarchy_bounds(parent_obj)
    parent_obj.location, parent_obj.rotation_euler, parent_obj.scale = orig_loc, orig_rot, orig_scale
    if min(dims) < 1e-6: return
    longest, shortest = max(dims), min(dims)
    if longest / shortest > ASPECT_RATIO_THRESHOLD:
        long_idx = list(dims).index(longest)
        squash = (shortest * CROP_TARGET_RATIO) / longest
        parent_obj.scale[long_idx] *= squash

def standardize_and_scale_object(parent_obj):
    dims, center = get_hierarchy_bounds(parent_obj)
    parent_obj.location -= center
    bpy.context.view_layer.update()
    
    orig_rot = parent_obj.rotation_euler.copy()
    parent_obj.rotation_euler = (0,0,0)
    dims, _ = get_hierarchy_bounds(parent_obj)
    diag = dims.length
    parent_obj.rotation_euler = orig_rot
    
    if diag == 0: return
    parent_obj.scale *= (NORMALIZED_DIAGONAL / diag)
    parent_obj.scale *= random.uniform(MIN_RELATIVE_SCALE, MAX_RELATIVE_SCALE)

def get_projected_2d_data(scene, camera, obj):
    bpy.context.view_layer.update()
    mesh_objs = [c for c in obj.children_recursive if c.type == 'MESH']
    if obj.type == 'MESH': mesh_objs.append(obj)
    if not mesh_objs: return 0, 0
    all_2d_verts = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for mesh_obj in mesh_objs:
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        world_verts = [eval_obj.matrix_world @ v.co for v in mesh.vertices]
        for v in world_verts:
            co_2d = world_to_camera_view(scene, camera, v)
            if co_2d.z > 0:
                all_2d_verts.append(co_2d.to_2d())
        eval_obj.to_mesh_clear()
    if not all_2d_verts: return 0, 0
    render = scene.render
    w, h = render.resolution_x, render.resolution_y
    pixel_verts = [(v.x * w, v.y * h) for v in all_2d_verts if 0.0 <= v.x <= 1.0 and 0.0 <= v.y <= 1.0]
    if not pixel_verts: return 0, 0
    min_x, max_x = min(v[0] for v in pixel_verts), max(v[0] for v in pixel_verts)
    min_y, max_y = min(v[1] for v in pixel_verts), max(v[1] for v in pixel_verts)
    bbox_area = (max_x - min_x) * (max_y - min_y)
    try:
        hull_indices = geometry.convex_hull_2d([Vector(v) for v in pixel_verts])
    except RuntimeError:
        return 0, bbox_area
    if not hull_indices: return 0, bbox_area
    hull_verts = [pixel_verts[i] for i in hull_indices]
    hull_area = 0.0
    for i in range(len(hull_verts)):
        p1 = hull_verts[i]; p2 = hull_verts[(i + 1) % len(hull_verts)]
        hull_area += p1[0] * p2[1] - p2[0] * p1[1]
    return abs(hull_area) / 2.0, bbox_area

def process_and_filter_objects(all_glb_files, num_to_select):
    bpy.ops.object.camera_add(location=(0, -10, 0), rotation=(math.pi/2, 0, 0))
    test_cam = bpy.context.object; test_cam.name = "_TestCamera"
    approved_objects = []
    # #7: 跳过历史已知不合格/导入失败的 glb(跨样本缓存), 省重复导入开销
    available_files = [f for f in all_glb_files if f not in _REJECT_CACHE]
    random.shuffle(available_files)

    while len(approved_objects) < num_to_select and available_files:
        glb_file = available_files.pop()
        try:
            bpy.ops.import_scene.gltf(filepath=glb_file)
        except Exception as e:
            print(f"导入文件 {os.path.basename(glb_file)} 失败: {e}")
            _REJECT_CACHE.add(glb_file)   # 导入失败 -> 永久跳过
            continue

        parent = next((obj for obj in bpy.context.selected_objects if obj.parent is None), None)
        if not parent: continue

        normalize_shape_if_exaggerated(parent)
        standardize_and_scale_object(parent)
        parent.location = (0, 0, 0)
        parent.rotation_euler = (0, 0, 0)

        hull_area, bbox_area = get_projected_2d_data(bpy.context.scene, test_cam, parent)
        ratio = hull_area / bbox_area if bbox_area > 1e-4 else 0

        if ratio >= PROJECTED_AREA_RATIO_THRESHOLD:
            approved_objects.append(parent)
        else:
            _REJECT_CACHE.add(glb_file)   # 投影面积比不达标 -> 永久跳过
            bpy.ops.object.select_all(action='DESELECT')
            parent.select_set(True)
            for child in parent.children_recursive: child.select_set(True)
            bpy.ops.object.delete()

    bpy.data.objects.remove(test_cam, do_unlink=True)
    return approved_objects

def get_base_positions(num_objects):
    spacing = NORMALIZED_DIAGONAL * LAYOUT_SPACING_FACTOR
    base_positions = []
    if num_objects == 2:
        base_positions.extend([Vector((-spacing/2, 0, 0)), Vector((spacing/2, 0, 0))])
    elif num_objects == 3:
        h = (math.sqrt(3)/2)*spacing; y_t = (1/3)*h; y_b = -(2/3)*h; x_s = spacing/2
        base_positions.extend([Vector((0, y_t, 0)), Vector((-x_s, y_b, 0)), Vector((x_s, y_b, 0))])
    return base_positions

def calculate_transforms(parent_objs):
    transforms = { 'frame1': {}, 'frame2': {} }
    base_pos_f1 = get_base_positions(len(parent_objs))
    base_pos_f2 = base_pos_f1.copy(); random.shuffle(base_pos_f2)
    jitter_strength = NORMALIZED_DIAGONAL * 0.3
    max_tilt_rad = math.radians(MAX_TILT_DEGREES)
    delta_rad_min, delta_rad_max = math.radians(FRAME2_ROTATION_DELTA_DEGREES_MIN), math.radians(FRAME2_ROTATION_DELTA_DEGREES_MAX)
    for i, parent in enumerate(parent_objs):
        jitter_f1 = Vector((random.uniform(-jitter_strength, jitter_strength), random.uniform(-jitter_strength, jitter_strength), 0))
        loc_f1 = base_pos_f1[i] + jitter_f1
        rot_f1 = [random.uniform(-max_tilt_rad, max_tilt_rad) for _ in range(3)]
        scale_f1 = parent.scale.copy()
        transforms['frame1'][parent.name] = {'loc': loc_f1, 'rot': rot_f1, 'scale': scale_f1}
        jitter_f2 = Vector((random.uniform(-jitter_strength, jitter_strength), random.uniform(-jitter_strength, jitter_strength), 0))
        loc_f2 = base_pos_f2[i] + jitter_f2
        delta_rot = [random.uniform(delta_rad_min, delta_rad_max) * random.choice([-1,1]) for _ in range(3)]
        rot_f2 = [rot_f1[j] + delta_rot[j] for j in range(3)]
        scale_f2 = scale_f1 * random.uniform(FRAME2_SCALE_VARIATION_MIN, FRAME2_SCALE_VARIATION_MAX)
        transforms['frame2'][parent.name] = {'loc': loc_f2, 'rot': rot_f2, 'scale': scale_f2}
    return transforms

def apply_transforms_and_keyframe(parent_objs, transforms):
    for parent in parent_objs:
        if parent.name not in transforms['frame1']: continue
        for frame, frame_transforms in transforms.items():
            frame_num = int(frame.replace('frame', ''))
            bpy.context.scene.frame_set(frame_num)
            parent.location = frame_transforms[parent.name]['loc']
            parent.rotation_euler = frame_transforms[parent.name]['rot']
            parent.scale = frame_transforms[parent.name]['scale']
            parent.keyframe_insert(data_path="location", frame=frame_num)
            parent.keyframe_insert(data_path="rotation_euler", frame=frame_num)
            parent.keyframe_insert(data_path="scale", frame=frame_num)

def setup_animated_camera(objects, transforms):
    bpy.ops.object.camera_add(location=(0,-10,0))
    camera = bpy.context.object; bpy.context.scene.camera = camera
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0,0,0))
    fp = bpy.context.object; fp.name = "FocusPoint"
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target, constraint.track_axis, constraint.up_axis = fp, 'TRACK_NEGATIVE_Z', 'UP_Y'
    
    original_transforms = {obj.name: {'loc': obj.location.copy(), 'rot': obj.rotation_euler.copy(), 'scale': obj.scale.copy()} for obj in objects}
    
    for frame, frame_transforms in transforms.items():
        frame_num = int(frame.replace('frame', ''))
        
        for obj in objects:
            if obj.name not in frame_transforms: continue
            obj.location = frame_transforms[obj.name]['loc']
            obj.rotation_euler = frame_transforms[obj.name]['rot']
            obj.scale = frame_transforms[obj.name]['scale']
        
        all_world_corners = []
        for obj in objects:
            dims, center = get_hierarchy_bounds(obj)
            if max(dims) == 0: continue
            half_dims = dims / 2.0
            obj_corners = [Vector(v) for v in [(x,y,z) for x in [-1,1] for y in [-1,1] for z in [-1,1]]]
            world_corners = [obj.matrix_world @ (center - obj.location) + (half_dims * c) for c in obj_corners]
            all_world_corners.extend(world_corners)

        if not all_world_corners:
            center_point = Vector((0,0,0)); max_dist = 5
        else:
            min_c = Vector(map(min, zip(*all_world_corners)))
            max_c = Vector(map(max, zip(*all_world_corners)))
            center_point = (min_c + max_c) / 2.0
            max_dist = max((c - center_point).length for c in all_world_corners)

        fp_location = center_point
        cam_location = center_point + Vector((0, -10, 0))
        if max_dist > 0:
            dist = (max_dist / math.tan(camera.data.angle/2)) * 1.3
            cam_location = center_point + (cam_location - center_point).normalized() * max(dist, 1.0)
        
        bpy.context.scene.frame_set(frame_num)
        camera.location = cam_location
        fp.location = fp_location
        camera.keyframe_insert(data_path="location", frame=frame_num)
        fp.keyframe_insert(data_path="location", frame=frame_num)

    for obj in objects:
        if obj.name in original_transforms:
            obj.location = original_transforms[obj.name]['loc']
            obj.rotation_euler = original_transforms[obj.name]['rot']
            obj.scale = original_transforms[obj.name]['scale']

def setup_lighting():
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.object; sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), 0, math.radians(-45))

# --- (修改) 移除 flow_visualization 的Compositor设置 ---
def setup_compositor_for_flow(output_path, file_index):
    """配置合成器节点以输出PNG图像, 光流EXR文件和蒙版。"""
    scene = bpy.context.scene; scene.use_nodes = True; tree = scene.node_tree
    for node in tree.nodes: tree.nodes.remove(node)
    
    rl = tree.nodes.new(type='CompositorNodeRLayers')
    
    file_prefix = f"sample_{file_index:05d}_"

    # 1. PNG 图像输出
    webp_out = tree.nodes.new(type='CompositorNodeOutputFile')
    webp_out.base_path = os.path.join(output_path, "frames")
    webp_out.format.file_format = 'WEBP'
    webp_out.format.quality = 90 
    webp_out.file_slots[0].path = file_prefix
    tree.links.new(rl.outputs['Image'], webp_out.inputs[0])
    
    # 2. 光流 EXR 数据输出
    exr_out = tree.nodes.new(type='CompositorNodeOutputFile')
    exr_out.base_path = os.path.join(output_path, "flow_data")
    exr_out.format.file_format, exr_out.format.color_depth = 'OPEN_EXR', '32'
    exr_out.file_slots[0].path = file_prefix
    tree.links.new(rl.outputs['Vector'], exr_out.inputs[0])
    
    # 3. 蒙版 PNG 输出
    mask_out = tree.nodes.new(type='CompositorNodeOutputFile')
    mask_out.base_path = os.path.join(output_path, "masks")
    mask_out.format.file_format = 'PNG'
    mask_out.format.color_mode = 'BW'
    mask_out.file_slots[0].path = file_prefix
    tree.links.new(rl.outputs['Alpha'], mask_out.inputs[0])
    
    print("三分支后期合成节点已设置 (图像, 光流数据, 蒙版)。")

# --- 封装后的单次样本生成函数 ---
def generate_single_sample(output_path, file_index, all_glb_files):
    """执行一次完整的场景清理、对象导入、动画设置和渲染流程。"""
    cleanup_scene()
    
    num_to_select = random.randint(2, min(3, len(all_glb_files)))
    
    final_objects = process_and_filter_objects(all_glb_files, num_to_select)
    if not final_objects or len(final_objects) < 2: 
        print(f"错误：[样本 {file_index}] 未能找到足够数量 ({len(final_objects)}/{num_to_select}) 的合格模型。")
        return False

    transforms = calculate_transforms(final_objects)
    
    setup_animated_camera(final_objects, transforms)
    
    setup_lighting()
    
    apply_transforms_and_keyframe(final_objects, transforms)

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128                      # #1: 按要求保持 128
    scene.render.resolution_x, scene.render.resolution_y = 640, 360
    scene.render.film_transparent = True
    bpy.context.view_layer.use_pass_vector = True
    scene.frame_start, scene.frame_end = 1, 2

    # #5: OptiX 降噪(低噪点更干净, OPTIX backend 时启用)
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = 'OPTIX' if ACTIVE_BACKEND == 'OPTIX' else 'OPENIMAGEDENOISE'
    except Exception:
        pass
    # #6: persistent data —— 双帧渲染复用 BVH/几何, 省第2帧重建开销
    scene.render.use_persistent_data = True

    setup_compositor_for_flow(output_path, file_index)

    # #8: 唯一临时路径(原 random.randint 多进程会撞名); compositor 实际接管输出
    scene.render.filepath = f"/tmp/blender_render_{os.getpid()}_{file_index}_"

    bpy.ops.render.render(animation=True)
    return True

# --- 多GPU脚本主入口 ---
def main():
    """解析命令行参数并为当前GPU启动批量生成循环。"""
    argv = sys.argv[sys.argv.index("--") + 1:]
    if len(argv) < 5:
        print("用法: blender -b -P <script> -- <root_dir> <output_path> <num_samples> <gpu_index> <start_index>")
        sys.exit(1)
        
    root_dir, output_path, num_samples, gpu_index, start_index = argv[0], argv[1], int(argv[2]), int(argv[3]), int(argv[4])

    print("="*40)
    print(f"脚本启动，参数如下:")
    print(f"  - 模型根目录: {root_dir}")
    print(f"  - 输出目录:   {output_path}")
    print(f"  - 生成数量:   {num_samples}")
    print(f"  - GPU 索引:   {gpu_index}")
    print(f"  - 起始序号:   {start_index}")
    print("="*40)
    
    # --- (修改) 移除 flow_visualization 目录创建 ---
    for sub_dir in ["frames", "flow_data", "masks"]:
        os.makedirs(os.path.join(output_path, sub_dir), exist_ok=True)

    setup_gpu(gpu_index)
    
    print("正在扫描模型目录以建立文件列表...")
    all_glb_files = get_all_glb_files(root_dir)
    if not all_glb_files:
        sys.exit(1)
    print(f"扫描完成，共找到 {len(all_glb_files)} 个 .glb 模型。")

    if gpu_index == 0 and TQDM_AVAILABLE:
        print("\nGPU 0 已启动，将显示总体任务进度条。")
        iterator = tqdm(range(num_samples), desc=f"GPU {gpu_index} Task", unit="sample", dynamic_ncols=True)
    else:
        iterator = range(num_samples)

    for i in iterator:
        current_file_index = start_index + i
        
        # --- 新增功能：检查文件是否存在，如果存在则跳过 ---
        # 我们以第一帧的WEBP图像作为检查的代表
        expected_output_file = os.path.join(output_path, "frames", f"sample_{current_file_index:05d}_0001.webp")
        if os.path.exists(expected_output_file):
            message = f"样本 {current_file_index} 已存在，自动跳过。"
            if gpu_index == 0 and TQDM_AVAILABLE and hasattr(iterator, 'write'):
                # 在tqdm进度条下方打印信息
                iterator.write(message)
            else:
                print(message)
            continue # 直接进入下一次循环
        # --- 新增功能结束 ---

        try:
            success = generate_single_sample(output_path, current_file_index, all_glb_files)
            if not success:
                message = f"样本 {current_file_index} 生成失败，跳过。"
                if gpu_index == 0 and TQDM_AVAILABLE and hasattr(iterator, 'write'):
                    iterator.write(message)
                else:
                    print(message)
        except Exception:
            error_message = f"!!!!!! [样本 {current_file_index}] 生成过程中发生严重错误 !!!!!!\n"
            error_message += traceback.format_exc()
            error_message += "!!!!!! 跳过此样本，继续下一个。 !!!!!!\n"
            if gpu_index == 0 and TQDM_AVAILABLE and hasattr(iterator, 'write'):
                iterator.write(error_message)
            else:
                print(error_message)

    print(f"\n任务完成！由 GPU {gpu_index} 生成的 {num_samples} 个样本已处理完毕。")

if __name__ == "__main__":
    main()
