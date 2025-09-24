import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from ..render.render import Renderer, check_valid_rotation_matrix
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import axis_angle_to_matrix
from pathlib import Path
import trimesh

MAX_ERROR_COUNT = 5

def normalize_points(points):
    """归一化点云到单位球内"""
    center = torch.mean(points, dim=1, keepdim=True)
    points = points - center
    scale = torch.max(torch.norm(points, dim=2, keepdim=True), dim=1, keepdim=True)[0]
    points = points / scale
    return points

def load_obj_file(file_path):
    """加载OBJ文件并返回点云数据"""
    try:
        mesh = trimesh.load_mesh(file_path)
        points = mesh.vertices
        if points.shape[1] != 3:
            raise ValueError(f"点云不是3D点: {file_path} 的 shape={points.shape}")
        return torch.tensor(points).float()
    except Exception as e:
        print(f"加载OBJ文件失败: {file_path}, 错误: {e}")
        return None

def load_off_file(file_path):
    """加载OFF文件并返回点云数据"""
    try:
        mesh = trimesh.load_mesh(file_path)
        points = mesh.vertices
        if points.shape[1] != 3:
            raise ValueError(f"点云不是3D点: {file_path} 的 shape={points.shape}")
        return torch.tensor(points).float()
    except Exception as e:
        print(f"加载OFF文件失败: {file_path}, 错误: {e}")
        return None

def get_dodecahedron_views():
    """获取十二面体的20个视角"""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi],
        [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
    ])
    azimuths = []
    elevations = []
    for v in vertices:
        x, y, z = v
        # azimuth
        if x == 0:
            az = 90.0 if y > 0 else -90.0
        else:
            az = np.arctan(y / x) * 180 / np.pi
            if x < 0:
                az += 180
        # elevation
        el = np.arctan(z / np.sqrt(x**2 + y**2)) * 180 / np.pi
        azimuths.append(az)
        elevations.append(el)
    return azimuths, elevations

def process_model(model_path, output_path, renderer, device, category_id, model_id):
    """处理单个模型"""
    # 确保输出目录存在
    output_model_path = os.path.join(output_path, f"{category_id}-{model_id}")
    os.makedirs(output_model_path, exist_ok=True)

    # 检查模型文件
    obj_file = os.path.join(model_path, "model.obj")
    off_file = os.path.join(model_path, "model.off")
    
    points = None
    if os.path.exists(obj_file):
        points = load_obj_file(obj_file)
    elif os.path.exists(off_file):
        points = load_off_file(off_file)
    else:
        print(f"未找到模型文件: {model_path}")
        return False
    
    if points is None:
        print(f"加载模型失败: {model_path}")
        return False
    
    if len(points.shape) != 2 or points.shape[1] != 3:
        print(f"点数据维度错误: {model_path}, shape={points.shape}")
        return False
    
    points = points.unsqueeze(0)  # 添加batch维度
    
    # 归一化点云
    points = normalize_points(points)
    points = points.to(device)

    # 视角对齐 MATLAB
    rota = axis_angle_to_matrix(torch.tensor([-0.5 * np.pi, 0.0, 0.0])).to(points.device)
    points = points @ rota.T
    
    # 获取十二面体的视角
    azimuths, elevations = get_dodecahedron_views()
    
    success_count = 0
    for view_idx in range(len(azimuths)):
        save_name = f"{category_id}-{model_id}_view{view_idx+1:02d}.png"
        save_path = os.path.join(output_model_path, save_name)
        
        if os.path.exists(save_path):
            print(f"跳过已存在文件: {save_name}")
            success_count += 1
            continue
        
        azim = torch.tensor([[float(azimuths[view_idx])]], device=device)
        elev = torch.tensor([[float(elevations[view_idx])]], device=device)
        dist = torch.tensor([[2.0]], device=device)
        
        try:
            with torch.no_grad():
                # 渲染深度图
                rendered_images = renderer.render_points(points, azim, elev, dist, view=1, aug=False, rot=False)
                depth_map = rendered_images[0, 0, 0]
                
                # 处理深度图
                bg_value = depth_map.max()
                valid_mask = depth_map < bg_value - 1e-4
                
                if valid_mask.any():
                    valid_depths = depth_map[valid_mask]
                    depth_min = valid_depths.min()
                    depth_max = valid_depths.max()
                    
                    if (depth_max - depth_min) < 1e-5:
                        print(f"警告：视角 {view_idx+1} 的深度范围太小，跳过...")
                        continue
                    
                    # 归一化
                    depth_map = (depth_max - depth_map) / (depth_max - depth_min + 1e-6)
                    depth_map[~valid_mask] = 0.0
                    
                    # 保存为PNG格式
                    depth_np = depth_map.cpu().numpy()
                    # 将深度值转换为0-255的灰度值
                    depth_uint8 = (depth_np * 255).astype(np.uint8)
                    
                    # 使用PIL保存为PNG
                    from PIL import Image
                    img = Image.fromarray(depth_uint8)
                    img.save(save_path)
                    success_count += 1
                else:
                    print(f"警告：视角 {view_idx+1} 无有效深度，跳过...")
                    
        except Exception as e:
            print(f"渲染视角 {view_idx+1} 失败: {e}")
            continue
    
    return success_count == len(azimuths)  # 返回是否所有视角都成功渲染

def process_category(category_path, output_path, renderer, device, category_id):
    """处理单个类别的所有模型"""
    print(f"处理类别: {category_id}")
    
    # 获取类别下的所有模型目录
    category_dir = os.path.join(category_path, category_id)
    if not os.path.exists(category_dir):
        print(f"类别目录不存在: {category_dir}")
        return
    
    model_dirs = [d for d in os.listdir(category_dir) 
                 if os.path.isdir(os.path.join(category_dir, d))]
    
    print(f"找到 {len(model_dirs)} 个模型")
    
    success_count = 0
    for model_id in tqdm(model_dirs, desc=f"Processing {category_id}"):
        model_path = os.path.join(category_dir, model_id)
        if process_model(model_path, output_path, renderer, device, category_id, model_id):
            success_count += 1
    
    print(f"类别 {category_id} 完成，成功处理 {success_count}/{len(model_dirs)} 个模型")

def main():
    parser = argparse.ArgumentParser(description='Render depth maps for ShapeNetCore dataset')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to ShapeNetCore dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save rendered depth maps')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of rendered images')
    parser.add_argument('--points_radius', type=float, default=0.02,
                        help='Radius of points for rendering')
    parser.add_argument('--points_per_pixel', type=int, default=1,
                        help='Number of points per pixel')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Specific categories to process (optional)')
    
    args = parser.parse_args()

    # 设置设备
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    print(f"使用GPU {args.gpu}")
    
    # 初始化渲染器
    renderer = Renderer(
        image_size=args.image_size,
        points_radius=args.points_radius,
        points_per_pixel=args.points_per_pixel
    ).to(device)

    # 获取所有类别
    if args.categories:
        categories = args.categories
    else:
        categories = [d for d in os.listdir(args.input_path) 
                     if os.path.isdir(os.path.join(args.input_path, d))]
    
    print(f"总共需要处理 {len(categories)} 个类别")
    
    # 处理每个类别
    for i, category_id in enumerate(categories):
        print("=" * 70)
        print(f"当前处理类别：{category_id} ({i+1}/{len(categories)})")
        print("=" * 70)
        
        process_category(args.input_path, args.output_path, renderer, device, category_id)
        print(f"类别 {category_id} 处理完成")

if __name__ == '__main__':
    main()
