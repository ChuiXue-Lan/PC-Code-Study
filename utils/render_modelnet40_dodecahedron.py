import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from render.render import Renderer, check_valid_rotation_matrix
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import look_at_view_transform
from pathlib import Path
import trimesh

def normalize_points(points):
    """归一化点云到单位球内"""
    center = torch.mean(points, dim=1, keepdim=True)
    points = points - center
    scale = torch.max(torch.norm(points, dim=2, keepdim=True), dim=1, keepdim=True)[0]
    points = points / scale
    return points

def load_off(file_path):
    """加载OFF文件并返回点云数据"""
    mesh = trimesh.load_mesh(file_path)
    points = mesh.vertices
    return torch.tensor(points).float()

def get_dodecahedron_views():
    """完全复现MATLAB render_views.m的视角生成逻辑"""
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

def process_category(category_path, output_path, renderer, device, partition):
    """处理单个类别的所有模型"""
    # 确保输出目录存在
    category_name = os.path.basename(category_path)
    output_category_path = os.path.join(output_path, category_name, partition)
    os.makedirs(output_category_path, exist_ok=True)

    # 获取所有OFF文件
    off_files = [f for f in os.listdir(os.path.join(category_path, partition)) if f.endswith('.off')]

    # 获取十二面体的视角（方位角和仰角）
    azimuths, elevations = get_dodecahedron_views()

    for off_file in tqdm(off_files, desc=f"Processing {category_name}-{partition}"):
        # 加载OFF文件
        points = load_off(os.path.join(category_path, partition, off_file))
        points = points.unsqueeze(0)  # 添加batch维度
        
        # 归一化点云
        points = normalize_points(points)
        points = points.to(device)

        azimuths, elevations = get_dodecahedron_views()
        # 测试
        # for i in range(len(azimuths)):
        #     print(f"View {i+1}: Azimuth={azimuths[i]:.2f}°, Elevation={elevations[i]:.2f}°")
        for view_idx in range(len(azimuths)):
            base_name = os.path.splitext(off_file)[0]
            save_name = f"{base_name}_view{view_idx+1:02d}"
            save_path = os.path.join(output_category_path, f"{save_name}.npy")
            if os.path.exists(save_path):
                print(f"跳过已存在文件: {save_name}.npy")
                continue
            azim = torch.tensor([[float(azimuths[view_idx])]], device=device)
            elev = torch.tensor([[float(elevations[view_idx])]], device=device)
            dist = torch.tensor([[2.0]], device=device)
            with torch.no_grad():
                rendered_images = renderer.render_points(points, azim, elev, dist, view=1, aug=False, rot=False)
                depth_map = rendered_images[0, 0, 0]
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
                    # 保存
                    np.save(save_path, depth_map.cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description='Render depth maps for ModelNet40 dataset')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to ModelNet40 dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save rendered depth maps')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Size of rendered images')
    parser.add_argument('--points_radius', type=float, default=0.02,
                        help='Radius of points for rendering')
    parser.add_argument('--points_per_pixel', type=int, default=1,
                        help='Number of points per pixel')
    
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化渲染器
    renderer = Renderer(
        image_size=args.image_size,
        points_radius=args.points_radius,
        points_per_pixel=args.points_per_pixel
    ).to(device)

    # 获取所有类别
    categories = [d for d in os.listdir(args.input_path) 
                 if os.path.isdir(os.path.join(args.input_path, d))]
    sum_num = len(categories)
    num = 0

    # 处理每个类别
    for category in categories:
        print("==========================================================================")
        print("当前处理类别：", category)
        print("剩余 %d 个类别"%(sum_num-num))
        category_path = os.path.join(args.input_path, category)
        # 处理训练集
        process_category(category_path, args.output_path, renderer, device, 'train')
        # 处理测试集
        process_category(category_path, args.output_path, renderer, device, 'test')
        num += 1

if __name__ == '__main__':
    main() 