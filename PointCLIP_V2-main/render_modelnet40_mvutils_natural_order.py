import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import trimesh
from zeroshot_cls.trainers.mv_utils_zs import Realistic_Projection

def normalize_points(points):
    center = torch.mean(points, dim=1, keepdim=True)
    points = points - center
    scale = torch.max(torch.norm(points, dim=2, keepdim=True), dim=1, keepdim=True)[0]
    points = points / scale
    return points

def y_up_to_z_up(points):
    # points: [B, N, 3]
    # 交换y和z轴
    return points[..., [0, 2, 1]]

def load_off(file_path):
    mesh = trimesh.load_mesh(file_path)
    points = mesh.vertices
    return torch.tensor(points).float()

def load_npy(file_path):
    arr = np.load(file_path)
    if arr.ndim == 2:
        return torch.tensor(arr).float().unsqueeze(0)
    elif arr.ndim == 3:
        return torch.tensor(arr).float()
    else:
        raise ValueError('npy格式不支持')

def process_category(category_path, output_path, device, partition, file_ext='.off'):
    category_name = os.path.basename(category_path)
    output_category_path = os.path.join(output_path, category_name, partition)
    os.makedirs(output_category_path, exist_ok=True)
    files = [f for f in os.listdir(os.path.join(category_path, partition)) if f.endswith(file_ext)]
    # 初始化渲染器
    proj = Realistic_Projection()
    num_views = proj.num_views
    for file in tqdm(files, desc=f"Processing {category_name}-{partition}"):
        if file_ext == '.off':
            points = load_off(os.path.join(category_path, partition, file)).unsqueeze(0)
        else:
            points = load_npy(os.path.join(category_path, partition, file))
        points = normalize_points(points).to(device)
        points = y_up_to_z_up(points)
        # 渲染所有视角
        with torch.no_grad():
            imgs = proj.get_img(points)
        # imgs: [num_views, 3, H, W]，取每个视角的第一个通道
        imgs = imgs.cpu().numpy()  # [num_views, 3, H, W]
        base_name = os.path.splitext(file)[0]
        for view_idx in range(num_views):
            save_name = f"{base_name}_view{view_idx+1:02d}"
            save_path = os.path.join(output_category_path, f"{save_name}.npy")
            np.save(save_path, imgs[view_idx, 0])

def main():
    parser = argparse.ArgumentParser(description='Render ModelNet40 using mv_utils_zs.py Realistic_Projection (natural order, z-up)')
    parser.add_argument('--input_path', type=str, required=True, help='Path to ModelNet40 dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save rendered depth maps')
    parser.add_argument('--file_ext', type=str, default='.off', help='File extension: .off or .npy')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = [d for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))]
    for category in categories:
        print("==============================")
        print("当前处理类别：", category)
        category_path = os.path.join(args.input_path, category)
        process_category(category_path, args.output_path, device, 'train', file_ext=args.file_ext)
        process_category(category_path, args.output_path, device, 'test', file_ext=args.file_ext)

if __name__ == '__main__':
    main() 