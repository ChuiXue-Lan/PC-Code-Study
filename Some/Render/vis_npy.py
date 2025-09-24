import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def analyze_depth_map(depth_map):
    """分析深度图的统计信息"""
    print("\n深度图分析:")
    print(f"形状: {depth_map.shape}")
    print(f"数据类型: {depth_map.dtype}")
    print(f"是否包含NaN: {np.isnan(depth_map).any()}")
    print(f"是否包含Inf: {np.isinf(depth_map).any()}")
    print(f"唯一值数量: {len(np.unique(depth_map))}")
    print(f"最小值: {depth_map.min():.4f}")
    print(f"最大值: {depth_map.max():.4f}")
    print(f"均值: {depth_map.mean():.4f}")
    print(f"标准差: {depth_map.std():.4f}")
    print(f"零值占比: {(depth_map == 0).sum() / depth_map.size * 100:.2f}%")
    
    # 计算非零值的统计信息
    non_zero_mask = depth_map != 0
    if non_zero_mask.any():
        non_zero_values = depth_map[non_zero_mask]
        print("\n非零值统计:")
        print(f"非零值数量: {len(non_zero_values)}")
        print(f"非零值最小值: {non_zero_values.min():.4f}")
        print(f"非零值最大值: {non_zero_values.max():.4f}")
        print(f"非零值均值: {non_zero_values.mean():.4f}")
        print(f"非零值标准差: {non_zero_values.std():.4f}")

def visualize_depth(depth_map, save_path):
    """将深度图可视化并保存为PNG图像
    Args:
        depth_map: 深度图数据，形状为(C, H, W)或(H, W)
        save_path: 保存路径，应以.png结尾
    """
    # 如果是3通道，取第一个通道作为深度图
    if len(depth_map.shape) == 3 and depth_map.shape[0] == 3:
        depth_map = depth_map[0]  # 使用第一个通道
    
    # 分析深度图
    # analyze_depth_map(depth_map)
    
    # 处理无效值
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 确保有效的深度值
    if depth_map.std() < 1e-6:  # 如果标准差接近0
        print("警告：深度图几乎没有变化！")
        return False
    
    # 移除异常值（使用百分位数）
    valid_mask = depth_map != 0  # 只考虑非零值
    if valid_mask.any():
        valid_depths = depth_map[valid_mask]
        p_min, p_max = np.percentile(valid_depths, [2, 98])
        depth_map = np.clip(depth_map, p_min, p_max)
    
    # 归一化到[0,1]范围
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max > depth_min:
        normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        print("警告：深度图最大值等于最小值！")
        return False
    
    # 反转颜色（使近处物体显示为白色，远处物体显示为黑色）
    normalized_depth = 1 - normalized_depth
    
    # 创建图像
    plt.figure(figsize=(8, 8))
    plt.imshow(normalized_depth, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    
    return True

def process_folder(input_path, output_path):
    """处理 input_path 目录下所有 .npy 深度图文件，并将可视化结果保存到 output_path"""
    os.makedirs(output_path, exist_ok=True)
    npy_files = [f for f in os.listdir(input_path) if f.endswith('.npy')]
    processed_count = 0
    skipped_count = 0
    for npy_file in tqdm(npy_files, desc=f"Visualizing npy files"):
        npy_path = os.path.join(input_path, npy_file)
        png_path = os.path.join(output_path, npy_file.replace('.npy', '.png'))
        # 如果 PNG 已存在可选择跳过
        # if os.path.exists(png_path):
        #     skipped_count += 1
        #     continue
        depth_map = np.load(npy_path)
        if not visualize_depth(depth_map, png_path):
            print(f"警告：文件 {npy_file} 可视化失败！")
        else:
            processed_count += 1
    print(f"处理完成:")
    print(f"- 新处理文件数: {processed_count}")
    print(f"- 跳过已存在文件数: {skipped_count}")
    print(f"- 总文件数: {len(npy_files)}")

def main():
    parser = argparse.ArgumentParser(description='Visualize rendered depth maps')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input .npy files')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save visualized depth maps')
    args = parser.parse_args()
    process_folder(args.input_path, args.output_path)

if __name__ == '__main__':
    main() 