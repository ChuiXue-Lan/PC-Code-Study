#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/16  19:55
# @Author  : 菠萝吹雪
# @Software: PyCharm
# @Describe:
# -*- encoding:utf-8 -*-
import os
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from GeoZe.lib_vis import get_colored_point_cloud_from_soft_labels, creat_labeled_point_cloud

'''
使用方法：
    运行脚本
    选择要查看的类别
    选择要查看的样本
    选择可视化方法：
        方法1：实时3D交互式可视化
        方法2：基于特征的软标签可视化（如果有特征数据）
        方法3：基于硬标签的可视化

    每种可视化方法都有其优势：
        方法1适合实时交互和查看细节
        方法2适合查看模型的预测概率分布
        方法3适合生成静态的可视化结果
'''


class PointCloudVisualizer:
    def __init__(self):
        # 设置基础路径
        self.base_path = r'D:\\Pycharm\\Remake\\GeoZe-master\\partseg\\'
        self.categories = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar',
                           'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket',
                           'skateboard', 'table']

        # 类别到部件的映射
        self.cat2part = {
            'airplane': ['body', 'wing', 'tail', 'engine'],
            'bag': ['handle', 'body'],
            'cap': ['panels', 'peak'],
            'car': ['roof', 'hood', 'wheel', 'body'],
            'chair': ['back', 'seat', 'leg', 'armrest']
            # 可以根据需要添加更多类别
        }

    def load_data(self, category, mode='test'):
        """加载特定类别的点云数据"""
        category_path = os.path.join(self.base_path, category)

        try:
            # 加载点云、标签和特征
            pc_path = os.path.join(category_path, f"{mode}_pc.pt")
            label_path = os.path.join(category_path, f"{mode}_labels.pt")
            feat_path = os.path.join(category_path, f"{mode}_features.pt")  # 添加特征文件路径

            if not os.path.exists(pc_path) or not os.path.exists(label_path):
                print(f"找不到数据文件: {category_path}")
                return None, None, None

            # 加载数据并移到CPU
            pc = torch.load(pc_path)
            labels = torch.load(label_path)
            feat = torch.load(feat_path) if os.path.exists(feat_path) else None
            if os.path.exists(feat_path):
                print(f"特征文件加载成功: {feat_path}")
            else:
                print(f"特征文件不存在: {feat_path}")

            # 确保数据在CPU上
            pc = pc.cpu()
            labels = labels.cpu()
            if feat is not None:
                feat = feat.cpu()

            # 确保数据格式正确
            if len(pc.shape) > 2:
                pc = pc.squeeze()
            if len(labels.shape) > 1:
                labels = labels.squeeze()
            if feat is not None and len(feat.shape) > 2:
                feat = feat.squeeze()

            # 转换为numpy数组
            pc_np = pc.numpy()
            labels_np = labels.numpy()
            feat_np = feat.numpy() if feat is not None else None

            print(f"点云形状: {pc_np.shape}")
            print(f"标签形状: {labels_np.shape}")
            if feat_np is not None:
                print(f"特征形状: {feat_np.shape}")

            return pc_np, labels_np, feat_np

        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def visualize_category(self, category, mode='test'):
        """可视化特定类别的点云"""
        print(f"\n正在处理类别: {category}")

        # 加载数据
        points, labels, features = self.load_data(category, mode)
        if points is None or labels is None:
            return

        # 如果是批量数据，让用户选择一个样本
        if len(points.shape) == 3:
            num_samples = points.shape[0]
            print(f"\n检测到{num_samples}个样本")
            while True:
                try:
                    sample_idx = int(input(f"请选择要查看的样本 (0-{num_samples - 1}): "))
                    if 0 <= sample_idx < num_samples:
                        points = points[sample_idx]
                        labels = labels[sample_idx]
                        if features is not None:
                            features = features[sample_idx]
                        break
                    else:
                        print("无效的样本索引，请重试")
                except ValueError:
                    print("请输入有效的数字")

        print(f"当前点云形状: {points.shape}")
        print(f"当前标签形状: {labels.shape}")

        # 选择可视化方法
        print("\n请选择可视化方法:")
        print("1. 使用Open3D直接可视化")
        print("2. 使用lib_vis中的get_colored_point_cloud_from_soft_labels")
        print("3. 使用lib_vis中的creat_labeled_point_cloud")

        while True:
            try:
                vis_choice = int(input("请选择 (1-3): "))
                if 1 <= vis_choice <= 3:
                    break
                else:
                    print("无效的选择，请重试")
            except ValueError:
                print("请输入有效的数字")

        if vis_choice == 1:
            self._visualize_with_open3d(points, labels, category)
        elif vis_choice == 2:
            if features is not None:
                try:
                    # 确保数据类型和维度正确
                    points = points.astype(np.float32)
                    features = features.astype(np.float32)
                    
                    print(f"原始特征形状: {features.shape}")
                    print(f"原始点云形状: {points.shape}")
                    
                    # 检查并处理点云维度
                    if len(points.shape) == 3:
                        points = points.reshape(-1, points.shape[-1])
                    
                    # 检查并处理特征维度
                    if len(features.shape) == 4:  # 形状为 (B, C, H, W)
                        B, C, H, W = features.shape
                        # 将特征转换为每个点的特征向量
                        features = features.transpose(0, 2, 3, 1)  # 变为 (B, H, W, C)
                        features = features.reshape(-1, C)  # 变为 (B*H*W, C)
                        
                        # 如果点云数量与特征数量不匹配，进行插值
                        if features.shape[0] != points.shape[0]:
                            # 计算每个点最近的特征
                            num_points = points.shape[0]
                            indices = np.linspace(0, features.shape[0]-1, num_points).astype(int)
                            features = features[indices]
                    elif len(features.shape) == 3:
                        features = features.reshape(-1, features.shape[-1])
                    
                    print(f"处理后特征形状: {features.shape}")
                    print(f"处理后点云形状: {points.shape}")
                    
                    # 确保特征和点云的数量匹配
                    if features.shape[0] != points.shape[0]:
                        raise ValueError(f"无法匹配点云({points.shape[0]})和特征({features.shape[0]})的数量")
                    
                    # 归一化特征到[0,1]区间
                    features = features - np.min(features, axis=-1, keepdims=True)
                    features = features / (np.max(features, axis=-1, keepdims=True) + 1e-6)
                    
                    # 计算软标签
                    soft_labels = np.exp(features) / (np.sum(np.exp(features), axis=-1, keepdims=True) + 1e-6)
                    
                    print(f"软标签形状: {soft_labels.shape}")
                    print(f"最终点云形状: {points.shape}")
                    
                    # 创建保存目录
                    save_path = "visualization_results"
                    os.makedirs(save_path, exist_ok=True)
                    
                    # 创建点云对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    
                    # 使用软标签生成颜色
                    cmap = plt.cm.get_cmap('tab20')
                    max_label_indices = np.argmax(soft_labels, axis=1)
                    colors = np.array([cmap(i / soft_labels.shape[1])[:3] for i in max_label_indices])
                    
                    # 设置点云颜色
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # 保存结果
                    save_file = os.path.join(save_path, f"{category}_soft.ply")
                    o3d.io.write_point_cloud(save_file, pcd)
                    
                    # 可视化
                    vis = o3d.visualization.Visualizer()
                    vis.create_window(window_name=f"Soft Labels: {category}")
                    vis.add_geometry(pcd)
                    
                    # 设置渲染选项
                    opt = vis.get_render_option()
                    opt.point_size = 2.0
                    opt.background_color = np.array([0, 0, 0])
                    
                    # 运行可视化
                    vis.run()
                    vis.destroy_window()
                    
                    print(f"点云已保存到: {save_file}")
                    
                except Exception as e:
                    print(f"可视化过程出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print("未找到特征数据，无法使用soft labels可视化")
        else:
            creat_labeled_point_cloud(points, labels, f"{category}_{mode}_labeled")

    def _visualize_with_open3d(self, points, labels, category):
        """使用Open3D进行可视化"""
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 获取唯一的标签
        unique_labels = np.unique(labels)

        # 创建颜色映射
        colors = plt.cm.get_cmap('tab20')(labels / max(labels.max(), 1))[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # 打印部件信息
        print(f"\n{category}的部件信息:")
        if category in self.cat2part:
            parts = self.cat2part[category]
            for i, part_name in enumerate(parts):
                if i in unique_labels:
                    count = np.sum(labels == i)
                    percentage = count / len(labels) * 100
                    print(f"部件 '{part_name}' (标签 {i}) 包含 {count} 个点 ({percentage:.2f}%)")

        # 显示点云
        print("\n显示完整点云...")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0, 0, 0])

        # 运行可视化
        vis.run()
        vis.destroy_window()

        # 分别显示每个部件
        if category in self.cat2part:
            for i, part_name in enumerate(self.cat2part[category]):
                if i in unique_labels:
                    self.view_single_part(points, labels, i, part_name)

    def view_single_part(self, points, labels, part_id, part_name):
        """显示单个部件"""
        # 创建该部件的点云
        part_indices = np.where(labels == part_id)[0]
        part_pcd = o3d.geometry.PointCloud()
        part_pcd.points = o3d.utility.Vector3dVector(points[part_indices])

        # 设置颜色
        color = plt.cm.get_cmap('tab20')(part_id / labels.max())[:3]
        part_colors = np.tile(color, (len(part_indices), 1))
        part_pcd.colors = o3d.utility.Vector3dVector(part_colors)

        print(f"\n显示部件: {part_name}")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Part: {part_name}")
        vis.add_geometry(part_pcd)

        # 设置渲染选项
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0, 0, 0])

        # 运行可视化
        vis.run()
        vis.destroy_window()


def main():
    visualizer = PointCloudVisualizer()

    # 打印可用类别
    print("可用的类别:")
    for i, category in enumerate(visualizer.categories):
        print(f"{i + 1}. {category}")

    # 让用户选择类别
    while True:
        try:
            choice = int(input("\n请选择要查看的类别 (输入序号): ")) - 1
            if 0 <= choice < len(visualizer.categories):
                break
            else:
                print("无效的选择，请重试")
        except ValueError:
            print("请输入有效的数字")

    # 可视化选择的类别
    category = visualizer.categories[choice]
    visualizer.visualize_category(category)


if __name__ == "__main__":
    main()