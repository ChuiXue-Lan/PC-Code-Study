import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# 配置路径
RESULTS_PATH = '/home/cls2024/ltx/Remake/PTPM-main/default/final_result/data/results_nusc.json'
SAVE_DIR = '/home/cls2024/ltx/Remake/PTPM-main/default/final_result/data/visualizations'

def load_results(results_path):
    """加载检测结果"""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data['results']

def convert_detection_to_boxes(detection):
    """将检测结果转换为边界框格式"""
    boxes = []
    scores = []
    labels = []
    
    for det in detection:
        # 提取边界框信息
        boxes.append([
            det['translation'][0],  # x
            det['translation'][1],  # y
            det['translation'][2],  # z
            det['size'][0],         # length
            det['size'][1],         # width
            det['size'][2],         # height
            det['rotation'][0]      # yaw (使用四元数的第一个分量作为yaw)
        ])
        scores.append(det['detection_score'])
        # 将检测名称转换为标签索引
        if det['detection_name'] == 'car':
            labels.append(1)  # 假设1是car的标签索引
        elif det['detection_name'] == 'pedestrian':
            labels.append(2)  # 假设2是pedestrian的标签索引
        elif det['detection_name'] == 'bicycle':
            labels.append(3)  # 假设3是bicycle的标签索引
        else:
            labels.append(0)  # 其他类别
    
    return np.array(boxes), np.array(scores), np.array(labels)

def draw_boxes(ax, boxes, scores, labels):
    """绘制3D边界框"""
    for box, score, label in zip(boxes, scores, labels):
        # 提取边界框参数
        x, y, z = box[0], box[1], box[2]
        l, w, h = box[3], box[4], box[5]
        yaw = box[6]
        
        # 计算边界框的8个顶点
        corners = np.array([
            [x - l/2, y - w/2, z - h/2],
            [x + l/2, y - w/2, z - h/2],
            [x + l/2, y + w/2, z - h/2],
            [x - l/2, y + w/2, z - h/2],
            [x - l/2, y - w/2, z + h/2],
            [x + l/2, y - w/2, z + h/2],
            [x + l/2, y + w/2, z + h/2],
            [x - l/2, y + w/2, z + h/2]
        ])
        
        # 应用旋转
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        corners = np.dot(corners - [x, y, z], rotation_matrix.T) + [x, y, z]
        
        # 绘制边界框
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 连接线
        ]
        
        for edge in edges:
            ax.plot3D(
                [corners[edge[0], 0], corners[edge[1], 0]],
                [corners[edge[0], 1], corners[edge[1], 1]],
                [corners[edge[0], 2], corners[edge[1], 2]],
                'b-', linewidth=1
            )
        
        # 添加标签和分数
        ax.text(x, y, z + h/2, f'{label}:{score:.2f}', color='red')

def main():
    print('-----------------开始可视化检测结果-------------------------')
    
    # 加载结果
    results_dict = load_results(RESULTS_PATH)
    total_frames = len(results_dict)
    print(f'总帧数: \t{total_frames}')
    
    # 创建保存目录
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用tqdm显示进度
    progress_bar = tqdm(results_dict.items(), total=total_frames, desc='可视化进度')
    success_count = 0
    
    # 可视化每个场景
    for frame_id, frame_detections in progress_bar:
        try:
            # 更新进度条描述
            progress_bar.set_description(f'处理帧: {frame_id}')
            
            # 获取检测结果
            boxes, scores, labels = convert_detection_to_boxes(frame_detections)
            
            # 创建3D图形
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制边界框
            draw_boxes(ax, boxes, scores, labels)
            
            # 设置坐标轴标签
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # 设置视角
            ax.view_init(elev=20, azim=45)
            
            # 保存图像
            save_path = save_dir / f"{frame_id}.png"
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            plt.close()
            
            success_count += 1
            progress_bar.set_postfix({'成功': success_count, '失败': len(results_dict) - success_count})
                
        except Exception as e:
            print(f'\n处理帧 {frame_id} 时出错: {e}')
            progress_bar.set_postfix({'成功': success_count, '失败': len(results_dict) - success_count})
    
    print(f'\n✅ 可视化完成！成功处理 {success_count}/{total_frames} 帧')

if __name__ == '__main__':
    main() 