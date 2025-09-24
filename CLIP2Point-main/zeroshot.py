from copy import deepcopy
import torch
import argparse
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
from pointnet2_ops import pointnet2_utils

from datasets import ModelNet10, ModelNet40Align, ModelNet40Ply, ScanObjectNN
from render.selector import Selector
from render.render import Renderer
from utils import read_state_dict
import os

# 设置使用GPU1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(1)  # 显式设置使用GPU1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

clip_model, _ = clip.load('ViT-B/32', device='cpu')


def inference(args):    
    if args.dataset == 'ModelNet10':
        dataset = ModelNet10()
        prompts = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    elif args.dataset == 'ModelNet40':
        dataset = ModelNet40Ply()
        prompts = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower pot', 'glass box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night stand', 'person', 'piano', 'plant', 'radio', 'range hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv stand', 'vase', 'wardrobe', 'xbox']
    else:
        dataset = ScanObjectNN()
        prompts = ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']
        

    dataloader = DataLoader(dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=True)
    prompts = ['image of a ' + prompts[i] for i in range(len(prompts))]
    prompts = clip.tokenize(prompts)
    prompts = clip_model.encode_text(prompts)
    prompts_feats = prompts / prompts.norm(dim=-1, keepdim=True)
    
    # =================================== 初始化模型 ===========================================================
    # 设置设备为GPU(如果可用)或CPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    # 复制CLIP视觉编码器并移至设备
    model = deepcopy(clip_model.visual).to(device)
    
    # 如果提供了checkpoint,加载模型权重
    if args.ckpt is not None:
        model.load_state_dict(read_state_dict(args.ckpt))
        
    # 初始化视角选择器和渲染器
    selector = Selector(args.views, 0).to(device)  # 选择多视角的角度
    render = Renderer(points_per_pixel=1, points_radius=0.02).to(device)  # 点云渲染器
    prompt_feats = prompts_feats.to(device)  # 文本提示特征移至设备
    
    # ==================================== 测试循环 =====================================================
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        correct_num = 0  # 正确预测数
        total = 0  # 总样本数
        for (points, label) in tqdm(dataloader):
            points = points.to(device)
            
            # 对ScanObjectNN数据集进行FPS采样至1024个点
            if args.dataset == 'ScanObjectNN':
                fps_idx = pointnet2_utils.furthest_point_sample(points, 1024)
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
                
            # 获取多视角的方位角、仰角和距离
            c_views_azim, c_views_elev, c_views_dist = selector(points)
            
            # 渲染多视角图像,ScanObjectNN不进行旋转增强
            if args.dataset == 'ScanObjectNN':
                images = render(points, c_views_azim, c_views_elev, c_views_dist, args.views, rot=False)
            else:
                images = render(points, c_views_azim, c_views_elev, c_views_dist, args.views, rot=True)
                
            # 处理渲染图像
            b, n, c, h, w = images.shape
            images = images.reshape(-1, c, h, w)  # 展平batch和视角维度
            
            # 提取图像特征并归一化
            image_feats = model(images)
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            
            # 计算与文本提示的相似度
            logits = image_feats @ prompt_feats.t()
            logits = logits.reshape(b, n, -1)  # 恢复batch和视角维度
            logits = torch.sum(logits, dim=1)  # 聚合多视角预测
            
            # 获取预测结果
            probs = logits.softmax(dim=-1)  # softmax得到概率
            index = torch.max(probs, dim=1).indices  # 取最大概率的类别
            
            # 统计正确预测数
            correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
            total += len(label)
            
    # 计算测试准确率
    test_acc = correct_num / total
    print(test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zero-shot Point Cloud Classification')
    parser.add_argument('--dataset', type=str, choices=['ModelNet10', 'ModelNet40', 'ScanObjectNN'])
    parser.add_argument('--views', type=int, default=6)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    args = parser.parse_args()

    inference(args)
