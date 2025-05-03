import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pointnet2_ops import pointnet2_utils
from tqdm import tqdm
import clip
import torch_optimizer as optim
from torch.utils.tensorboard import SummaryWriter

from models import CLIP2Point
from datasets import ModelNet40Align, ShapeNetRender
from utils import IOStream

# 设置使用GPU1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(1)  # 显式设置使用GPU1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

clip_model, _ = clip.load("ViT-B/32", device='cpu')


def _init_(path):
    if not os.path.exists('/home/cls2024/ltx/tmp/'+path):
        os.makedirs('/home/cls2024/ltx/tmp/'+path)
    if not os.path.exists('/home/cls2024/ltx/tmp/'+path + '/' + args.exp_name):
        os.makedirs('/home/cls2024/ltx/tmp/'+path + '/' + args.exp_name)


def train(args, io):
    """训练函数
    
    Args:
        args: 命令行参数
        io: 日志输出工具
    """
    # print("=================================== TRAINING SETUP ======================================================")
    # 定义测试集和验证集的提示词
    test_prompts = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup',
                    'curtain', 'desk', 'door', 'dresser', 'flower pot', 'glass box', 'guitar', 'keyboard', 'lamp',
                    'laptop', 'mantel', 'monitor', 'night stand', 'person', 'piano', 'plant', 'radio', 'range hood',
                    'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv stand', 'vase', 'wardrobe',
                    'xbox']
    val_prompts = ['airplane', 'ashcan', 'bag', 'basket', 'bathtub', 'bed', 'bench', 'birdhouse', 'bookshelf', 'bottle',
                   'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellular telephone', 'chair', 'clock',
                   'computer keyboard', 'dishwasher', 'display', 'earphone', 'faucet', 'file', 'guitar', 'helmet',
                   'jar', 'knife', 'lamp', 'laptop', 'loudspeaker', 'mailbox', 'microphone', 'microwave', 'motorcycle',
                   'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote control', 'rifle', 'rocket',
                   'skateboard', 'sofa', 'stove', 'table', 'telephone', 'tower', 'train', 'vessel', 'washer']
    
    # 为每个提示词添加前缀"image of a"
    test_prompts = ['image of a ' + test_prompts[i] for i in range(len(test_prompts))]
    val_prompts = ['image of a ' + val_prompts[i] for i in range(len(val_prompts))]
    
    # 使用CLIP模型对提示词进行编码
    test_prompts_ = clip.tokenize(test_prompts)
    test_prompt_feats = clip_model.encode_text(test_prompts_)
    test_prompt_feats = test_prompt_feats / test_prompt_feats.norm(dim=-1, keepdim=True)  # 特征归一化
    test_prompt_feats = test_prompt_feats
    val_prompts_ = clip.tokenize(val_prompts)
    val_prompt_feats = clip_model.encode_text(val_prompts_)
    val_prompt_feats = val_prompt_feats / val_prompt_feats.norm(dim=-1, keepdim=True)  # 特征归一化
    val_prompt_feats = val_prompt_feats

    # 创建数据加载器
    train_dataloader = DataLoader(ShapeNetRender(partition='train', num_points=args.num_points),
                                  batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(ShapeNetRender(partition='test', num_points=args.num_points),
                            batch_size=args.test_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(ModelNet40Align(num_points=args.num_points), batch_size=args.test_batch_size,
                             num_workers=4)
                             
    # 设置GPU设备
    gpus = [1]  # 只使用GPU1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # =================================== 初始化模型 ==========================================================
    # print("=================================== INIT MODEL ==========================================================")
    # 创建tensorboard记录器
    summary_writer = SummaryWriter("/home/cls2024/ltx/tmp/pre_results/%s/tensorboard" % (args.exp_name))
    
    # 初始化模型
    model = CLIP2Point(args)
    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])  # 多卡训练设置
    model = model.to(device)
    
    # 冻结图像编码器参数
    for name, param in model.named_parameters():
        if 'image_model' in name:
            param.requires_grad_(False)
            
    # 将提示词特征转移到GPU
    val_prompt_feats = val_prompt_feats.to(device)
    test_prompt_feats = test_prompt_feats.to(device)
    
    # ==================================== 训练循环 ======================================================
    # print("=================================== TRAINING LOOP ======================================================")
    # 初始化优化器
    optimizer = optim.Lamb(model.parameters(), lr=0.006, weight_decay=1e-4)
    # 初始化学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * len(train_dataloader),
        T_mult=1,
        eta_min=max(1e-2 * 1e-3, 1e-6),
        last_epoch=-1,
    )

    n_epochs = args.epoch
    max_val_acc = 0  # 记录最佳验证集准确率
    max_test_acc = 0  # 记录最佳测试集准确率
    
    # 开始训练循环
    for epoch in range(n_epochs):
        model.train()
        loss_sum = 0  # 总损失
        depth_sum = 0  # 深度损失
        image_sum = 0  # 图像损失

        # 训练阶段
        print("=================================== epoch：%d TRAINING PHASE ======================================================" % (epoch + 1))
        for (image, points, a, e, d) in tqdm(train_dataloader):
            optimizer.zero_grad()  # 清空梯度
            # 将数据转移到GPU
            image = image.to(device)
            points = points.to(device)
            a = a.unsqueeze(-1).to(device)
            e = e.unsqueeze(-1).to(device)
            d = d.unsqueeze(-1).to(device)
            # 前向传播
            loss, image_loss, depth_loss = model(points, image, a, e, d)
            loss = torch.mean(loss)
            # 累计损失
            image_sum += torch.mean(image_loss).item()
            depth_sum += torch.mean(depth_loss).item()
            loss_sum += loss.item()
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 验证阶段
        print("=================================== epoch：%d VALIDATION PHASE ======================================================" % (epoch + 1))
        model.eval()
        with torch.no_grad():
            correct_num = 0
            total = 0
            for (points, label) in tqdm(val_loader):
                b = points.shape[0]
                points = points.to(device)
                img_feats = model.module.infer(points)
                # 计算logits和预测结果
                logits = img_feats @ val_prompt_feats.t()
                logits = logits.reshape(b, args.views, -1)
                logits = torch.sum(logits, dim=1)
                probs = logits.softmax(dim=-1)
                index = torch.max(probs, dim=1).indices
                correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
                total += len(label)
        val_acc = correct_num / total

        # 测试阶段
        print("=================================== epoch：%d TESTING PHASE ======================================================" % (epoch + 1))
        with torch.no_grad():
            correct_num = 0
            total = 0
            for (points, label) in tqdm(test_loader):
                b = points.shape[0]
                points = points.to(device)
                img_feats = model.module.infer(points, True)
                # 计算logits和预测结果
                logits = img_feats @ test_prompt_feats.t()
                logits = logits.reshape(b, args.views, -1)
                logits = torch.sum(logits, dim=1)
                probs = logits.softmax(dim=-1)
                index = torch.max(probs, dim=1).indices
                correct_num += torch.sum(torch.eq(index.detach().cpu(), label)).item()
                total += len(label)
        test_acc = correct_num / total

        # 计算平均损失
        depth_loss = depth_sum / len(train_dataloader)
        image_loss = image_sum / len(train_dataloader)
        mean_loss = loss_sum / len(train_dataloader)
        
        # 打印训练信息
        io.cprint(
            'epoch%d total_loss: %.4f, image_loss: %.4f, depth_loss: %.4f, balance_weights: %.4f, val_acc: %.4f, test_acc: %.4f' % (
            epoch + 1, mean_loss, image_loss, depth_loss, model.module.weights, val_acc, test_acc))
            
        # 记录tensorboard日志
        summary_writer.add_scalar('train/loss', mean_loss, epoch + 1)
        summary_writer.add_scalar('train/depth_loss', depth_loss, epoch + 1)
        summary_writer.add_scalar('train/image_loss', image_loss, epoch + 1)
        summary_writer.add_scalar('train/weights', model.module.weights, epoch + 1)
        summary_writer.add_scalar("val/acc", val_acc, epoch + 1)
        summary_writer.add_scalar("test/acc", test_acc, epoch + 1)
        
        # 保存最佳验证集模型
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), '%s/%s/best_val.pth' % ('/home/cls2024/ltx/tmp/pre_results', args.exp_name))
            io.cprint('save the best val acc at %d' % (epoch + 1))
            
        # 保存最佳测试集模型
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            torch.save(model.state_dict(), '%s/%s/best_test.pth' % ('/home/cls2024/ltx/tmp/pre_results', args.exp_name))
            io.cprint('save the best test acc at %d' % (epoch + 1))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='test', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--views', type=int, default=10)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--dim', type=int, default=0, choices=[0, 512], help='0 if the view angle is not learnable')
    parser.add_argument('--model', type=str, default='PointNet', metavar='N',
                        choices=['DGCNN', 'PointNet'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--batch_size', type=int, default=256, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epoch', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    args = parser.parse_args()

    _init_('pre_results')
    io = IOStream('/home/cls2024/ltx/tmp/pre_results' + '/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    train(args, io)
