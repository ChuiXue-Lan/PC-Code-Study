import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

from data.dataset_3d import ModelNet40_img_pcl_depth_multiview, customized_collate_fn
from models.CrossMoST_models import CrossMoST_3Modal
from utils.utils import (NativeScalerWithGradNormCount as NativeScaler,
                       create_optimizer, init_distributed_mode)
from utils import utils

def get_args():
    parser = argparse.ArgumentParser('CrossMoST training script', add_help=False)
    
    # 基本配置
    parser.add_argument('--config', default='config/modelnet40_crossmost_3modal.yaml', type=str)
    parser.add_argument('--output_dir', default='outputs/modelnet40_crossmost_3modal', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', action='store_true')
    parser.set_defaults(pin_memory=True)
    
    # 分布式训练
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    
    # 训练配置
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--save_freq', default=10, type=int)
    
    # 优化器配置
    parser.add_argument('--opt', default='adamw', type=str, help='优化器类型 (adamw/adam/sgd)')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--clip_grad', default=None, type=float)
    parser.add_argument('--layer_decay', default=0.75, type=float)
    
    # 模型配置
    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--model_ema_decay', default=0.9998, type=float)
    parser.add_argument('--model_ema_force_cpu', action='store_true')
    parser.add_argument('--resume', default='', type=str)
    
    # 其他配置
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--run_id', type=str)
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # 初始化分布式训练
    init_distributed_mode(args)
    
    # 设置随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # 加载配置文件
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建模型
    model = CrossMoST_3Modal(config)
    model.to(args.device)
    
    # 创建优化器
    optimizer = create_optimizer(args, model)
    
    # 创建损失缩放器
    loss_scaler = NativeScaler()
    
    # 创建数据集和数据加载器
    train_config = config.dataset.copy()
    train_config.subset = 'train'
    val_config = config.dataset.copy()
    val_config.subset = 'test'
    
    dataset_train = ModelNet40_img_pcl_depth_multiview(
        config=train_config
    )
    dataset_val = ModelNet40_img_pcl_depth_multiview(
        config=val_config
    )
    
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    train_loader = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        collate_fn=customized_collate_fn,
    )
    
    val_loader = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        collate_fn=customized_collate_fn,
    )
    
    # 创建学习率调度器
    lr_scheduler = utils.cosine_scheduler(
        args.lr * args.batch_size / 256,
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    
    # 创建模型EMA
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )
        
    # 加载检查点
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if model_ema is not None:
            model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        
    # 分布式训练
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True
        )
        
    # 开始训练
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=args.device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            model_ema=model_ema,
            args=args,
            config=config
        )
        
        # 评估
        if (epoch + 1) % args.eval_freq == 0:
            val_stats = evaluate(
                model=model,
                data_loader=val_loader,
                device=args.device,
                epoch=epoch,
                args=args,
                config=config
            )
            
            # 保存检查点
            if args.output_dir and utils.is_main_process():
                checkpoint_path = os.path.join(
                    args.output_dir,
                    f'checkpoint_{epoch}.pth'
                )
                state_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if model_ema is not None:
                    state_dict['model_ema'] = model_ema.state_dict()
                utils.save_on_master(state_dict, checkpoint_path)
                
        # 记录日志
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }
        if (epoch + 1) % args.eval_freq == 0:
            log_stats.update({f'val_{k}': v for k, v in val_stats.items()})
            
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    main() 