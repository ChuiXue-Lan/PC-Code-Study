import math
import sys
from typing import Iterable
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import kl_divergence
import utils
from utils.utils import *
from timm.utils import accuracy
from calculate_loss import (main_loss, pseudo_label_consistency_loss)
from make_logger import (make_metric_logger, make_log)

def train_one_epoch(model: torch.nn.Module, args, train_config,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, amp_autocast,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, model_ema=None):
    print(f"========================================== [INFO] Begin train_one_epoch: % d=========================================="%(epoch))
    model.train()

    # 初始化一个用于记录训练过程各类指标的MetricLogger对象，分隔符为两个空格
    metric_logger = MetricLogger(delimiter="  ")
    # 添加一个名为'lr'的指标，用于记录学习率，窗口大小为1，格式为小数点后6位
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 添加一个名为'min_lr'的指标，用于记录最小学习率，窗口大小为1，格式为小数点后6位
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1
    
    # 添加
    base_threshold_image = 0.6
    base_threshold_pc = 0.6
    base_threshold_depth = 0.6
    base_threshold_combined = 0.6
    max_threshold = 0.9
    increment_every = 5 # 每隔多少个 epoch 增加一次
    increment_step = 0.05 # 每次增加多少

    adjusted_threshold_image = min(max_threshold, base_threshold_image + (epoch // increment_every) * increment_step)
    adjusted_threshold_pc = min(max_threshold, base_threshold_pc + (epoch // increment_every) * increment_step)
    adjusted_threshold_depth = min(max_threshold, base_threshold_depth + (epoch // increment_every) * increment_step)
    adjusted_threshold_combined = min(max_threshold, base_threshold_combined + (epoch // increment_every) * increment_step)

    train_config["conf_threshold_image"] = adjusted_threshold_image
    train_config["conf_threshold_pc"] = adjusted_threshold_pc
    train_config["conf_threshold_depth"] = adjusted_threshold_depth
    train_config["base_threshold_combined"] = adjusted_threshold_combined
    
    if utils.utils.is_main_process():
        print(f"[INFO] Epoch {epoch}: Using conf_threshold_image = {adjusted_threshold_image:.2f}, conf_threshold_pc = {adjusted_threshold_pc:.2f}, conf_threshold_depth = {adjusted_threshold_depth:.2f}, conf_threshold_combined = {adjusted_threshold_combined:.2f}")
    
    for step, ((images_weak, images_strong, mask, pc_weak, pc_strong, depth_weak, depth_strong), targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

        # ramp-up ema decay
        model_ema.decay = train_config['model_ema_decay_init'] + (args.model_ema_decay - train_config['model_ema_decay_init']) * min(1, it/train_config['warm_it'])
        metric_logger.update(ema_decay=model_ema.decay)

        images_weak, images_strong = images_weak.to(device, non_blocking=True), images_strong.to(device, non_blocking=True)
        pc_weak, pc_strong = pc_weak.to(device, non_blocking=True), pc_strong.to(device, non_blocking=True)
        depth_weak, depth_strong = depth_weak.to(device, non_blocking=True), depth_strong.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        try:
            combine_naive= train_config['naive']
        except:
            combine_naive= False

        with torch.no_grad():
            # pseudo-label with ema model
            #image_logits, pc_logits = model_ema.ema(images_weak, pc_weak)
            image_logits, pc_logits, depth_logits = model_ema.ema(images_weak, pc_weak, depth_weak)
            print("shape", image_logits.shape, pc_logits.shape, depth_logits.shape)

            probs_ema_image = F.softmax(image_logits, dim=-1)
            probs_ema_pc = F.softmax(pc_logits, dim=-1)
            probs_ema_depth = F.softmax(depth_logits, dim=-1)

            score_image, pseudo_targets_image = probs_ema_image.max(-1)
            score_pc, pseudo_targets_pc = probs_ema_pc.max(-1)
            score_depth, pseudo_targets_depth = probs_ema_depth.max(-1)
            if score_depth.dim() == 3:
                score_depth = score_depth.mean(dim=(1, 2))  # [32]

            b = (1 / probs_ema_image.shape[1]) * torch.ones(probs_ema_image.shape).cuda()
            
            # 计算图像和点云预测概率分布与均匀分布之间的KL散度的负值
            loss_entropy_image = -kl_divergence(probs_ema_image, b)
            loss_entropy_pc = -kl_divergence(probs_ema_pc, b)
            if probs_ema_depth.dim() == 3:
                probs_ema_depth_flat = probs_ema_depth.mean(dim=(1, 2))  # [batch, 40]
                loss_entropy_depth = -kl_divergence(probs_ema_depth_flat, b)
            else:
                loss_entropy_depth = -kl_divergence(probs_ema_depth, b)
                
            if train_config['combined_pseudolabels']:  # True
                # 就是概率向量（一维），每个元素是概率值
                score_pc = score_pc* train_config['conf_weight_pc']

                # 取最大置信度的模态作为伪标签
                combined_scores = torch.stack([score_image, score_pc, score_depth], dim=0)  # (3, B)
                max_scores, max_indices = combined_scores.max(dim=0)  # (B,)

                combined_targets = torch.where(max_indices == 0, pseudo_targets_image,
                                    torch.where(max_indices == 1, pseudo_targets_pc, pseudo_targets_depth))

                conf_mask_all = max_scores > train_config['conf_threshold_combined']
                conf_mask_image = conf_mask_all
                conf_mask_pc = conf_mask_all
                conf_mask_depth = conf_mask_all 
                pseudo_targets_image = combined_targets #刚刚得分最高的那个模态的伪标签
                pseudo_targets_pc = combined_targets
                pseudo_targets_depth = combined_targets
                    
                # combined_targets = pseudo_targets_pc * (combined_scores == score_pc) + pseudo_targets_image * (combined_scores == score_image) + pseudo_targets_depth * (combined_scores == score_depth)
                # conf_mask_image = combined_scores > train_config['conf_threshold_combined']
                # conf_mask_pc = conf_mask_image
                # conf_mask_depth = conf_mask_image 
                # conf_mask_combined = conf_mask_image

                pseudolabel_agreement_loss = pseudo_label_consistency_loss(pseudo_targets_image, pseudo_targets_pc, \
                    pseudo_targets_depth, conf_mask_image, conf_mask_pc, conf_mask_depth, train_config)

            else:  # False
                conf_mask_image = score_image > train_config['conf_threshold_image']
                conf_mask_pc = score_pc > train_config['conf_threshold_pc']
                conf_mask_depth = score_depth > train_config['conf_threshold_depth']
                conf_mask_all = conf_mask_image & conf_mask_pc & conf_mask_depth
                
               
            # TODO 
            # pseudo_label_acc_image = (pseudo_targets_image[conf_mask_image] == targets[conf_mask_image]).float().mean().item()
            # conf_ratio_image = conf_mask_image.float().sum()/conf_mask_image.size(0) # 计算图像模态的高置信度比例
            # if train_config['from_scratch']:
            #     pseudo_label_acc_pc = (pseudo_targets_image[conf_mask_image] == targets[conf_mask_image]).float().mean().item()
            # else:
            #     pseudo_label_acc_pc = (pseudo_targets_pc[conf_mask_pc] == targets[conf_mask_pc]).float().mean().item()
            # conf_ratio_pc = conf_mask_pc.float().sum() / conf_mask_pc.size(0) # 计算点云模态的高置信度比例

            # metric_logger.update(conf_ratio_image=conf_ratio_image)
            # metric_logger.update(pseudo_label_acc_image=pseudo_label_acc_image)
            # metric_logger.update(conf_ratio_pc=conf_ratio_pc)
            # metric_logger.update(pseudo_label_acc_pc=pseudo_label_acc_pc)


        with amp_autocast():
            # 使用自动混合精度训练
            if args.mask:
                logits_image, logits_pc, logits_depth, loss_mim_image, loss_mim_pc, loss_mim_depth, loss_align_image, loss_align_pc, loss_align_depth, pc_image_align_logits, image_pc_align_logits, depth_image_align_logits, image_depth_align_logits, depth_pc_align_logits, pc_depth_align_logits = model(images_strong, pc_strong, depth_strong, Mask=mask)
            else:
                logits_image, logits_pc, logits_depth = model(images_strong, pc_strong, depth_strong)

            if train_config['pairwise_alignment']:
                loss, (loss_st_image, loss_st_pc, loss_st_depth, loss_fair_image, loss_fair_pc, loss_fair_depth, loss_pc_image_align, loss_image_pc_align, loss_depth_image_align, loss_image_depth_align, loss_depth_pc_align, loss_pc_depth_align, loss_align_image, loss_align_pc, loss_align_depth)\
                    = main_loss(logits_image, logits_pc, logits_depth, pseudo_targets_image, pseudo_targets_pc, pseudo_targets_depth, \
                    conf_mask_all, conf_mask_pc, conf_mask_image, conf_mask_depth, pc_image_align_logits, image_pc_align_logits, \
                    depth_image_align_logits, image_depth_align_logits, depth_pc_align_logits, pc_depth_align_logits, \
                    loss_entropy_image, loss_entropy_pc, loss_entropy_depth, loss_align_image, loss_align_pc, loss_align_depth, \
                    loss_mim_image, loss_mim_pc, loss_mim_depth, train_config, args)
            elif train_config['depth_modality_center']:
                loss, (loss_st_image, loss_st_pc, loss_st_depth, loss_fair_image, loss_fair_pc, loss_fair_depth, loss_depth_image_align, loss_image_depth_align, loss_depth_pc_align, loss_pc_depth_align, loss_align_image, loss_align_pc, loss_align_depth)\
                    =main_loss(logits_image, logits_pc, logits_depth, pseudo_targets_image, pseudo_targets_pc, pseudo_targets_depth, \
                    conf_mask_all, conf_mask_pc, conf_mask_image, conf_mask_depth, pc_image_align_logits, image_pc_align_logits, \
                    depth_image_align_logits, image_depth_align_logits, depth_pc_align_logits, pc_depth_align_logits, \
                    loss_entropy_image, loss_entropy_pc, loss_entropy_depth, train_config, args)
            
        # loss_value = loss.item()
        loss_value = loss.mean().item()
        # loss_value = loss.sum().item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
        else:
            loss = loss.mean()
            loss.backward(create_graph=False)
            optimizer.step()

        model_ema.update(model)
        torch.cuda.synchronize()

        if train_config['pairwise_alignment']:
            max_lr, min_lr = make_metric_logger(metric_logger,loss_st_image, loss_st_pc, loss_st_depth, loss_fair_image, loss_fair_pc, loss_fair_depth, loss_depth_image_align, loss_image_depth_align, \
                loss_depth_pc_align, loss_pc_depth_align, loss_align_image, loss_align_pc, loss_align_depth, loss_mim_image, loss_mim_pc, loss_mim_depth, loss_entropy_image, loss_entropy_pc, \
                loss_entropy_depth, pseudolabel_agreement_loss, optimizer, train_config, args, loss_pc_image_align, loss_image_pc_align)
        elif train_config['depth_modality_center']:
            max_lr, min_lr = make_metric_logger(metric_logger,loss_st_image, loss_st_pc, loss_st_depth, loss_fair_image, loss_fair_pc, loss_fair_depth, loss_depth_image_align, loss_image_depth_align, \
                loss_depth_pc_align, loss_pc_depth_align, loss_align_image, loss_align_pc, loss_align_depth, loss_mim_image, loss_mim_pc, loss_mim_depth, loss_entropy_image, loss_entropy_pc, \
                loss_entropy_depth, pseudolabel_agreement_loss, optimizer, train_config, args)
        
        make_log(log_writer, loss_st_image, loss_st_pc, loss_st_depth, loss_fair_image, loss_fair_pc, loss_fair_depth, loss_align_image, loss_align_pc, loss_align_depth, \
            loss_mim_image, loss_mim_pc, loss_mim_depth, max_lr, min_lr, args)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
        loss_dict={}
        for k, v in metric_logger.meters.items():
            loss_dict[k] = metric_logger.meters.get(k).value
        if utils.utils.is_main_process() and args.wandb:
            wandb.log(loss_dict)

        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device, model_ema=None, args=None):

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if model_ema is not None:
        model_ema.ema.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0][0].to(device, non_blocking=True)
        depths = batch[0][1].to(device, non_blocking=True)
        pcs = batch[0][2].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        # images = batch[0].to(device, non_blocking=True)
        # pcs = batch[1].to(device, non_blocking=True)
        # target = batch[-1].to(device, non_blocking=True)

        # compute output
        output = model(images, pcs, depths)

        acc_image = accuracy(output[0], target)[0]
        acc_pc = accuracy(output[1], target)[0]
        acc_depth = accuracy(output[2], target)[0]
        metric_logger.meters['acc1_image'].update(acc_image.item(), n=images.shape[0])
        metric_logger.meters['acc1_pc'].update(acc_pc.item(), n=images.shape[0])
        metric_logger.meters['acc1_depth'].update(acc_depth.item(), n=images.shape[0])

        if model_ema is not None:
            ema_output = model_ema.ema(images, pcs, depths)

            ema_acc1_image = accuracy(ema_output[0], target)[0]
            ema_acc1_pc = accuracy(ema_output[1], target)[0]
            ema_acc1_depth = accuracy(ema_output[2], target)[0]
            metric_logger.meters['ema_acc1_image'].update(ema_acc1_image.item(), n=images.shape[0])
            metric_logger.meters['ema_acc1_pc'].update(ema_acc1_pc.item(), n=images.shape[0])
            metric_logger.meters['ema_acc1_depth'].update(ema_acc1_depth.item(), n=images.shape[0])

    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_image))
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_pc))
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_depth))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

