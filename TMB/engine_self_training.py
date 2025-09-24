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

    # 这里的(images_weak, images_strong, mask, pc_weak, pc_strong, pc_mask), targets 是从 data_loader 中获取的一个batch的数据。
    # data_loader 是 PyTorch 的 DataLoader 实例，其每次迭代返回一个batch的数据。
    # 通常，数据集的 __getitem__ 方法会返回一个元组：(输入数据, 标签)。
    # 在本例中，输入数据被进一步拆分为六个部分：
    #   - images_weak: 图像的弱增强版本
    #   - images_strong: 图像的强增强版本
    #   - mask: 图像掩码
    #   - pc_weak: 点云的弱增强版本
    #   - pc_strong: 点云的强增强版本
    #   - pc_mask: 点云掩码
    # targets: 真实标签
    # 这些数据通常由自定义的Dataset的__getitem__方法返回，形如：
    #   return (images_weak, images_strong, mask, pc_weak, pc_strong, pc_mask), targets

    # enumerate(metric_logger.log_every(data_loader, print_freq, header)) 的作用如下：
    # - metric_logger.log_every 是一个包装器，用于每隔 print_freq 步打印一次训练进度和指标，并在每个epoch开始时打印header。
    # - enumerate 用于为每个batch分配一个step索引。
    # 这样，for循环每次迭代会获得当前step编号，以及一个batch的输入数据和标签。
    
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

            probs_ema_image = F.softmax(image_logits, dim=-1)
            probs_ema_pc = F.softmax(pc_logits, dim=-1)
            probs_ema_depth = F.softmax(depth_logits, dim=-1)

            score_image, pseudo_targets_image = probs_ema_image.max(-1)
            score_pc, pseudo_targets_pc = probs_ema_pc.max(-1)
            score_depth, pseudo_targets_depth = probs_ema_depth.max(-1)

            b = (1 / probs_ema_image.shape[1]) * torch.ones(probs_ema_image.shape).cuda()
            
            # 计算图像和点云预测概率分布与均匀分布之间的KL散度的负值
            # loss_entropy_image 计算图像模态的熵损失
            # loss_entropy_pc 计算点云模态的熵损失
            # probs_ema_image 和 probs_ema_pc 分别是图像和点云的预测概率分布
            # b 是均匀分布
            # KL散度越大，说明模型预测越偏离均匀分布（越有信心）。
            # 取负号（-kl_divergence）后，作为损失项时，鼓励模型输出更"均匀"的分布，即增加预测的不确定性（熵），防止过拟合或过度自信。
            # 负的KL散度作为损失，实际上是鼓励模型输出更高熵、更均匀的分布。
            loss_entropy_image = -kl_divergence(probs_ema_image, b)
            loss_entropy_pc = -kl_divergence(probs_ema_pc, b)
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
                pseudo_targets_image = combined_targets
                pseudo_targets_pc = combined_targets
                pseudo_targets_depth = combined_targets
                    
                # combined_targets = pseudo_targets_pc * (combined_scores == score_pc) + pseudo_targets_image * (combined_scores == score_image) + pseudo_targets_depth * (combined_scores == score_depth)

                # conf_mask_image = combined_scores > train_config['conf_threshold_combined']
                # conf_mask_pc = conf_mask_image
                # conf_mask_depth = conf_mask_image 
                # conf_mask_combined = conf_mask_image

                # # TODO: 这里需要修改，因为现在有三个模态，所以需要修改伪标签一致性损失的计算方式
                # pseudolabel_agreement_loss = (pseudo_targets_image[conf_mask_image]!=pseudo_targets_pc[conf_mask_pc]).sum()/pseudo_targets_image[conf_mask_image].shape[0]
                # # 👆
                pseudo_targets_image = combined_targets #刚刚得分最高的那个模态的伪标签
                pseudo_targets_pc = combined_targets
                pseudo_targets_depth = combined_targets
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
                # 如果使用掩码,模型会返回更多的输出,包括:
                # - logits_image/pc: 图像和点云分支的预测logits 
                # - loss_mim_image/pc: 掩码图像建模损失
                # - loss_align_image/pc: 特征对齐损失
                # - pc_image_align_logits/image_pc_align_logits: 跨模态对齐的logits
                logits_image, logits_pc, logits_depth, loss_mim_image, loss_mim_pc, loss_mim_depth, loss_align_image, loss_align_pc, loss_align_depth, pc_image_align_logits, image_pc_align_logits, depth_image_align_logits, image_depth_align_logits, depth_pc_align_logits, pc_depth_align_logits = model(images_strong, pc_strong, depth_strong, Mask=mask)
            else:
                # 不使用掩码时只返回预测logits
                # logits_image, logits_pc = model(images_strong, pc_strong)
                logits_image, logits_pc, logits_depth = model(images_strong, pc_strong, depth_strong)

            # 计算自训练损失
            loss_st_image = F.cross_entropy(logits_image[conf_mask_all], pseudo_targets_image[conf_mask_all])
            loss_st_pc = F.cross_entropy(logits_pc[conf_mask_all], pseudo_targets_pc[conf_mask_all])
            loss_st_depth = F.cross_entropy(logits_depth[conf_mask_all], pseudo_targets_depth[conf_mask_all])

            if train_config['trans_pcl_img']:
                # 如果启用点云到图像的迁移,使用点云的置信度掩码和伪标签来监督图像分支
                loss_st_image = F.cross_entropy(logits_image[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])
            else:
                # 否则使用图像自身的置信度掩码和伪标签
                loss_st_image = F.cross_entropy(logits_image[conf_mask_image], pseudo_targets_image[conf_mask_image])

            # 计算点云分支的自训练损失
            if train_config['from_scratch']:
                # 从头训练时使用图像的置信度掩码和伪标签
                loss_st_pc = F.cross_entropy(logits_pc[conf_mask_image], pseudo_targets_image[conf_mask_image])
            elif train_config['trans_img_pcl']:
                # 如果启用图像到点云的迁移,使用图像的置信度掩码和伪标签
                loss_st_pc = F.cross_entropy(logits_pc[conf_mask_image], pseudo_targets_image[conf_mask_image])
            else:
                # 否则使用点云自身的置信度掩码和伪标签
                loss_st_pc = F.cross_entropy(logits_pc[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])

            # fairness regularization
            # 计算图像分支的公平性正则化损失
            # 首先计算图像分支的softmax概率
            probs_image = F.softmax(logits_image,dim=-1)
            # 收集所有GPU上的概率
            probs_all_image = all_gather_with_grad(probs_image)
            # 计算所有GPU上的平均预测概率
            probs_batch_avg_image = probs_all_image.mean(0) 

            # 计算点云分支的公平性正则化损失
            # 计算点云分支的softmax概率
            probs_pc = F.softmax(logits_pc, dim=-1)
            # 收集所有GPU上的概率
            probs_all_pc = all_gather_with_grad(probs_pc)
            # 计算所有GPU上的平均预测概率
            probs_batch_avg_pc = probs_all_pc.mean(0)  

            # 计算深度分支的公平性正则化损失
            probs_depth = F.softmax(logits_depth, dim=-1)
            probs_all_depth = all_gather_with_grad(probs_depth)
            probs_batch_avg_depth = probs_all_depth.mean(0)

            # 计算图像分支的公平性损失
            probs_avg = probs_batch_avg_image
            loss_fair_image = -(torch.log(probs_avg)).mean()
            # 计算点云分支的公平性损失
            probs_avg = probs_batch_avg_pc
            loss_fair_pc = -(torch.log(probs_avg)).mean()
            # 计算深度分支的公平性损失
            probs_avg = probs_batch_avg_depth
            loss_fair_depth = -(torch.log(probs_avg)).mean()

            # 如果使用掩码,计算额外的对齐损失
            if args.mask:
                # def align_loss(f1, f2):
                #     return (1 - F.cosine_similarity(f1, f2, dim=-1)).mean()

                # loss_align_img_pc = align_loss(img_feat, pc_feat)
                # loss_align_img_depth = align_loss(img_feat, depth_feat)
                # loss_align_pc_depth = align_loss(pc_feat, depth_feat)
                # 创建对角矩阵作为标签
                labels = torch.eye(pc_image_align_logits.shape[0]).cuda()
                # 计算点云到图像的对齐损失
                loss_pc_image_align = F.cross_entropy(pc_image_align_logits, labels)
                # 计算图像到点云的对齐损失
                loss_image_pc_align = F.cross_entropy(image_pc_align_logits, labels)
                # 计算深度图到图像的对齐损失
                loss_depth_image_align = F.cross_entropy(depth_image_align_logits, labels)
                # 计算图像到深度图的对齐损失
                loss_image_depth_align = F.cross_entropy(image_depth_align_logits, labels)
                # 计算深度图到点云的对齐损失
                loss_depth_pc_align = F.cross_entropy(depth_pc_align_logits, labels)
                # 计算点云到深度图的对齐损失
                loss_pc_depth_align = F.cross_entropy(pc_depth_align_logits, labels)
                # 计算全局-局部特征对齐损失
                loss_align_image = torch.mean(loss_align_image)
                loss_align_pc = torch.mean(loss_align_pc)
                loss_align_depth = torch.mean(loss_align_depth)


                # 根据训练配置计算总损失
                if train_config['only_image']:
                    # 仅使用图像分支的损失
                    loss = loss_st_image + train_config['w_fair_image'] * loss_fair_image + train_config['w_mim_image']*loss_mim_image + train_config['w_align_image'] * loss_align_image
                elif train_config['only_pc']:
                    # 仅使用点云分支的损失
                    loss= loss_st_pc + train_config['w_fair_pc'] * loss_fair_pc + train_config['w_mim_pc']*loss_mim_pc + train_config['w_align_pc'] * loss_align_pc
                elif train_config['only_depth']:
                    # 仅使用深度分支的损失
                    loss= loss_st_depth + train_config['w_fair_depth'] * loss_fair_depth + train_config['w_mim_depth']*loss_mim_depth + train_config['w_align_depth'] * loss_align_depth
                else:
                    # 使用两个分支的组合损失
                    loss = loss_st_image + train_config['w_fair_image'] * loss_fair_image + train_config['w_mim_image']*loss_mim_image + train_config['w_align_image'] * loss_align_image + args.pc_loss_weight * (loss_st_pc + train_config['w_fair_pc'] * loss_fair_pc + train_config['w_mim_pc']*loss_mim_pc +train_config['w_align_pc'] * loss_align_pc)
                    # 添加深度图分支的损失
                    loss = loss + loss_st_depth + train_config['w_fair_depth'] * loss_fair_depth + train_config['w_mim_depth']*loss_mim_depth + train_config['w_align_depth'] * loss_align_depth

                    # 添加可选的额外损失项
                    if train_config['image_pc_align']:
                        # 添加图像-点云对齐损失
                        loss = loss + train_config['w_image_pc_align']*loss_pc_image_align + train_config['w_image_pc_align']*loss_image_pc_align
                    # 添加深度图-图像对齐损失
                    if train_config['depth_image_align']:
                        loss = loss + train_config['w_depth_image_align']*loss_depth_image_align + train_config['w_depth_image_align']*loss_image_depth_align
                    # 添加点云-深度图对齐损失
                    if train_config['pc_depth_align']:
                        loss = loss + train_config['w_pc_depth_align']*loss_pc_depth_align + train_config['w_pc_depth_align']*loss_depth_pc_align
                    # if train_config['pseudolabel_agreement_loss']:
                    #     # 添加伪标签一致性损失
                    #     loss = loss + train_config['w_pseudo_agree'] * pseudolabel_agreement_loss
                    if train_config['entropy_image']:
                        # 添加图像熵损失
                        loss = loss + loss_entropy_image
                    if train_config['entropy_pc']:
                        # 添加点云熵损失
                        loss = loss + loss_entropy_pc
                    if train_config['entropy_depth']:
                        # 添加深度熵损失
                        loss = loss + loss_entropy_depth

            else:
                # 不使用掩码时的损失计算
                if train_config['only_image']:
                    # 仅使用图像分支损失
                    loss = loss_st_image + train_config['w_fair_image'] * loss_fair_image
                elif train_config['only_pc']:
                    # 仅使用点云分支损失
                    loss = loss_st_pc + train_config['w_fair_pc'] * loss_fair_pc
                elif train_config['only_depth']:
                    # 仅使用深度分支损失
                    loss = loss_st_depth + train_config['w_fair_depth'] * loss_fair_depth
                else:
                    # 使用3分支的组合损失
                    loss = loss_st_image + loss_st_pc + loss_st_depth + train_config['w_fair_image'] * loss_fair_image + train_config['w_fair_pc'] * loss_fair_pc + train_config['w_fair_depth'] * loss_fair_depth

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

        metric_logger.update(loss_st_image=loss_st_image.item())
        metric_logger.update(loss_fair_image=loss_fair_image.item())
        metric_logger.update(loss_st_pc=loss_st_pc.item())
        metric_logger.update(loss_fair_pc=loss_fair_pc.item())
        metric_logger.update(loss_st_depth=loss_st_depth.item())
        metric_logger.update(loss_fair_depth=loss_fair_depth.item())
        metric_logger.update(loss_entropy_image=loss_entropy_image.item())
        metric_logger.update(loss_entropy_pc=loss_entropy_pc.item())
        metric_logger.update(loss_entropy_depth=loss_entropy_depth.item())
        
        # if train_config['combined_pseudolabels']:
            # metric_logger.update(loss_pseudolabel_agreement=pseudolabel_agreement_loss.item())

        if args.mask:
            metric_logger.update(loss_pc_image_align=loss_pc_image_align.item())
            metric_logger.update(loss_image_pc_align=loss_image_pc_align.item())
            metric_logger.update(loss_mim_image=loss_mim_image.item())
            metric_logger.update(loss_align_image=loss_align_image.item())
            metric_logger.update(loss_mim_pc=loss_mim_pc.item())
            metric_logger.update(loss_align_pc=loss_align_pc.item())
            metric_logger.update(loss_depth_image_align=loss_depth_image_align.item())
            metric_logger.update(loss_image_depth_align=loss_image_depth_align.item())
            metric_logger.update(loss_depth_pc_align=loss_depth_pc_align.item())
            metric_logger.update(loss_pc_depth_align=loss_pc_depth_align.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:
            log_writer.update(loss_st_image=loss_st_image.item(), head="train")
            log_writer.update(loss_fair_image=loss_fair_image.item(), head="train")
            log_writer.update(loss_st_pc=loss_st_pc.item(), head="train")
            log_writer.update(loss_fair_pc=loss_fair_pc.item(), head="train")
            log_writer.update(loss_st_depth=loss_st_depth.item(), head="train")
            log_writer.update(loss_fair_depth=loss_fair_depth.item(), head="train")
            log_writer.update(loss_entropy_depth=loss_entropy_depth.item(), head="train")

            if args.mask:
                log_writer.update(loss_mim_image=loss_mim_image.item(), head="train")
                log_writer.update(loss_align_image=loss_align_image.item(), head="train")
                log_writer.update(loss_mim_pc=loss_mim_pc.item(), head="train")
                log_writer.update(loss_align_pc=loss_align_pc.item(), head="train")
                log_writer.update(loss_depth_image_align=loss_depth_image_align.item(), head="train")
                log_writer.update(loss_image_depth_align=loss_image_depth_align.item(), head="train")
                log_writer.update(loss_depth_pc_align=loss_depth_pc_align.item(), head="train")
                log_writer.update(loss_pc_depth_align=loss_pc_depth_align.item(), head="train")

            # log_writer.update(conf_ratio_image=conf_ratio_image, head="train")
            # log_writer.update(pseudo_label_acc_image=pseudo_label_acc_image, head="train")
            # log_writer.update(conf_ratio_pc=conf_ratio_pc, head="train")
            # log_writer.update(pseudo_label_acc_pc=pseudo_label_acc_pc, head="train")

            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()

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

