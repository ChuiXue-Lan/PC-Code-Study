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

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    skipped_steps = 0
    print_freq = 1
    
    # 添加
    base_threshold_image = 0.6
    base_threshold_pc = 0.6
    base_threshold_combined = 0.6
    max_threshold = 0.9
    increment_every = 5 # 每隔多少个 epoch 增加一次
    increment_step = 0.05 # 每次增加多少

    adjusted_threshold_image = min(max_threshold, base_threshold_image + (epoch // increment_every) * increment_step)
    adjusted_threshold_pc = min(max_threshold, base_threshold_pc + (epoch // increment_every) * increment_step)
    adjusted_threshold_combined = min(max_threshold, base_threshold_combined + (epoch // increment_every) * increment_step)

    train_config["conf_threshold_image"] = adjusted_threshold_image
    train_config["conf_threshold_pc"] = adjusted_threshold_pc
    # train_config["base_threshold_combined"] = adjusted_threshold_combined
    train_config["conf_threshold_combined"] = adjusted_threshold_combined

    if utils.utils.is_main_process():
        print(f"[INFO] Epoch {epoch}: Using conf_threshold_image = {adjusted_threshold_image:.2f}, conf_threshold_pc = {adjusted_threshold_pc:.2f}, conf_threshold_combined = {adjusted_threshold_combined:.2f}")
    # 。。。

    for step, ((images_weak, images_strong, mask, pc_weak, pc_strong, pc_mask), targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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
        mask = mask.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        try:
            combine_naive= train_config['naive']
        except:
            combine_naive= False

        # with torch.no_grad():
            # # pseudo-label with ema model
            # image_logits, pc_logits = model_ema.ema(images_weak, pc_weak)
            # probs_ema_image = F.softmax(image_logits, dim=-1)
            # probs_ema_pc = F.softmax(pc_logits, dim=-1)

            # score_image, pseudo_targets_image = probs_ema_image.max(-1)
            # score_pc, pseudo_targets_pc = probs_ema_pc.max(-1)

            # b = (1 / probs_ema_image.shape[1]) * torch.ones(probs_ema_image.shape).cuda()
            # loss_entropy_image = -kl_divergence(probs_ema_image, b)
            # loss_entropy_pc = -kl_divergence(probs_ema_pc, b)

            # if train_config['combined_pseudolabels']:
            #     score_pc = score_pc* train_config['conf_weight_pc']

            #     if train_config['agreement_pseudolabels']:
            #         combined_scores = torch.min(score_pc, score_image)
            #         conf_mask = (pseudo_targets_pc == pseudo_targets_image)*(combined_scores > train_config['agreement_pseudolabels_min_thresh'])
            #         conf_mask_image = conf_mask
            #         conf_mask_pc = conf_mask_image
            #         pseudolabel_agreement_loss = (pseudo_targets_image[conf_mask_image] != pseudo_targets_pc[
            #             conf_mask_pc]).sum() / pseudo_targets_image[conf_mask_image].shape[0]


            #     else:
            #         if combine_naive:
            #             bs = score_pc.shape[0]
            #             picked = torch.randint(2, (bs,)).cuda()
            #             combined_scores = score_pc * (picked == 1) + score_image * (picked == 0)
            #         else:
            #             combined_scores = torch.max(score_pc, score_image)
            #         combined_targets = pseudo_targets_pc * (combined_scores == score_pc) + pseudo_targets_image * (combined_scores == score_image)

            #         conf_mask_image = combined_scores > train_config['conf_threshold_combined']
            #         conf_mask_pc = conf_mask_image

            #         pseudolabel_agreement_loss = (pseudo_targets_image[conf_mask_image]!=pseudo_targets_pc[conf_mask_pc]).sum()/pseudo_targets_image[conf_mask_image].shape[0]
            #         pseudo_targets_image = combined_targets
            #         pseudo_targets_pc = combined_targets

            # else:
            #     conf_mask_image = score_image > train_config['conf_threshold_image']
            #     conf_mask_pc = score_pc > train_config['conf_threshold_pc']

            # pseudo_label_acc_image = (pseudo_targets_image[conf_mask_image] == targets[conf_mask_image]).float().mean().item()
            # conf_ratio_image = conf_mask_image.float().sum()/conf_mask_image.size(0)
            # if train_config['from_scratch']:
            #     pseudo_label_acc_pc = (pseudo_targets_image[conf_mask_image] == targets[conf_mask_image]).float().mean().item()
            # else:
            #     pseudo_label_acc_pc = (pseudo_targets_pc[conf_mask_pc] == targets[conf_mask_pc]).float().mean().item()
            # conf_ratio_pc = conf_mask_pc.float().sum() / conf_mask_pc.size(0)

            # metric_logger.update(conf_ratio_image=conf_ratio_image)
            # metric_logger.update(pseudo_label_acc_image=pseudo_label_acc_image)
            # metric_logger.update(conf_ratio_pc=conf_ratio_pc)
            # metric_logger.update(pseudo_label_acc_pc=pseudo_label_acc_pc)
            # -------------------- 新版：安全处理伪标签精度、避免 NaN --------------------m1
        with torch.no_grad():
            image_logits, pc_logits = model_ema.ema(images_weak, pc_weak)
            probs_ema_image = F.softmax(image_logits, dim=-1)
            probs_ema_pc = F.softmax(pc_logits, dim=-1)

            score_image, pseudo_targets_image = probs_ema_image.max(-1)
            score_pc, pseudo_targets_pc = probs_ema_pc.max(-1)

            b = (1 / probs_ema_image.shape[1]) * torch.ones(probs_ema_image.shape).cuda()
            loss_entropy_image = -kl_divergence(probs_ema_image, b)
            loss_entropy_pc = -kl_divergence(probs_ema_pc, b)

            if train_config['combined_pseudolabels']:
                score_pc = score_pc * train_config['conf_weight_pc']

                if train_config['agreement_pseudolabels']:
                    combined_scores = torch.min(score_pc, score_image)
                    conf_mask = (pseudo_targets_pc == pseudo_targets_image) * (
                        combined_scores > train_config['agreement_pseudolabels_min_thresh']
                    )
                    conf_mask_image = conf_mask
                    conf_mask_pc = conf_mask

                    if conf_mask_image.sum() > 0:
                        pseudolabel_agreement_loss = (
                            (pseudo_targets_image[conf_mask_image] != pseudo_targets_pc[conf_mask_pc])
                            .float()
                            .mean()
                        )
                    else:
                        pseudolabel_agreement_loss = torch.tensor(0.0, device=device)

                else:
                    if train_config.get("naive", False):
                        bs = score_pc.shape[0]
                        picked = torch.randint(2, (bs,), device=device)
                        combined_scores = score_pc * (picked == 1) + score_image * (picked == 0)
                    else:
                        combined_scores = torch.max(score_pc, score_image)

                    combined_targets = (
                        pseudo_targets_pc * (combined_scores == score_pc)
                        + pseudo_targets_image * (combined_scores == score_image)
                    )
                    conf_mask_image = score_image > train_config['conf_threshold_image']
                    conf_mask_pc = score_pc > train_config['conf_threshold_image']
                    # 基础共享掩码（保留高置信度的公共区域）
                    shared_mask = conf_mask_image & conf_mask_pc


                    # # 图像特有的高置信区域
                    # image_specific = conf_mask_image & (~conf_mask_pc)

                    # # 点云特有的高置信区域
                    # pc_specific = conf_mask_pc & (~conf_mask_image)

                    # # 最终掩码：共享区域 + 各自特有的高置信区域
                    # conf_mask_image_final = shared_mask | (image_specific & (score_image > train_config['conf_threshold_image']))
                    # conf_mask_pc_final = shared_mask | (pc_specific & (score_pc > train_config['conf_threshold_image']))
                    # conf_mask_image = conf_mask_image_final
                    # conf_mask_pc = conf_mask_pc_final
                    
                    # conf_mask_image = combined_scores > train_config['conf_threshold_combined']
                    # conf_mask_pc = conf_mask_image
                    
                    # 计算置信度比例
                    conf_ratio = score_image / (score_pc + 1e-8)  # 防止除零

                    ratio_threshold = 1.5
                    # 图像主导区域（图像置信度显著高于点云）
                    image_dominant = conf_ratio > ratio_threshold  # 例如 threshold=1.5

                    # 点云主导区域（点云置信度显著高于图像）
                    pc_dominant = conf_ratio < (1/ratio_threshold)

                    # 最终掩码：各自主导区域 + 共享高置信区域
                    conf_mask_image_final = (conf_mask_image & (image_dominant | shared_mask))
                    conf_mask_pc_final = (conf_mask_pc & (pc_dominant | shared_mask))
                    conf_mask_image = conf_mask_image_final
                    conf_mask_pc = conf_mask_pc_final
                    
                    pseudo_targets_image = combined_targets
                    pseudo_targets_pc = combined_targets
                    
                    # 假设两个掩码对应相同的空间网格，但可能有不同的选中模式
                    # 将掩码重塑为相同维度（如果尚未对齐）
                    conf_mask_image_aligned = F.interpolate(conf_mask_image.float().unsqueeze(0).unsqueeze(0), 
                                                        size=pseudo_targets_pc.shape[-2:], 
                                                        mode='nearest').squeeze() > 0.5
                    conf_mask_pc_aligned = F.interpolate(conf_mask_pc.float().unsqueeze(0).unsqueeze(0), 
                                                        size=pseudo_targets_image.shape[-2:], 
                                                        mode='nearest').squeeze() > 0.5
                    # 计算共享区域
                    shared_mask = conf_mask_image_aligned & conf_mask_pc_aligned

                    if conf_mask_image.sum() > 0:
                        pseudolabel_agreement_loss = (
                            (pseudo_targets_image[shared_mask] != pseudo_targets_pc[shared_mask])
                            .float()
                            .sum()  # 计算不一致的样本数
                            / pseudo_targets_image[shared_mask].shape[0]
                            # / (conf_mask_image.sum() + conf_mask_pc.sum()).float()  # 除以两个掩码的总样本数 TODO
                        )
                        # pseudolabel_agreement_loss = (
                        #     (pseudo_targets_image[conf_mask_image] != pseudo_targets_pc[conf_mask_pc])
                        #     .float()
                        #     .mean()
                        # )
                    else:
                        pseudolabel_agreement_loss = torch.tensor(0.0, device=device)

            else:
                conf_mask_image = score_image > train_config['conf_threshold_image']
                conf_mask_pc = score_pc > train_config['conf_threshold_pc']

            if conf_mask_image.sum() > 0:
                pseudo_label_acc_image = (
                    (pseudo_targets_image[conf_mask_image] == targets[conf_mask_image])
                    .float()
                    .mean()
                    .item()
                )
            else:
                pseudo_label_acc_image = 0.0

            if conf_mask_pc.sum() > 0:
                if train_config['from_scratch']:
                    pseudo_label_acc_pc = (
                        (pseudo_targets_image[conf_mask_image] == targets[conf_mask_image])
                        .float()
                        .mean()
                        .item()
                    )
                else:
                    pseudo_label_acc_pc = (
                        (pseudo_targets_pc[conf_mask_pc] == targets[conf_mask_pc])
                        .float()
                        .mean()
                        .item()
                    )
            else:
                pseudo_label_acc_pc = 0.0

            conf_ratio_image = conf_mask_image.float().sum() / conf_mask_image.size(0)
            conf_ratio_pc = conf_mask_pc.float().sum() / conf_mask_pc.size(0)

            metric_logger.update(conf_ratio_image=conf_ratio_image)
            metric_logger.update(pseudo_label_acc_image=pseudo_label_acc_image)
            metric_logger.update(conf_ratio_pc=conf_ratio_pc)
            metric_logger.update(pseudo_label_acc_pc=pseudo_label_acc_pc)
        # --------------------------------------------------------------------------



        with amp_autocast():
            # 使用自动混合精度训练
            if args.mask:
                # 如果使用掩码,模型会返回更多的输出,包括:
                # - logits_image/pc: 图像和点云分支的预测logits 
                # - loss_mim_image/pc: 掩码图像建模损失
                # - loss_align_image/pc: 特征对齐损失
                # - pc_image_align_logits/image_pc_align_logits: 跨模态对齐的logits
                logits_image, logits_pc, loss_mim_image, loss_mim_pc, loss_align_image, loss_align_pc, pc_image_align_logits, image_pc_align_logits = model(images_strong, pc_strong, Mask=mask)
            else:
                # 不使用掩码时只返回预测logits
                logits_image, logits_pc = model(images_strong, pc_strong)

            # 计算自训练损失
            if train_config['trans_pcl_img']:
                
                # 如果启用点云到图像的迁移,使用点云的置信度掩码和伪标签来监督图像分支
                if conf_mask_image.sum() == 0:
                    skipped_steps += 1
                    print("logits_image[conf_mask_pc]:", logits_image[conf_mask_pc])
                    print("conf_mask_pc sum:", conf_mask_pc.sum().item())
                    print("logits_image range:", logits_image.min().item(), logits_image.max().item())
                    print(f"[WARNING] Step {step}: No confident image samples, skipping image loss")
                    loss_st_image = torch.tensor(0.0, device=device)
                else:
                    loss_st_image = F.cross_entropy(logits_image[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])
            else:
                
                # 否则使用图像自身的置信度掩码和伪标签
                # loss_st_image = F.cross_entropy(logits_image[conf_mask_image], pseudo_targets_image[conf_mask_image])
                # 添加
                if conf_mask_image.sum() == 0:
                    skipped_steps += 1
                    print("logits_image[conf_mask_image]:", logits_image[conf_mask_image])
                    print("conf_mask_image sum:", conf_mask_image.sum().item())
                    print("logits_image range:", logits_image.min().item(), logits_image.max().item())
                    print(f"[WARNING] Step {step}: No confident image samples, skipping image loss")
                    loss_st_image = torch.tensor(0.0, device=device)
                else:
                    loss_st_image = F.cross_entropy(logits_image[conf_mask_image], pseudo_targets_image[conf_mask_image])
                # 


            # 计算点云分支的自训练损失
            if train_config['from_scratch']:
                
                # 从头训练时使用图像的置信度掩码和伪标签
                if conf_mask_pc.sum() == 0:
                    skipped_steps += 1
                    print("logits_pc[conf_mask_pc]:", logits_pc[conf_mask_image])
                    print("conf_mask_image sum:", conf_mask_image.sum().item())
                    print("logits_pc range:", logits_pc.min().item(), logits_pc.max().item())
                    print(f"[WARNING] Step {step}: No confident pc samples, skipping pc loss")
                    loss_st_pc = torch.tensor(0.0, device=device)
                else:
                    loss_st_pc = F.cross_entropy(logits_pc[conf_mask_image], pseudo_targets_image[conf_mask_image])
            elif train_config['trans_img_pcl']:
                
                # 如果启用图像到点云的迁移,使用图像的置信度掩码和伪标签
                if conf_mask_pc.sum() == 0:
                    skipped_steps += 1
                    print("logits_pc[conf_mask_pc]:", logits_pc[conf_mask_image])
                    print("conf_mask_image sum:", conf_mask_image.sum().item())
                    print("logits_pc range:", logits_pc.min().item(), logits_pc.max().item())
                    print(f"[WARNING] Step {step}: No confident pc samples, skipping pc loss")
                    loss_st_pc = torch.tensor(0.0, device=device)
                else:
                    loss_st_pc = F.cross_entropy(logits_pc[conf_mask_image], pseudo_targets_image[conf_mask_image])
                
            else:
                # 否则使用点云自身的置信度掩码和伪标签
                # loss_st_pc = F.cross_entropy(logits_pc[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])
                # 添加
                if conf_mask_pc.sum() == 0:
                    skipped_steps += 1
                    print("logits_pc[conf_mask_pc]:", logits_pc[conf_mask_pc])
                    print("conf_mask_pc sum:", conf_mask_pc.sum().item())
                    print("logits_pc range:", logits_pc.min().item(), logits_pc.max().item())
                    print(f"[WARNING] Step {step}: No confident pc samples, skipping pc loss")
                    loss_st_pc = torch.tensor(0.0, device=device)
                else:
                    loss_st_pc = F.cross_entropy(logits_pc[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])
                #

            # fairness regularization
            # 计算图像分支的公平性正则化损失
            # 首先计算图像分支的softmax概率
            probs = F.softmax(logits_image,dim=-1)
            # 收集所有GPU上的概率
            if args.distributed:
                probs_all = all_gather_with_grad(probs)
            else:
                probs_all = probs
            # 计算所有GPU上的平均预测概率
            probs_batch_avg_image = probs_all.mean(0) 

            # 计算点云分支的公平性正则化损失
            # 计算点云分支的softmax概率
            probs = F.softmax(logits_pc, dim=-1)
            # 收集所有GPU上的概率
            if args.distributed:
                probs_all = all_gather_with_grad(probs)
            else:
                probs_all = probs
            # 计算所有GPU上的平均预测概率
            probs_batch_avg_pc = probs_all.mean(0)  

            # 计算图像分支的公平性损失
            probs_avg = probs_batch_avg_image
            loss_fair_image = -(torch.log(probs_avg)).mean()
            # 计算点云分支的公平性损失
            probs_avg = probs_batch_avg_pc
            loss_fair_pc = -(torch.log(probs_avg)).mean()

            # 如果使用掩码,计算额外的对齐损失
            if args.mask:
                # 创建对角矩阵作为标签
                labels = torch.eye(pc_image_align_logits.shape[0]).cuda()
                # 计算点云到图像的对齐损失
                loss_pc_image_align = F.cross_entropy(pc_image_align_logits, labels)
                # 计算图像到点云的对齐损失
                loss_image_pc_align = F.cross_entropy(image_pc_align_logits, labels)
                # 计算全局-局部特征对齐损失
                loss_align_image = torch.mean(loss_align_image)
                loss_align_pc = torch.mean(loss_align_pc)

                # 根据训练配置计算总损失
                if train_config['only_image']:
                    # 仅使用图像分支的损失
                    loss = loss_st_image + train_config['w_fair_image'] * loss_fair_image + train_config['w_mim_image']*loss_mim_image + train_config['w_align_image'] * loss_align_image
                elif train_config['only_pc']:
                    # 仅使用点云分支的损失
                    loss= loss_st_pc + train_config['w_fair_pc'] * loss_fair_pc + train_config['w_mim_pc']*loss_mim_pc + train_config['w_align_pc'] * loss_align_pc
                else:
                    # 使用两个分支的组合损失
                    loss = loss_st_image + train_config['w_fair_image'] * loss_fair_image + train_config['w_mim_image']*loss_mim_image + train_config['w_align_image'] * loss_align_image + args.pc_loss_weight * (loss_st_pc + train_config['w_fair_pc'] * loss_fair_pc + train_config['w_mim_pc']*loss_mim_pc +train_config['w_align_pc'] * loss_align_pc)

                    # 添加可选的额外损失项
                    if train_config['image_pc_align']:
                        # 添加图像-点云对齐损失
                        loss = loss + train_config['w_image_pc_align']*loss_pc_image_align + train_config['w_image_pc_align']*loss_image_pc_align
                    if train_config['pseudolabel_agreement_loss']:
                        # 添加伪标签一致性损失
                        loss = loss + train_config['w_pseudo_agree'] * pseudolabel_agreement_loss
                    if train_config['entropy_image']:
                        # 添加图像熵损失
                        loss = loss + loss_entropy_image
                    if train_config['entropy_pc']:
                        # 添加点云熵损失
                        loss = loss + loss_entropy_pc

            else:
                # 不使用掩码时的损失计算
                if train_config['only_image']:
                    # 仅使用图像分支损失
                    loss = loss_st_image + train_config['w_fair_image'] * loss_fair_image
                elif train_config['only_pc']:
                    # 仅使用点云分支损失
                    loss = loss_st_pc + train_config['w_fair_pc'] * loss_fair_pc
                else:
                    # 使用两个分支的组合损失
                    loss = loss_st_image + loss_st_pc + train_config['w_fair_image'] * loss_fair_image + train_config['w_fair_pc'] * loss_fair_pc

        loss_value = loss.item()
        
        # print("Debug - loss_st_image:", loss_st_image.isnan().item())
        # print("Debug - loss_st_pc:", loss_st_pc.isnan().item())
        # print("Debug - loss_fair_image:", loss_fair_image.isnan().item())
        # print("Debug - loss_fair_pc:", loss_fair_pc.isnan().item())
        # print("Debug - loss_mim_image:", loss_mim_image.isnan().item())
        # print("Debug - loss_mim_pc:", loss_mim_pc.isnan().item())
        # print("Debug - loss_align_image:", loss_align_image.isnan().item())
        # print("Debug - loss_align_pc:", loss_align_pc.isnan().item())
        # print("Debug - loss_pc_image_align:", loss_pc_image_align.isnan().item())
        # print("Debug - loss_image_pc_align:", loss_image_pc_align.isnan().item())
        # print("Debug - loss_entropy_image:", loss_entropy_image.isnan().item())
        # print("Debug - loss_entropy_pc:", loss_entropy_pc.isnan().item())
        if loss_st_image.isnan().item():
            print("Debug - loss_st_image: nan")
        if loss_st_pc.isnan().item():
            print("Debug - loss_st_pc: nan")
        if loss_fair_image.isnan().item():
            print("Debug - loss_fair_image: nan")
        if loss_fair_pc.isnan().item():
            print("Debug - loss_fair_pc: nan")
        if loss_mim_image.isnan().item():
            print("Debug - loss_mim_image: nan")
        if loss_mim_pc.isnan().item():
            print("Debug - loss_mim_pc: nan")
        if loss_align_image.isnan().item():
            print("Debug - loss_align_image: nan")
        if loss_align_pc.isnan().item():
            print("Debug - loss_align_pc: nan")
        if loss_pc_image_align.isnan().item():
            print("Debug - loss_pc_image_align: nan")
        if loss_image_pc_align.isnan().item():
            print("Debug - loss_image_pc_align: nan")
        if loss_entropy_image.isnan().item():
            print("Debug - loss_entropy_image: nan")
        if loss_entropy_pc.isnan().item():
            print("Debug - loss_entropy_pc: nan")

        
        # 添加
        if torch.isnan(loss):
            print(f"[ERROR] Epoch {epoch}, Step {step}: Loss is NaN!")
            sys.exit(1)
        #

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
            loss.backward(create_graph=False)
            optimizer.step()

        model_ema.update(model)
        torch.cuda.synchronize()

        metric_logger.update(loss_st_image=loss_st_image.item())
        metric_logger.update(loss_fair_image=loss_fair_image.item())
        metric_logger.update(loss_st_pc=loss_st_pc.item())
        metric_logger.update(loss_fair_pc=loss_fair_pc.item())
        metric_logger.update(loss_entropy_image=loss_entropy_image.item())
        metric_logger.update(loss_entropy_pc=loss_entropy_pc.item())
        if train_config['combined_pseudolabels']:
            metric_logger.update(loss_pseudolabel_agreement=pseudolabel_agreement_loss.item())

        if args.mask:
            metric_logger.update(loss_pc_image_align=loss_pc_image_align.item())
            metric_logger.update(loss_image_pc_align=loss_image_pc_align.item())
            metric_logger.update(loss_mim_image=loss_mim_image.item())
            metric_logger.update(loss_align_image=loss_align_image.item())
            metric_logger.update(loss_mim_pc=loss_mim_pc.item())
            metric_logger.update(loss_align_pc=loss_align_pc.item())

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

            if args.mask:
                log_writer.update(loss_mim_image=loss_mim_image.item(), head="train")
                log_writer.update(loss_align_image=loss_align_image.item(), head="train")
                log_writer.update(loss_mim_pc=loss_mim_pc.item(), head="train")
                log_writer.update(loss_align_pc=loss_align_pc.item(), head="train")

            log_writer.update(conf_ratio_image=conf_ratio_image, head="train")
            log_writer.update(pseudo_label_acc_image=pseudo_label_acc_image, head="train")
            log_writer.update(conf_ratio_pc=conf_ratio_pc, head="train")
            log_writer.update(pseudo_label_acc_pc=pseudo_label_acc_pc, head="train")

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
    print("skipped_steps:", skipped_steps)
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
        pcs = batch[0][1].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        # images = batch[0].to(device, non_blocking=True)
        # pcs = batch[1].to(device, non_blocking=True)
        # target = batch[-1].to(device, non_blocking=True)

        # compute output
        output = model(images, pcs)

        acc_image = accuracy(output[0], target)[0]
        acc_pc = accuracy(output[1], target)[0]
        metric_logger.meters['acc1_image'].update(acc_image.item(), n=images.shape[0])
        metric_logger.meters['acc1_pc'].update(acc_pc.item(), n=images.shape[0])

        if model_ema is not None:
            ema_output = model_ema.ema(images, pcs)

            ema_acc1_image = accuracy(ema_output[0], target)[0]
            ema_acc1_pc = accuracy(ema_output[1], target)[0]
            metric_logger.meters['ema_acc1_image'].update(ema_acc1_image.item(), n=images.shape[0])
            metric_logger.meters['ema_acc1_pc'].update(ema_acc1_pc.item(), n=images.shape[0])

    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_image))
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1_pc))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

