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
from models import losses
from collections import OrderedDict


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
    train_config["base_threshold_combined"] = adjusted_threshold_combined

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
        if not args.ulip:
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
                        conf_mask_image = combined_scores > train_config['conf_threshold_combined']
                        conf_mask_pc = conf_mask_image
                        pseudo_targets_image = combined_targets
                        pseudo_targets_pc = combined_targets

                        if conf_mask_image.sum() > 0:
                            pseudolabel_agreement_loss = (
                                (pseudo_targets_image[conf_mask_image] != pseudo_targets_pc[conf_mask_pc])
                                .float()
                                .sum()  # 计算不一致的样本数
                                / pseudo_targets_image[conf_mask_image].shape[0]
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
        if args.ulip:
            batch_time = AverageMeter('Time', ':6.2f')
            data_time = AverageMeter('Data', ':6.2f')
            mem = AverageMeter('Mem (GB)', ':6.1f') 
            metric_names = model.get_metric_names(args.model)
            iters_per_epoch = len(data_loader) // args.update_freq
            metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
            progress = ProgressMeter(
                iters_per_epoch,
                [batch_time, data_time, mem, *metrics.values()],
                prefix="Epoch: [{}]".format(epoch))
            
            for data_iter, inputs in enumerate(data_loader):
                optim_iter = data_iter // args.update_freq

                # measure data loading time
                data_time.update(time.time() - end)

                # update weight decay and learning rate according to their schedule
                it = iters_per_epoch * epoch + optim_iter  # global training iteration
                for k, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = lr_schedule_values[it]

                pc = inputs[3]
                texts = inputs[2]

                image = inputs[4]
                inputs = [pc, texts, image]

                inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
                with amp_autocast(enabled=not args.disable_amp):
                    # define loss function (criterion) and optimizer
                    criterion = model.get_loss(args).cuda(args.gpu)
                    
                    outputs = model(*inputs)
                    loss_dict = criterion(outputs)
                    loss = loss_dict['loss']
                    loss /= args.update_freq

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()))
                    sys.exit(1)

                loss_scaler.scale(loss).backward()
                
                if (data_iter + 1) % args.update_freq != 0:
                    continue

                # compute gradient and do SGD step
                loss_scaler.step(optimizer)
                loss_scaler.update()
                model.zero_grad(set_to_none=True)

                # clamp logit scale to [0, 100]

                utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
                logit_scale = utils.get_model(model).logit_scale.exp().item()

                for k in loss_dict:
                    metrics[k].update(loss_dict[k].item(), args.batch_size)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                mem.update(torch.cuda.max_memory_allocated() // 1e9)

                if optim_iter % args.print_freq == 0:
                    if utils.is_main_process() and args.wandb:
                        wandb.log({**{k: v.item() for k, v in loss_dict.items()},
                                'scaler': loss_scaler.get_scale(),
                                'logit': logit_scale})
                    progress.display(optim_iter)
                
            progress.synchronize()
            return {**{k: v.avg for k, v in metrics.items()},
                'lr': optimizer.param_groups[0]['lr'],
                'logit_scale': logit_scale}


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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """计算指定k值下的top-k准确率
    
    Args:
        output: 模型输出的预测分数
        target: 真实标签
        topk: 需要计算的top-k值列表，默认为(1,)
        
    Returns:
        res: 每个top-k值对应的准确率列表
        correct: 预测正确的布尔值矩阵
    """
    with torch.no_grad():
        # 获取最大的k值
        maxk = max(topk)
        batch_size = target.size(0)

        # 获取每个样本的top-k预测结果
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置预测结果矩阵
        # 将预测结果与真实标签比较
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        # 计算每个top-k值的准确率
        for k in topk:
            # 统计前k个预测中正确的数量
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # 计算准确率百分比
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct