import torch
import torch.nn.functional as F
from utils.utils import all_gather_with_grad

# 两两对齐损失 TODO:contrastive_loss
def pairwise_alignment_loss():
    # loss_img_pc = contrastive_loss(img_feat, pc_feat)
    # loss_img_depth = contrastive_loss(img_feat, depth_feat)
    # loss_pc_depth = contrastive_loss(pc_feat, depth_feat)
    # pairwise_alignment_loss = loss_img_pc + loss_img_depth + loss_pc_depth
    pairwise_alignment_loss = 0
    return pairwise_alignment_loss

# 中心对齐损失 TODO
def modality_center_loss(img_feat, pc_feat, depth_feat):
    # feat: (B, D)
    center = (img_feat + pc_feat + depth_feat) / 3.0
    loss = (
        (img_feat - center).pow(2).sum(dim=1) +
        (pc_feat - center).pow(2).sum(dim=1) +
        (depth_feat - center).pow(2).sum(dim=1)
    )
    return loss.mean()

# 三元组对齐损失 TODO
def triplet_loss(anchor, positive, negative, margin=0.2):
    d_pos = F.pairwise_distance(anchor, positive, p=2)
    d_neg = F.pairwise_distance(anchor, negative, p=2)
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()

# 计算自训练损失
def self_supervised_loss(logits_image, logits_pc, logits_depth, pseudo_targets_image, pseudo_targets_pc,\
    pseudo_targets_depth, conf_mask_all, conf_mask_pc, conf_mask_image, conf_mask_depth, train_config):
    
    loss_st_image = F.cross_entropy(logits_image[conf_mask_all], pseudo_targets_image[conf_mask_all])
    loss_st_pc = F.cross_entropy(logits_pc[conf_mask_all], pseudo_targets_pc[conf_mask_all])
    loss_st_depth = F.cross_entropy(logits_depth[conf_mask_all], pseudo_targets_depth[conf_mask_all])
    
    if train_config['pairwise_alignment']:
        # 计算图像分支的自训练损失
        loss_st_image = 0
        if train_config['trans_pcl_img']:
            # 如果启用点云到图像的迁移,使用点云的置信度掩码和伪标签来监督图像分支
            loss_st_image += F.cross_entropy(logits_image[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])
        if train_config['trans_depth_img']:
            loss_st_image += F.cross_entropy(logits_image[conf_mask_depth], pseudo_targets_depth[conf_mask_depth])
        if not train_config['trans_pcl_img'] and not train_config['trans_depth_img']:
            # 否则使用图像自身的置信度掩码和伪标签
            loss_st_image += F.cross_entropy(logits_image[conf_mask_image], pseudo_targets_image[conf_mask_image])

        # 计算点云分支的自训练损失
        loss_st_pc = 0
        if train_config['from_scratch']:
            # 从头训练时使用图像的置信度掩码和伪标签
            loss_st_pc += F.cross_entropy(logits_pc[conf_mask_image], pseudo_targets_image[conf_mask_image])
        else:
            if train_config['trans_img_pcl']:
                loss_st_pc += F.cross_entropy(logits_pc[conf_mask_image], pseudo_targets_image[conf_mask_image])
            if train_config['trans_depth_pcl']:
                loss_st_pc += F.cross_entropy(logits_pc[conf_mask_depth], pseudo_targets_depth[conf_mask_depth])
            if not train_config['trans_img_pcl'] and not train_config['trans_depth_pcl']:
                loss_st_pc += F.cross_entropy(logits_pc[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])

        # 计算深度分支的自训练损失
        loss_st_depth = 0
        if train_config['trans_img_depth']:
            loss_st_depth += F.cross_entropy(logits_depth[conf_mask_image], pseudo_targets_image[conf_mask_image])
        if train_config['trans_pcl_depth']:
            loss_st_depth += F.cross_entropy(logits_depth[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])
        if not train_config['trans_img_depth'] and not train_config['trans_pcl_depth']:
            loss_st_depth = F.cross_entropy(logits_depth[conf_mask_depth], pseudo_targets_depth[conf_mask_depth])

    elif train_config['depth_modality_center']:
        # 图像分支自训练损失
        loss_st_image = 0
        if train_config['trans_depth_img']:
            loss_st_image += F.cross_entropy(logits_image[conf_mask_depth], pseudo_targets_depth[conf_mask_depth])
        if not train_config['trans_depth_img']:
            loss_st_image += F.cross_entropy(logits_image[conf_mask_image], pseudo_targets_image[conf_mask_image])

        # 点云分支自训练损失
        loss_st_pc = 0
        if train_config['trans_depth_pcl']:
            loss_st_pc += F.cross_entropy(logits_pc[conf_mask_depth], pseudo_targets_depth[conf_mask_depth])
        if not train_config['trans_depth_pcl']:
            loss_st_pc += F.cross_entropy(logits_pc[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])

        # 深度分支自训练损失
        loss_st_depth = 0
        if train_config['trans_img_depth']:
            loss_st_depth += F.cross_entropy(logits_depth[conf_mask_image], pseudo_targets_image[conf_mask_image])
        if train_config['trans_pcl_depth']:
            loss_st_depth += F.cross_entropy(logits_depth[conf_mask_pc], pseudo_targets_pc[conf_mask_pc])
        if not train_config['trans_img_depth'] and not train_config['trans_pcl_depth']:
            loss_st_depth += F.cross_entropy(logits_depth[conf_mask_depth], pseudo_targets_depth[conf_mask_depth])
        
    return loss_st_image, loss_st_pc, loss_st_depth

# 计算公平性损失
def fair_loss(logits_image, logits_pc, logits_depth):
    # 计算图像分支的softmax概率
    probs_image = F.softmax(logits_image,dim=-1)
    # 收集所有GPU上的概率
    probs_all_image = all_gather_with_grad(probs_image)
    # 计算所有GPU上的平均预测概率
    probs_batch_avg_image = probs_all_image.mean(0) 

    probs_pc = F.softmax(logits_pc, dim=-1)
    probs_all_pc = all_gather_with_grad(probs_pc)
    probs_batch_avg_pc = probs_all_pc.mean(0)  

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
    
    return loss_fair_image, loss_fair_pc, loss_fair_depth

# 计算使用掩码的对齐损失
def align_loss_with_mask(pc_image_align_logits, image_pc_align_logits, depth_image_align_logits, \
    image_depth_align_logits, depth_pc_align_logits, pc_depth_align_logits, train_config):
    
    if train_config['pairwise_alignment']:
        # 创建对角矩阵作为标签
        # image-pc 对齐
        labels_ip = torch.eye(image_pc_align_logits.shape[0]).cuda()
        labels_pi = torch.eye(pc_image_align_logits.shape[0]).cuda()
        # image-depth 对齐
        labels_id = torch.eye(image_depth_align_logits.shape[0]).cuda()
        labels_di = torch.eye(depth_image_align_logits.shape[0]).cuda()
        # pc-depth 对齐
        labels_pd = torch.eye(pc_depth_align_logits.shape[0]).cuda()
        labels_dp = torch.eye(depth_pc_align_logits.shape[0]).cuda()
        
        # 计算点云到图像的对齐损失
        loss_pc_image_align = F.cross_entropy(pc_image_align_logits, labels_pi)
        # 计算图像到点云的对齐损失
        loss_image_pc_align = F.cross_entropy(image_pc_align_logits, labels_ip)
        # 计算深度图到图像的对齐损失
        loss_depth_image_align = F.cross_entropy(depth_image_align_logits, labels_di)
        # 计算图像到深度图的对齐损失
        loss_image_depth_align = F.cross_entropy(image_depth_align_logits, labels_id)
        # 计算深度图到点云的对齐损失
        loss_depth_pc_align = F.cross_entropy(depth_pc_align_logits, labels_dp)
        # 计算点云到深度图的对齐损失
        loss_pc_depth_align = F.cross_entropy(pc_depth_align_logits, labels_pd)
        
        return loss_pc_image_align, loss_image_pc_align, loss_depth_image_align, loss_image_depth_align, loss_depth_pc_align, loss_pc_depth_align
        
    elif train_config['depth_modality_center']:
        # # depth-image 对齐
        # labels_di = torch.eye(depth_image_align_logits.shape[0]).cuda()
        # labels_id = torch.eye(image_depth_align_logits.shape[0]).cuda()
        # # depth-pc 对齐
        # labels_dp = torch.eye(depth_pc_align_logits.shape[0]).cuda()
        # labels_pd = torch.eye(pc_depth_align_logits.shape[0]).cuda()
        
        # loss_depth_image_align = F.cross_entropy(depth_image_align_logits, labels_di)
        # loss_image_depth_align = F.cross_entropy(image_depth_align_logits, labels_id)
        # loss_depth_pc_align = F.cross_entropy(depth_pc_align_logits, labels_dp)
        # loss_pc_depth_align = F.cross_entropy(pc_depth_align_logits, labels_pd)
        
        # depth-image 对齐
        labels_di = torch.arange(depth_image_align_logits.shape[0]).cuda()  # [0,1,2,...,N-1]
        labels_id = torch.arange(image_depth_align_logits.shape[0]).cuda()
        # depth-pc 对齐
        labels_dp = torch.arange(depth_pc_align_logits.shape[0]).cuda()
        labels_pd = torch.arange(pc_depth_align_logits.shape[0]).cuda()
        
        loss_depth_image_align = F.cross_entropy(depth_image_align_logits, labels_di)
        loss_image_depth_align = F.cross_entropy(image_depth_align_logits, labels_id)
        loss_depth_pc_align = F.cross_entropy(depth_pc_align_logits, labels_dp)
        loss_pc_depth_align = F.cross_entropy(pc_depth_align_logits, labels_pd)
        
        return loss_depth_image_align, loss_image_depth_align, loss_depth_pc_align, loss_pc_depth_align

# 计算全局-局部特征对齐损失
def global_local_alignment_loss(loss_align_image, loss_align_pc, loss_align_depth):
    loss_align_image = torch.mean(loss_align_image)
    loss_align_pc = torch.mean(loss_align_pc)
    loss_align_depth = torch.mean(loss_align_depth)
    return loss_align_image, loss_align_pc, loss_align_depth

def main_loss(logits_image, logits_pc, logits_depth, pseudo_targets_image, pseudo_targets_pc, pseudo_targets_depth, \
    conf_mask_all, conf_mask_pc, conf_mask_image, conf_mask_depth, pc_image_align_logits, image_pc_align_logits, \
    depth_image_align_logits, image_depth_align_logits, depth_pc_align_logits, pc_depth_align_logits, \
    loss_entropy_image, loss_entropy_pc, loss_entropy_depth, loss_align_image, loss_align_pc, loss_align_depth, train_config, args):
    
    # 计算自训练损失
    loss_st_image, loss_st_pc, loss_st_depth = self_supervised_loss(logits_image, logits_pc, logits_depth, \
        pseudo_targets_image, pseudo_targets_pc,pseudo_targets_depth, conf_mask_all, conf_mask_pc, \
        conf_mask_image, conf_mask_depth, train_config)
    
    # 计算公平性损失
    loss_fair_image, loss_fair_pc, loss_fair_depth = fair_loss(logits_image, logits_pc, logits_depth)
    
    # 计算使用掩码的对齐损失
    if train_config['pairwise_alignment']:
        loss_pc_image_align, loss_image_pc_align, loss_depth_image_align, loss_image_depth_align, loss_depth_pc_align, \
            loss_pc_depth_align = align_loss_with_mask(pc_image_align_logits, image_pc_align_logits, \
            depth_image_align_logits, image_depth_align_logits, depth_pc_align_logits, pc_depth_align_logits, train_config)
    
    elif train_config['depth_modality_center']:
        loss_depth_image_align, loss_image_depth_align, loss_depth_pc_align, loss_pc_depth_align =\
            align_loss_with_mask(pc_image_align_logits, image_pc_align_logits, depth_image_align_logits, \
            image_depth_align_logits, depth_pc_align_logits, pc_depth_align_logits, train_config)
    
    # 计算全局-局部特征对齐损失
    loss_align_image, loss_align_pc, loss_align_depth = global_local_alignment_loss(loss_align_image, loss_align_pc, loss_align_depth)
    
    loss = 0
    if args.mask:
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

            
            if train_config['pairwise_alignment']:
                # 添加图像-点云对齐损失
                if train_config['image_pc_align']:
                    loss = loss + train_config['w_image_pc_align']*loss_pc_image_align + train_config['w_image_pc_align']*loss_image_pc_align
                # 添加深度图-图像对齐损失
                if train_config['depth_image_align']:
                    loss = loss + train_config['w_depth_image_align']*loss_depth_image_align + train_config['w_depth_image_align']*loss_image_depth_align
                # 添加点云-深度图对齐损失
                if train_config['pc_depth_align']:
                    loss = loss + train_config['w_pc_depth_align']*loss_pc_depth_align + train_config['w_pc_depth_align']*loss_depth_pc_align
                
            elif train_config['depth_modality_center']:
                # 添加深度图-图像对齐损失
                if train_config['depth_image_align']:
                    loss = loss + train_config['w_depth_image_align']*loss_depth_image_align + train_config['w_depth_image_align']*loss_image_depth_align
                # 添加点云-深度图对齐损失
                if train_config['pc_depth_align']:
                    loss = loss + train_config['w_pc_depth_align']*loss_pc_depth_align + train_config['w_pc_depth_align']*loss_depth_pc_align
                
            
            if train_config['entropy_image']:
                    # 添加图像熵损失
                    loss = loss + loss_entropy_image
            if train_config['entropy_pc']:
                # 添加点云熵损失
                loss = loss + loss_entropy_pc
            if train_config['entropy_depth']:
                # 添加深度熵损失
                loss = loss + loss_entropy_depth
            
            # TODO:伪标签一致性损失
            # if train_config['pseudolabel_agreement_loss']:
                #     # 添加伪标签一致性损失
                #     loss = loss + train_config['w_pseudo_agree'] * pseudolabel_agreement_loss
    else:
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
            loss = loss_st_image + loss_st_pc + loss_st_depth + train_config['w_fair_image'] * loss_fair_image + \
                train_config['w_fair_pc'] * loss_fair_pc + train_config['w_fair_depth'] * loss_fair_depth
    
    if train_config['pairwise_alignment']:
        return loss, (loss_st_image, loss_st_pc, loss_st_depth, loss_fair_image, loss_fair_pc, loss_fair_depth, loss_pc_image_align, loss_image_pc_align, loss_depth_image_align, loss_image_depth_align, loss_depth_pc_align, loss_pc_depth_align, loss_align_image, loss_align_pc, loss_align_depth)
    elif train_config['depth_modality_center']:
        return loss, (loss_st_image, loss_st_pc, loss_st_depth, loss_fair_image, loss_fair_pc, loss_fair_depth, loss_depth_image_align, loss_image_depth_align, loss_depth_pc_align, loss_pc_depth_align, loss_align_image, loss_align_pc, loss_align_depth)

# 伪标签一致性损失
def pseudo_label_consistency_loss(pseudo_targets_image, pseudo_targets_pc, pseudo_targets_depth, conf_mask_image, \
    conf_mask_pc, conf_mask_depth, train_config):
    
    if train_config['pairwise_alignment']:
        # 计算三对模态的伪标签一致性损失（两两对齐）
        agree_ip = (pseudo_targets_image[conf_mask_image] != pseudo_targets_pc[conf_mask_pc]).float().mean()
        agree_id = (pseudo_targets_image[conf_mask_image] != pseudo_targets_depth[conf_mask_depth]).float().mean()
        agree_pd = (pseudo_targets_pc[conf_mask_pc] != pseudo_targets_depth[conf_mask_depth]).float().mean()
        pseudolabel_agreement_loss = (agree_ip + agree_id + agree_pd) / 3
    elif train_config['depth_modality_center']:
        # 只计算深度与其它模态的伪标签一致性
        agree_id = (pseudo_targets_image[conf_mask_image] != pseudo_targets_depth[conf_mask_depth]).float().mean()
        agree_pd = (pseudo_targets_pc[conf_mask_pc] != pseudo_targets_depth[conf_mask_depth]).float().mean()
        pseudolabel_agreement_loss = (agree_id + agree_pd) / 2
        
    return pseudolabel_agreement_loss