
def make_metric_logger(metric_logger,loss_st_image, loss_st_pc, loss_st_depth, loss_fair_image, loss_fair_pc, loss_fair_depth, loss_depth_image_align, loss_image_depth_align, \
    loss_depth_pc_align, loss_pc_depth_align, loss_align_image, loss_align_pc, loss_align_depth, loss_mim_image, loss_mim_pc, loss_mim_depth, loss_entropy_image, loss_entropy_pc, \
        loss_entropy_depth, pseudolabel_agreement_loss, optimizer, train_config, args, loss_pc_image_align = None, loss_image_pc_align = None):
    if train_config['pairwise_alignment']:
        metric_logger.update(loss_st_image=loss_st_image.item())
        metric_logger.update(loss_fair_image=loss_fair_image.item())
        
        metric_logger.update(loss_st_pc=loss_st_pc.item())
        metric_logger.update(loss_fair_pc=loss_fair_pc.item())
        
        metric_logger.update(loss_st_depth=loss_st_depth.item())
        metric_logger.update(loss_fair_depth=loss_fair_depth.item())
        
        metric_logger.update(loss_entropy_image=loss_entropy_image.item())
        metric_logger.update(loss_entropy_pc=loss_entropy_pc.item())
        metric_logger.update(loss_entropy_depth=loss_entropy_depth.item())
        
        if train_config['combined_pseudolabels']:
            metric_logger.update(loss_pseudolabel_agreement=pseudolabel_agreement_loss.item())

        if args.mask:
            metric_logger.update(loss_align_image=loss_align_image.item())
            metric_logger.update(loss_align_pc=loss_align_pc.item())
            metric_logger.update(loss_align_depth=loss_align_depth.item())
            
            metric_logger.update(loss_mim_image=loss_mim_image.item())
            metric_logger.update(loss_mim_pc=loss_mim_pc.item())
            metric_logger.update(loss_mim_depth=loss_mim_depth.item())
            
            metric_logger.update(loss_pc_image_align=loss_pc_image_align.item())
            metric_logger.update(loss_image_pc_align=loss_image_pc_align.item())
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
    elif train_config['depth_modality_center']:
        metric_logger.update(loss_st_image=loss_st_image.item())
        metric_logger.update(loss_fair_image=loss_fair_image.item())
        
        metric_logger.update(loss_st_pc=loss_st_pc.item())
        metric_logger.update(loss_fair_pc=loss_fair_pc.item())
        
        metric_logger.update(loss_st_depth=loss_st_depth.item())
        metric_logger.update(loss_fair_depth=loss_fair_depth.item())
        
        metric_logger.update(loss_entropy_image=loss_entropy_image.item())
        metric_logger.update(loss_entropy_pc=loss_entropy_pc.item())
        metric_logger.update(loss_entropy_depth=loss_entropy_depth.item())
        
        if train_config['combined_pseudolabels']:
            metric_logger.update(loss_pseudolabel_agreement=pseudolabel_agreement_loss.item())

        if args.mask:
            metric_logger.update(loss_align_image=loss_align_image.item())
            metric_logger.update(loss_align_pc=loss_align_pc.item())
            metric_logger.update(loss_align_depth=loss_align_depth.item())
            
            metric_logger.update(loss_mim_image=loss_mim_image.item())
            metric_logger.update(loss_mim_pc=loss_mim_pc.item())
            metric_logger.update(loss_mim_depth=loss_mim_depth.item())
            
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
        
        return max_lr, min_lr

def make_log(log_writer, loss_st_image, loss_st_pc, loss_st_depth, loss_fair_image, loss_fair_pc, loss_fair_depth, loss_align_image, loss_align_pc, loss_align_depth, \
    loss_mim_image, loss_mim_pc, loss_mim_depth, max_lr, min_lr, args):
    if log_writer is not None:
        log_writer.update(loss_st_image=loss_st_image.item(), head="train")
        log_writer.update(loss_fair_image=loss_fair_image.item(), head="train")
        
        log_writer.update(loss_st_pc=loss_st_pc.item(), head="train")
        log_writer.update(loss_fair_pc=loss_fair_pc.item(), head="train")
        
        log_writer.update(loss_st_depth=loss_st_depth.item(), head="train")
        log_writer.update(loss_fair_depth=loss_fair_depth.item(), head="train")

        if args.mask:
            log_writer.update(loss_mim_image=loss_mim_image.item(), head="train")
            log_writer.update(loss_align_image=loss_align_image.item(), head="train")
            
            log_writer.update(loss_mim_pc=loss_mim_pc.item(), head="train")
            log_writer.update(loss_align_pc=loss_align_pc.item(), head="train")
            
            log_writer.update(loss_mim_depth=loss_mim_depth.item(), head="train")
            log_writer.update(loss_align_depth=loss_align_depth.item(), head="train")

        # log_writer.update(conf_ratio_image=conf_ratio_image, head="train")
        # log_writer.update(pseudo_label_acc_image=pseudo_label_acc_image, head="train")
        # log_writer.update(conf_ratio_pc=conf_ratio_pc, head="train")
        # log_writer.update(pseudo_label_acc_pc=pseudo_label_acc_pc, head="train")

        log_writer.update(lr=max_lr, head="opt")
        log_writer.update(min_lr=min_lr, head="opt")
        log_writer.set_step()