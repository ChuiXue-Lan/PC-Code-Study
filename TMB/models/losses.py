'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        depth_embed = outputs.get('depth_embed', None) 
        logit_scale = outputs['logit_scale']
        logit_scale_depth = outputs.get('logit_scale_depth', logit_scale)
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size
            
        # 在F.normalize前先拷贝一份 TODO
        pc_raw    = pc_embed.detach()
        text_raw  = text_embed.detach()
        image_raw = image_embed.detach()
        depth_raw = depth_embed.detach() if depth_embed is not None else None

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        if depth_embed is not None:
            depth_embed = F.normalize(depth_embed, dim=-1, p=2)
        
        # TODO
        if self.training and (self.last_local_batch_size is not None):
            pc_norm_mean   = pc_raw.norm(dim=-1).mean().item()
            img_norm_mean  = image_raw.norm(dim=-1).mean().item()
            txt_norm_mean  = text_raw.norm(dim=-1).mean().item()
            if depth_raw is not None:
                dep_norm_mean = depth_raw.norm(dim=-1).mean().item()
                # 可选：打印或用logger记录
                print(f"[NORM] pc:{pc_norm_mean:.2f} img:{img_norm_mean:.2f} txt:{txt_norm_mean:.2f} dep:{dep_norm_mean:.2f}")

        # gather features from all GPUs
        if depth_embed is not None:
            pc_embed_all, text_embed_all, image_embed_all, depth_embed_all = \
                utils.all_gather_batch([pc_embed, text_embed, image_embed, depth_embed])
        else:
            pc_embed_all, text_embed_all, image_embed_all = \
                utils.all_gather_batch([pc_embed, text_embed, image_embed])

        # cosine similarity as logits
        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()
        

        # loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
        #         F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
        #         (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2
        # 添加深度相关的损失
        if depth_embed is not None:
            # print(f"depth_embed in loss: {depth_embed.shape}, norm: {depth_embed.norm()}")
            logits_per_pc_depth = logit_scale_depth * pc_embed @ depth_embed_all.t()
            logits_per_depth_pc = logit_scale_depth * depth_embed @ pc_embed_all.t()
            logits_per_text_depth = logit_scale_depth * text_embed @ depth_embed_all.t()
            logits_per_depth_text = logit_scale_depth * depth_embed @ text_embed_all.t()
            logits_per_image_depth = logit_scale_depth * image_embed @ depth_embed_all.t()
            logits_per_depth_image = logit_scale_depth * depth_embed @ image_embed_all.t()
            
            depth_text_cos = (F.normalize(depth_embed,dim=-1)*F.normalize(text_embed,dim=-1)).sum(-1).mean()
            pc_text_cos = (F.normalize(pc_embed,dim=-1)*F.normalize(text_embed,dim=-1)).sum(-1).mean()
            image_text_cos = (F.normalize(image_embed,dim=-1)*F.normalize(text_embed,dim=-1)).sum(-1).mean()
            print(f"debug -- depth_text_cos: {depth_text_cos}, pc_text_cos: {pc_text_cos}, image_text_cos: {image_text_cos}")
            dep_var = depth_raw.var(dim=0).mean().item()
            print(f"debug -- dep_var: {dep_var}")
            
            loss = (F.cross_entropy(logits_per_pc_text, self.labels) + F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
                    (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2 + \
                    (F.cross_entropy(logits_per_pc_depth, self.labels) + F.cross_entropy(logits_per_depth_pc, self.labels)) / 2 * 0.5 +  \
                    (F.cross_entropy(logits_per_text_depth, self.labels) + F.cross_entropy(logits_per_depth_text, self.labels)) / 2 * 2
            
            # loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
            #         F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
            #         (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2 + \
            #         (F.cross_entropy(logits_per_pc_depth, self.labels) + F.cross_entropy(logits_per_depth_pc, self.labels)) / 2 + \
            #         (F.cross_entropy(logits_per_text_depth, self.labels) + F.cross_entropy(logits_per_depth_text, self.labels)) / 2 + \
            #         (F.cross_entropy(logits_per_image_depth, self.labels) + F.cross_entropy(logits_per_depth_image, self.labels)) / 2
        else:
            print("depth_embed is None in loss")
            loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
                    F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
                    (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_image_acc = 100 * correct / local_batch_size
            
            if depth_embed is not None:
                pred = torch.argmax(logits_per_pc_depth, dim=-1)
                correct = pred.eq(self.labels).sum()
                pc_depth_acc = 100 * correct / local_batch_size
            else:
                pc_depth_acc = 0.0

        # return {'loss': loss, 'ulip_loss': loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc}
        return {'loss': loss, 'ulip_loss': loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc, 'ulip_pc_depth_acc': pc_depth_acc}
