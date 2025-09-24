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
        depth_embed = outputs.get('depth_embed', None)  # 获取深度嵌入，如果不存在则为None
        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        if depth_embed is not None:
            depth_embed = F.normalize(depth_embed, dim=-1, p=2)

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
        
        # 添加深度相关的损失
        if depth_embed is not None:
            print(f"depth_embed in loss: {depth_embed.shape}, norm: {depth_embed.norm()}")
            logits_per_pc_depth = logit_scale * pc_embed @ depth_embed_all.t()
            logits_per_depth_pc = logit_scale * depth_embed @ pc_embed_all.t()
            logits_per_text_depth = logit_scale * text_embed @ depth_embed_all.t()
            logits_per_depth_text = logit_scale * depth_embed @ text_embed_all.t()
            
            loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
                    F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
                    (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2 + \
                    (F.cross_entropy(logits_per_pc_depth, self.labels) + F.cross_entropy(logits_per_depth_pc, self.labels)) / 2 + \
                    (F.cross_entropy(logits_per_text_depth, self.labels) + F.cross_entropy(logits_per_depth_text, self.labels)) / 2
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

        return {'loss': loss, 'ulip_loss': loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc, 'ulip_pc_depth_acc': pc_depth_acc}
