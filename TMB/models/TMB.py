# Modified from github.com/openai/CLIP
from collections import OrderedDict

import timm
from torch import nn
from models.pointnet2.pointnet2 import Pointnet2_Ssg
from data.dataset_3d import *

from models import losses
from torch.nn.parameter import Parameter
from easydict import EasyDict
from models.pointbert.point_encoder import MaskTransformerMUST, MaskTransformerMUST_withdvaeloss
from models.must_clip.model import VisionTransformer_MIM
from models.DepthFeatureEncoder import DepthFeatureEncoder
from models.UResnetEncoder import UResnetEncoder
from models.DPTDepthEncoder import DPTDepthEncoder
# from models.DepthFeatureEncoder import DepthFeatureEncoderWithMask

from typing import Any, Union, List
import torch
import torch.nn as nn
import json
from tqdm import tqdm

import hashlib
import os
import urllib
import warnings

import torch.nn.functional as F

# TODO
import numpy as np

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, download_root: str = None, mask: bool = False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict(), mask=mask).to(device)
        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model


def load_state_dict(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                    jit: bool = False, download_root: str = None, mask: bool = False):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # try:
    #     # loading JIT archive
    #     model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
    #     state_dict = None
    #
    # except RuntimeError:
    #     # loading saved state dict
    #     if jit:
    #         warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
    #         jit = False
    state_dict = torch.load(model_path, map_location="cpu")
    return state_dict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    
def get_activation(name, activation_dict):
    def hook(model, input, output):
        activation_dict[name] = output
    return hook

def register_hook(DPTDepthEncoder, use_vit_only=False):
    pretrained = DPTDepthEncoder.pretrained
    backbone = DPTDepthEncoder.backbone
    hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }
    hooks = hooks[backbone]
    
    if use_vit_only == True:
        pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1", pretrained.activations))
        pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2", pretrained.activations))
    else:
        pretrained.model.patch_embed.backbone.stages[0].register_forward_hook(get_activation("1", pretrained.activations))
        pretrained.model.patch_embed.backbone.stages[1].register_forward_hook(get_activation("2", pretrained.activations))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3", pretrained.activations))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4", pretrained.activations))
    return 0

class TMB(nn.Module):
    def __init__(self, args, config, point_encoder, depth_encoder, **kwargs):
        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model
        self.classes = kwargs.classes
        self.templates = kwargs.templates
        self.tokenizer = kwargs.tokenizer

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, kwargs.transformer_width))
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(torch.empty(kwargs.vision_width, kwargs.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(kwargs.transformer_width, kwargs.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # TODO
        # self.logit_scale_depth = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.logit_scale_depth = nn.Parameter(torch.ones([]) * np.log(1/0.03)) # ≈3.5，exp≈33

        self.initialize_parameters()

        self.point_encoder = point_encoder
        self.depth_encoder = depth_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512 ** -0.5)
        self.depth_projection = nn.Parameter(torch.empty(kwargs.depth_feat_dims, 512))
        nn.init.normal_(self.depth_projection, std=512 ** -0.5)

        self.ulip = config.ulip

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    # 下面对 init_classifier 和 initialize_parameters 两个函数进行详细解释

    def init_classifier(self, args):
        """
        初始化零样本分类器权重。

        主要流程如下：
        1. 遍历所有类别（self.classes），对每个类别生成一组文本描述（self.templates），
           并用 tokenizer 编码成 token。
        2. 对每个类别的所有模板文本，使用 encode_text 得到文本特征（class_embeddings）。
        3. 对每个类别的所有模板特征先做归一化，再对模板特征取均值，再归一化，得到该类别的最终特征。
        4. 将所有类别的特征堆叠，形成 zeroshot_weights。
        5. 将 zeroshot_weights 转置后作为可学习参数 self.classifier。
        6. 删除不再需要的文本编码相关模块，节省显存。

        参数:
            args: 训练参数，包含设备信息等。
        """
        text_features = []
        for l in self.classes.keys():
            # 1. 生成该类别的所有模板文本
            texts = [t.format(l) for t in self.templates]
            # 2. 编码为 token，并放到指定 GPU 上
            texts = self.tokenizer(texts).cuda(args.gpu, non_blocking=True)
            # 3. 得到每个模板的文本特征
            with torch.no_grad():
                class_embeddings = self.encode_text(texts)
            # 4. 对每个模板特征归一化
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            # 5. 对所有模板特征取均值
            class_embeddings = class_embeddings.mean(dim=0)
            # 6. 再归一化
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            # 7. 加入类别特征列表
            text_features.append(class_embeddings)

        # 8. 堆叠所有类别的特征，形成零样本权重
        self.zeroshot_weights = torch.stack(text_features, dim=0)
        # 9. 将权重转置后作为可学习参数 classifier
        self.classifier = nn.Parameter(self.zeroshot_weights.T.to(args.device))

        # 10. 删除不再需要的文本编码相关模块，节省显存
        del self.transformer, self.token_embedding, self.positional_embedding, self.ln_final, self.text_projection #, self.logit_scale
        return

    def initialize_parameters(self):
        """
        初始化模型参数，包括 token embedding、位置编码、transformer 各层参数、投影参数等。

        主要流程如下：
        1. 初始化 token embedding 权重，正态分布 std=0.02。
        2. 初始化位置编码，正态分布 std=0.01。
        3. 计算 transformer 各层参数初始化的标准差。
        4. 遍历每个 transformer block，分别初始化注意力和 MLP 层的权重。
        5. 初始化图像和文本投影参数。
        """
        # 1. 初始化 token embedding
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        # 2. 初始化位置编码
        nn.init.normal_(self.positional_embedding, std=0.01)

        # 3. 计算 transformer 各层参数初始化的标准差
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        # 4. 初始化 transformer 各 block 的参数
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 5. 初始化图像和文本投影参数
        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # 下面三个方法分别用于编码文本、图像和点云特征。

    def encode_text(self, text, mask=None):
        """
        编码文本输入，得到文本特征。

        步骤说明：
        1. 通过 token_embedding 将文本 token 序列映射为嵌入向量。
        2. 加上位置编码 positional_embedding。
        3. 变换维度顺序以适配 transformer 输入（NLD -> LND）。
        4. 输入 transformer 进行特征提取。
        5. 再变换回原始维度顺序（LND -> NLD）。
        6. 通过 LayerNorm 归一化。
        7. 取每个样本的 EOT（end of text）token 位置的特征，经过 text_projection 得到最终文本特征。
        """
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # 取每个样本的 EOT token 位置的特征，经过 text_projection 得到最终文本特征
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_image(self, image, Mask):
        """
        编码图像输入，得到图像特征。

        参数说明：
        - image: 输入图像
        - Mask: 掩码信息（可选）

        返回：
        - 图像特征
        """
        x = self.visual(image, Mask)
        return x

    def encode_pc(self, pc, Mask):
        """
        编码点云输入，得到点云特征。

        参数说明：
        - pc: 输入点云
        - Mask: 掩码信息（可选）

        返回：
        - 点云特征
        """
        pc_feat = self.point_encoder(pc, Mask)
        return pc_feat
    
    def encode_depth(self, depth, Mask):
        """
        编码深度图输入，得到深度图特征。
        参数:
        - depth: 输入深度图
        - Mask: 掩码信息（可选）
        返回:
        - 深度图特征
        """
        # print("encode_depth_mask")
        # depth_feat = self.depth_encoder(depth, Mask)
        depth_feat = self.depth_encoder(depth)
        return depth_feat

    def forward(self, image, pc, depth, text=None, Mask=None, val=False):
        batch_size = pc.shape[0]
        pc_embed = self.encode_pc(pc, Mask)
        image_embed = self.encode_image(image, Mask)
        # register_hook(self.depth_encoder)
        depth_embed = self.encode_depth(depth, Mask)
        depth_feat = depth_embed
        
        # 兼容多种depth输入shape，保证depth1和depth2都能获得
        # 先去除多余的1维（如评估时[B, 1, 2, H, W]）
        # while depth.dim() > 4:
        #     print("depth.dim() > 4")
        #     depth = depth.squeeze(1)
        # # 如果是[B, H, W]，加channel维
        # if depth.dim() == 3:
        #     print("depth.dim() == 3")
        #     depth = depth.unsqueeze(1)  # [B, 1, H, W]
        # # 兼容通道数为2的情况
        # if depth.shape[1] == 2:
        #     depth1 = depth[:, 0].unsqueeze(1)  # [B, 1, H, W]
        #     depth2 = depth[:, 1].unsqueeze(1)  # [B, 1, H, W]
        # elif depth.shape[1] == 1 and depth.shape[2] == 2:
        #     # 兼容评估时[B, 1, 2, H, W] squeeze后变[B, 2, H, W]
        #     depth1 = depth[:, 0].unsqueeze(1)
        #     depth2 = depth[:, 1].unsqueeze(1)
        # else:
        #     raise ValueError(f"depth shape不支持: {depth.shape}")
        # depth_cat = torch.cat([depth1, depth2], dim=0)  # [2B, 1, H, W]
        # depth_embed = self.encode_depth(depth_cat, Mask)
        
        # depth1 = depth[:, 0]
        # depth2 = depth[:, 1]
        # depth = torch.cat([depth1, depth2], dim=0)
        # depth_embed = self.encode_depth(depth, Mask)
        
        # depth1_feat = depth_embed[: batch_size]
        # depth2_feat = depth_embed[batch_size:]
        # depth_feat = (depth1_feat + depth2_feat) * 0.5
        # print("depth_feat.shape,depth1_feat.shape,depth2_feat.shape,depth_embed.shape",depth_feat.shape,depth1_feat.shape,depth2_feat.shape,depth_embed.shape)
        
        
        if self.ulip:
            if val:
                pc_embed = pc_embed @ self.pc_projection
                image_embed = image_embed @ self.image_projection
                depth_embed = depth_feat @ self.depth_projection

                return {'pc_embed': pc_embed,
                        'image_embed': image_embed,
                        'depth_embed': depth_embed,
                        'logit_scale_depth':self.logit_scale_depth.exp()}

            text_embed_all = []
            for i in range(text.shape[0]):
                text_for_one_sample = text[i]
                text_embed = self.encode_text(text_for_one_sample)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed = text_embed.mean(dim=0)
                text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
                text_embed_all.append(text_embed)

            text_embed_all = torch.stack(text_embed_all)

            pc_embed = pc_embed @ self.pc_projection
            image_embed = image_embed @ self.image_projection
            depth_embed = depth_feat @ self.depth_projection

            return {'text_embed': text_embed_all,
                    'pc_embed': pc_embed,
                    'image_embed': image_embed,
                    'depth_embed': depth_embed,
                    'logit_scale': self.logit_scale.exp(),
                    'logit_scale_depth':self.logit_scale_depth.exp()
                    }

        elif Mask is None:
            # 计算图像的logits。首先将image_embed通过image_projection线性变换，然后进行L2归一化（p=2，按最后一个维度），
            # 接着乘以logit_scale的指数（用于缩放logits），最后与分类器权重做矩阵乘法得到最终的logits。
            image_logits = self.logit_scale.exp() * F.normalize(image_embed @ self.image_projection, dim=-1, p=2) @ self.classifier
            # image_logits = 100 * F.normalize(image_embed @ self.image_projection, dim=-1, p=2) @ self.classifier

            # a = pc_embed @ self.pc_projection
            # a = a / a.norm(dim=-1, keepdim=True)
            # pc_logits = a @ self.classifier

            pc_logits = self.logit_scale.exp() * F.normalize(pc_embed @ self.pc_projection, dim=-1, p=2) @ self.classifier
            # pc_logits = 100 * F.normalize(pc_embed @ self.pc_projection, dim=-1, p=2) @ self.classifier
            # print("pc_embed.shape,self.pc_projection.shape,pc_logits.shape:",pc_embed.shape,self.pc_projection.shape,pc_logits.shape)
            depth_logits = self.logit_scale.exp() * F.normalize(depth_feat @ self.depth_projection, dim=-1, p=2) @ self.classifier
            # print("depth_feat.shape,self.depth_projection.shape,depth_logits.shape:",depth_feat.shape,self.depth_projection.shape,depth_logits.shape)
            return image_logits, pc_logits, depth_logits

        else:
            pc_features = F.normalize(pc_embed[0] @ self.pc_projection, dim=-1, p=2)
            pc_logits, pc_mim, pc_align = self.logit_scale.exp() * pc_features @ self.classifier, pc_embed[1], pc_embed[2]
            # pc_logits, pc_mim, pc_align = 100 * pc_features @ self.classifier, pc_embed[1], pc_embed[2]


            image_features = F.normalize(image_embed[0] @ self.image_projection, dim=-1, p=2)
            image_logits, image_mim, x_patch_image = self.logit_scale.exp() * image_features @ self.classifier, image_embed[1], \
                                                     image_embed[2]
            # image_logits, image_mim, x_patch_image = 100 * image_features @ self.classifier, image_embed[1], \
            #                                          image_embed[2]
            depth_features = F.normalize(depth_feat @ self.depth_projection, dim=-1, p=2)
            # depth_logits, depth_mim, x_patch_depth = self.logit_scale.exp() * depth_features @ self.classifier, depth_feat[1], \
            #                                          depth_feat[2]
            depth_logits, depth_mim, depth_align = self.logit_scale.exp() * depth_features @ self.classifier, depth_feat[1], \
                                                     depth_feat[2]

            x_mask = F.normalize(x_patch_image @ self.image_projection, dim=-1)
            x_cls = F.normalize(image_embed[0] @ self.image_projection, dim=-1, p=2)
            loss_align = torch.sum((x_mask - x_cls.unsqueeze(1)).pow(2), dim=-1, keepdim=True)
            w = Mask.flatten(1).unsqueeze(-1)
            image_align = loss_align[w.bool()].view(w.size(0), -1)
            
            # x_mask_depth = F.normalize(x_patch_depth @ self.depth_projection, dim=-1)
            # x_cls_depth = F.normalize(depth_feat[0] @ self.depth_projection, dim=-1, p=2)
            # loss_align_depth = torch.sum((x_mask_depth - x_cls_depth.unsqueeze(1)).pow(2), dim=-1, keepdim=True)
            
            # depth_align = loss_align_depth[w.bool()].view(w.size(0), -1)

            pc_image_align_logits = pc_features @ image_features.T
            image_pc_align_logits = image_features @ pc_features.T
            depth_image_align_logits = depth_features @ image_features.T   
            image_depth_align_logits = image_features @ depth_features.T
            depth_pc_align_logits = depth_features @ pc_features.T
            pc_depth_align_logits = pc_features @ depth_features.T
            return image_logits, pc_logits, depth_logits, image_mim, pc_mim, depth_mim, image_align, pc_align, depth_align, pc_image_align_logits, image_pc_align_logits, depth_image_align_logits, image_depth_align_logits, depth_pc_align_logits, pc_depth_align_logits


def get_loss(args):
    return losses.ULIPWithImageLoss()


def get_metric_names(model):
    return ['loss', 'ulip_loss', 'ulip_pc_image_acc', 'ulip_pc_text_acc']


def ULIP_MUST_PointBERT(args, classes, templates, tokenizer):
    """
    该函数用于构建ULIP_MUST_PointBERT模型，并根据不同的训练/微调需求加载相应的预训练权重。

    主要流程如下：
    1. 解析CLIP模型权重，提取视觉和文本相关的结构参数（如宽度、层数、patch大小、分辨率等）。
    2. 构建视觉Transformer（VisionTransformer_MIM）和点云编码器（MaskTransformerMUST）。
    3. 实例化CrossMoST多模态模型，指定各类参数。
    4. 从CLIP权重中筛选部分参数加载到模型（如视觉投影层、部分BN统计量等）。
    5. 根据args.ulip和args.from_scratch决定是否加载ULIP/PointBERT等预训练权重，并设置参数是否可训练。

    参数说明：
    - args: 命令行参数，包含模型结构、预训练权重路径、是否ulip等信息
    - classes: 类别列表
    - templates: 模板列表
    - tokenizer: 分词器

    返回：
    - model: 构建并加载好权重的CrossMoST模型
    """

    # 1. 解析CLIP模型权重，提取结构参数
    state_dict = load_state_dict(args.clip_model).state_dict()
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_heads = vision_width // 64

    # 2. 构建视觉Transformer
    vision_model = VisionTransformer_MIM(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=embed_dim,
        mask=args.mask
    )

    # 3. 构建点云编码器
    config_addr = args.config
    config = cfg_from_yaml_file(config_addr)
    point_encoder = MaskTransformerMUST(config.model)
    pc_feat_dims = 768  # 点云特征维度
    
    # 构建深度图编码器
    # V1
    # encoder = UResnetEncoder(ngf=32, n_blocks=1, n_down=2)
    # depth_encoder = DepthFeatureEncoder(encoder, proj_dim=512, use_multi_scale=True)
    
    # DPT
    # depth_encoder = DepthFeatureEncoderWithMask(encoder, proj_dim=512, use_multi_scale=True)
    # depth_encoder = DPTDepthEncoder(
    #     backbone="vitb_rn50_384",
    #     output_level="path_1",
    #     enable_attention_hooks=True  
    # )
    
    # V2
    # print("创建V2深度编码器")
    from models.depth.DepthEncoderV2 import DepthFeatureEncoderV2,UResnetEncoderV2,BottleneckResidualBlock
    base_encoder = UResnetEncoderV2(
        ngf=32, 
        n_blocks=2, 
        n_down=2, 
        norm_type='instance', 
        use_dilation=True
    )
    # 初始化特征投影层：proj_dim=512（固定输出维度，适配论文端到端训练），use_multi_scale=True（融合多尺度特征）
    depth_encoder = DepthFeatureEncoderV2(
        encoder=base_encoder, 
        proj_dim=512, 
        use_multi_scale=True
    )
    depth_feat_dims = 512  # 深度图特征维度

    # 4. 实例化CrossMoST模型
    model = TMB(
        args, config,
        embed_dim=embed_dim,
        vision_width=vision_width,
        point_encoder=point_encoder,
        depth_encoder=depth_encoder,
        vision_model=vision_model,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        pc_feat_dims=pc_feat_dims,
        depth_feat_dims=depth_feat_dims,
        classes=classes,
        templates=templates,
        tokenizer=tokenizer,
        mask=args.mask
    )

    # 5. 从CLIP权重中筛选部分参数加载到模型
    needed = [name for name, v in model.named_parameters()]
    # BN统计量等特殊参数
    other_needed = [
        "point_encoder.encoder.first_conv.1.running_mean", "point_encoder.encoder.first_conv.1.running_var",
        "point_encoder.encoder.second_conv.1.running_mean", "point_encoder.encoder.second_conv.1.running_var"
    ]
    needed.extend(other_needed)

    clip_model = load_state_dict(args.clip_model).state_dict()
    avail = list(clip_model.keys())
    temp = {}
    for i in needed:
        if i in avail:
            temp[i] = clip_model[i]
    temp['image_projection'] = clip_model['visual.proj']
    model.load_state_dict(temp, strict=False)

    # 6. 如果ulip模式，加载PointBERT预训练权重，并设置参数可训练性
    if args.ulip:
        for name, param in model.named_parameters():
            # 训练点云分支
            if 'point_encoder' in name or 'pc_projection' in name:
                param.requires_grad = True
            # 训练深度分支
            elif 'depth_encoder' in name or 'depth_projection' in name:
                param.requires_grad = True
            elif 'logit_scale_depth' in name: param.requires_grad = True
            # 其余保持冻结（如文本/视觉CLIP部分）
            else:
                param.requires_grad = False

        # 加载PointBERT预训练权重
        bertpretrain_ulip_model = torch.load(
            './checkpoints/pretrained_models_ckpt_zero-sho_classification_checkpoint_pointbert.pt',
            map_location=torch.device('cpu')
        )
        bertpretrain_ulip_model = bertpretrain_ulip_model['base_model']
        temp = {}
        for key, value in bertpretrain_ulip_model.items():
            r = ("transformer_q.", "point_encoder.")
            k = key.replace(*r)
            if "point_encoder.lm_head" not in k:
                temp[k] = value
        model.load_state_dict(temp, strict=False)

        print("pointbert weights loaded")
        return model

    # 7. 如果不是从头训练，加载ULIP预训练权重
    if not args.from_scratch:
        pretrained_ulip_model = torch.load(
            './checkpoints/ulip-june11-checkpoint_best.pt',
            map_location=torch.device('cpu')
        )
        pretrained_ulip_model = pretrained_ulip_model['state_dict']
        temp = {}
        for key, value in pretrained_ulip_model.items():
            r = ("module.", "")
            k = key.replace(*r)
            if k != key:
                temp[k] = value
        model.load_state_dict(temp, strict=False)

    return model

