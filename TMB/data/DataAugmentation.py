import random
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import torchvision.transforms as T
from data.dataset_3d import MaskGenerator

class DataAugmentation:
    def __init__(self, weak_transform, strong_transform, weak_pcltransform, strong_pcltransform, args, config=None):
        self.args = args
        self.istrain = weak_pcltransform is not None
        self.config = config

        if self.istrain:
            self.mask_generator = MaskGenerator(
                input_size=config.input_size,
                mask_patch_size=config.mask_patch_size,
                model_patch_size=config.model_patch_size,
                mask_ratio=config.mask_ratio,
            )

            self.strong_pcltransform_base = strong_pcltransform
            self.weak_pcltransform_base = weak_pcltransform
            self.normalize = transforms.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std))

        else:
            self.transforms = [weak_transform, strong_transform]
            self.pcl_transform = weak_pcltransform
            self.depth_transform = transforms.Compose([
                transforms.Resize(args.input_size, interpolation=T.InterpolationMode.NEAREST),
                transforms.CenterCrop(args.input_size)
            ])

    def apply_shared_transform(self, image, depth, flip, angle, crop_params):
        # 图像增强
        image = TF.resize(image, self.args.input_size, interpolation=TF.InterpolationMode.BICUBIC)
        # depth = TF.resize(depth, self.args.input_size, interpolation=TF.InterpolationMode.NEAREST)
        if isinstance(depth, torch.Tensor):
            # 期望形状 [C,H,W]
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            # 只缩放空间维度
            depth = TF.resize(depth, [self.input_size, self.input_size], interpolation=TF.InterpolationMode.NEAREST)
        else:
            depth = TF.resize(depth, self.input_size, interpolation=TF.InterpolationMode.NEAREST)

        i, j, h, w = crop_params
        image = TF.crop(image, i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)

        if flip:
            image = TF.hflip(image)
            depth = TF.hflip(depth)

        image = TF.rotate(image, angle)
        depth = TF.rotate(depth, angle, interpolation=TF.InterpolationMode.NEAREST)

        return image, depth

    def __call__(self, x, depth, y):
        if self.istrain:
            # ==== 随机参数采样 ====
            flip = random.random() > 0.5
            # angle = random.uniform(-10, 10)
            angle = random.uniform(-5, 5)
            crop_params = transforms.RandomCrop.get_params(x, output_size=(self.args.input_size, self.args.input_size))

            # ==== 处理图像 ====
            depth = torch.from_numpy(depth).float()  # 保持原量纲
            # 先确保为 [1,H,W]，再做几何变换
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)  # [1,H,W]
                
            # image_weak, depth_weak = self.apply_shared_transform(x, TF.to_pil_image(torch.from_numpy(depth).float()), flip, angle, crop_params)
            # image_strong, depth_strong = self.apply_shared_transform(x, TF.to_pil_image(torch.from_numpy(depth).float()), flip, angle, crop_params)

            image_weak, depth_weak = self.apply_shared_transform(x, depth, flip, angle, crop_params)
            image_strong, depth_strong = self.apply_shared_transform(x, depth, flip, angle, crop_params)

            image_weak = TF.to_tensor(image_weak)
            image_strong = TF.to_tensor(image_strong)
            image_weak = self.normalize(image_weak)
            image_strong = self.normalize(image_strong)

            # depth_weak = TF.to_tensor(depth_weak)
            # depth_strong = TF.to_tensor(depth_strong)
            if not isinstance(depth_weak, torch.Tensor):
                depth_weak = TF.to_tensor(depth_weak)
            if not isinstance(depth_strong, torch.Tensor):
                depth_strong = TF.to_tensor(depth_strong)

            # 深度逐样本 min-max 归一化到[0,1]
            def _minmax_norm(d):
                # d: [1,H,W]
                d_min = torch.nan_to_num(d.amin(dim=[1,2], keepdim=True), nan=0.0)
                d_max = torch.nan_to_num(d.amax(dim=[1,2], keepdim=True), nan=1.0)
                return (d - d_min) / (d_max - d_min + 1e-6)
            depth_weak = _minmax_norm(depth_weak.float())
            depth_strong = _minmax_norm(depth_strong.float())

            # ==== 点云同步增强 ====
            pcl_weak = self.weak_pcltransform_base(y, flip=flip, angle=angle)
            pcl_strong = self.strong_pcltransform_base(y, flip=flip, angle=angle)

            # ==== Mask + 返回 ====
            mask = self.mask_generator()
            return image_weak, image_strong, mask, depth_weak, depth_strong, pcl_weak, pcl_strong, 0
        else:
            image = self.transforms[0](x)
            pcl = self.transforms[1](y)

            depth = torch.from_numpy(depth).float().unsqueeze(0)
            depth = self.depth_transform(depth)
            # 测试同样归一化
            d_min = torch.nan_to_num(depth.amin(dim=[1,2], keepdim=True), nan=0.0)
            d_max = torch.nan_to_num(depth.amax(dim=[1,2], keepdim=True), nan=1.0)
            depth = (depth - d_min) / (d_max - d_min + 1e-6)

            return image, depth, pcl
