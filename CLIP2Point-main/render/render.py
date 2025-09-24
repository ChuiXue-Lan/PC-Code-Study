from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, OpenGLOrthographicCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, HardPhongShader, PointsRasterizationSettings, PointsRasterizer, DirectionalLights)
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer.mesh import TexturesAtlas
from pytorch3d.structures import Meshes, Pointclouds
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch
from torchvision.transforms import Normalize
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ORTHOGONAL_THRESHOLD = 1e-6
EXAHSTION_LIMIT = 20


def batch_tensor(tensor, dim=1, squeeze=False):
    """
    a function to reshape pytorch tensor `tensor` along some dimension `dim` to the batch dimension 0 such that the tensor can be processed in parallel. 
    if `sqeeze`=True , the diension `dim` will be removed completelelky, otherwize it will be of size=1.  cehck `unbatch_tensor()` for the reverese function
    该函数用于将PyTorch张量tensor沿着指定维度dim重塑到batch维度(0维),使得张量可以并行处理。

    参数:
        tensor: 输入的PyTorch张量
        dim: 要重塑的维度,默认为1 
        squeeze: 布尔值,如果为True,则完全移除dim维度;如果为False,则该维度大小变为1

    返回:
        重塑后的张量。可以使用unbatch_tensor()函数恢复原始形状。
    """
    batch_size, dim_size = tensor.shape[0], tensor.shape[dim]
    returned_size = list(tensor.shape) 
    returned_size[0] = batch_size*dim_size
    returned_size[dim] = 1
    if squeeze:
        return tensor.transpose(0, dim).reshape(returned_size).squeeze_(dim)
    else:
        return tensor.transpose(0, dim).reshape(returned_size)


def unbatch_tensor(tensor, batch_size, dim=1, unsqueeze=False):
    """
    a function to chunk pytorch tensor `tensor` along the batch dimension 0 and cincatenate the chuncks on dimension `dim` to recover from `batch_tensor()` function.
    if `unsqueee`=True , it will add a dimension `dim` before the unbatching 
    """
    fake_batch_size = tensor.shape[0]
    nb_chunks = int(fake_batch_size / batch_size)
    if unsqueeze:
        return torch.cat(torch.chunk(tensor.unsqueeze_(dim), nb_chunks, dim=0), dim=dim).contiguous()
    else:
        return torch.cat(torch.chunk(tensor, nb_chunks, dim=0), dim=dim).contiguous()


def check_valid_rotation_matrix(R, tol: float = 1e-6):
    """
    判断输入的R是否为有效的旋转矩阵。
    有效旋转矩阵需满足以下两个条件：
    1. R * R^T = I（正交性）
    2. det(R) = 1（行列式为1，无畸变）

    参数:
        R: 形状为(N, 3, 3)的张量，表示N个旋转矩阵
        tol: 容差，判断正交性时的误差上限

    返回:
        bool: 若R为有效旋转矩阵，返回True，否则返回False
    """
    N = R.shape[0]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    # 检查正交性：R * R^T 是否接近单位阵
    orthogonal = torch.allclose(R.bmm(R.transpose(1, 2)), eye, atol=tol)
    # 检查行列式是否为1
    det_R = torch.det(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))
    return orthogonal and no_distortion


def check_and_correct_rotation_matrix(R, T, nb_trials, azim, elev, dist):
    """
    检查并修正旋转矩阵R是否有效。
    若R无效，则通过扰动视角参数（azim, elev, dist）重新生成旋转矩阵，最多尝试nb_trials次。
    若多次尝试后仍无效，则程序退出。

    参数:
        R: 初始旋转矩阵
        T: 初始平移向量
        nb_trials: 最大尝试次数
        azim, elev, dist: 当前视角参数

    返回:
        (R, T): 有效的旋转矩阵和平移向量
    """
    exhastion = 0
    while not check_valid_rotation_matrix(R):
        exhastion += 1
        # 随机扰动elev和azim，重新生成旋转矩阵
        R, T = look_at_view_transform(
            dist=batch_tensor(dist.T, dim=1, squeeze=True),
            elev=batch_tensor(elev.T + 90.0 * torch.rand_like(elev.T, device=elev.device), dim=1, squeeze=True),
            azim=batch_tensor(azim.T + 180.0 * torch.rand_like(azim.T, device=elev.device), dim=1, squeeze=True)
        )
        # 若多次尝试仍无效，直接退出
        if not check_valid_rotation_matrix(R) and exhastion > nb_trials:
            sys.exit("Remedy did not work")
    return R, T


class Renderer(nn.Module):
    """
    The Multi-view differntiable renderer main class that render multiple views differntiably from some given viewpoints. It can render meshes and point clouds as well
    Args: 
        `nb_views` int , The number of views used in the multi-view setup
        `image_size` int , The image sizes of the rendered views.
        `pc_rendering` : bool , flag to use point cloud rendering instead of mesh rendering
        `object_color` : str , The color setup of the objects/points rendered. Choices: ["white", "random","black","red","green","blue", "custom"]
        `background_color` : str , The color setup of the rendering background. Choices: ["white", "random","black","red","green","blue", "custom"]
        `faces_per_pixel` int , The number of faces rendered per pixel when mesh rendering is used (`pc_rendering` == `False`) .
        `points_radius`: float , the radius of the points rendered. The more points in a specific `image_size`, the less radius required for proper rendering.
        `points_per_pixel` int , The number of points rendered per pixel when point cloud rendering is used (`pc_rendering` == `True`) .
        `light_direction` : str , The setup of the light used in rendering when mesh rendering is available. Choices: ["fixed", "random", "relative"]
        `cull_backfaces` : bool , Allow backface-culling when rendering meshes (`pc_rendering` == `False`).

    Returns:
        an MVTN object that can render multiple views according to predefined setup
    """

    def __init__(self, image_size=224, points_radius=0.02, points_per_pixel=1):
        super().__init__()
        self.image_size = image_size
        self.points_radius = points_radius
        self.points_per_pixel = points_per_pixel
        self.normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.light_direction_type = 'random'

    def norm(self, img):  # [B, H, W]
        detached_img = img.detach()
        B, H, W = detached_img.shape

        mask = detached_img > 0
        batch_points = detached_img.reshape(B, -1)
        batch_max, _ = torch.max(batch_points, dim=1, keepdim=True)
        batch_max = batch_max.unsqueeze(-1).repeat(1, H, W)
        detached_img[~mask] = 1.
        batch_points = detached_img.reshape(B, -1)
        batch_min, _ = torch.min(batch_points, dim=1, keepdim=True)
        batch_min = batch_min.unsqueeze(-1).repeat(1, H, W)
        img = img.sub_(batch_min).div_(batch_max) * 200. / 255.
        img[~mask] = 1.
        # return self.normalize(img.unsqueeze(1).repeat(1, 3, 1, 1))
        return img.unsqueeze(1).repeat(1, 3, 1, 1)

    def render_meshes(self, meshes, azim, elev, dist, view, lights, background_color=(1.0, 1.0, 1.0)):
        """渲染3D网格模型
        Args:
            meshes: 包含网格信息的字典列表,每个字典包含顶点(verts)、面片(faces)和纹理(textures)
            azim: 方位角,控制相机水平旋转角度
            elev: 仰角,控制相机垂直旋转角度 
            dist: 相机距离物体的距离
            view: 渲染视角的数量
            lights: 光照设置
            background_color: 背景颜色,默认为白色(1.0, 1.0, 1.0)
        Returns:
            normalized_images: 归一化后的渲染图像
        """
        # 将所有网格信息整合到一个字典中
        collated_dict = {}
        for k in meshes[0].keys():
            collated_dict[k] = [d[k] for d in meshes]
        # 创建纹理图集
        textures = TexturesAtlas(atlas=collated_dict["textures"])

        # 创建Pytorch3D网格对象
        new_meshes = Meshes(
            verts=collated_dict["verts"],
            faces=collated_dict["faces"], 
            textures=textures,
        ).to(lights.device)

        # 计算相机的旋转矩阵R和平移向量T
        R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
            elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))

        # 创建透视相机
        cameras = OpenGLPerspectiveCameras(
            device=lights.device, R=R, T=T)
        camera = OpenGLPerspectiveCameras(device=lights.device, R=R[None, 0, ...],
                                          T=T[None, 0, ...])

        # 设置光栅化参数
        raster_settings = RasterizationSettings(
            image_size=self.image_size,  # 渲染图像大小
            blur_radius=0.0,  # 不使用模糊
            faces_per_pixel=1,  # 每个像素最多渲染1个面片
            cull_backfaces=False,  # 不剔除背面
            bin_size=0  # 不使用空间划分加速
        )
        
        # 创建网格渲染器
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, raster_settings=raster_settings),
            shader=HardPhongShader(blend_params=BlendParams(background_color=background_color), device=lights.device, cameras=camera, lights=lights)
        )
        
        # 扩展网格以适应多视角渲染
        new_meshes = new_meshes.extend(view)
        # 执行渲染
        rendered_images = renderer(new_meshes, cameras=cameras, lights=lights)

        # 处理渲染结果的维度
        rendered_images = unbatch_tensor(
            rendered_images, batch_size=view, dim=1, unsqueeze=True).transpose(0, 1)

        # 提取RGB通道并调整维度顺序
        rendered_images = rendered_images[...,
                                          0:3].transpose(2, 4).transpose(3, 4)
        # 返回归一化的渲染图像
        return self.normalize(rendered_images)

    def light_direction(self, azim, elev, dist):
        """
        计算光照方向
        Args:
            azim: 方位角
            elev: 仰角 
            dist: 距离
        Returns:
            光照方向向量
        """
        # 固定光照方向 - 从上方照射
        if self.light_direction_type == "fixed":
            return ((0, 1.0, 0),)
            
        # 训练时使用随机光照方向
        elif self.light_direction_type == "random" and self.training:
            # 生成-1到1之间的随机向量
            return (tuple(1.0 - 2 * np.random.rand(3)),)
            
        # 其他情况使用相对视角作为光照方向
        else:
            # 根据球面坐标计算相机位置作为光照方向
            relative_view = Variable(camera_position_from_spherical_angles(
                distance=batch_tensor(dist.T, dim=1, squeeze=True),
                elevation=batch_tensor(elev.T, dim=1, squeeze=True), 
                azimuth=batch_tensor(azim.T, dim=1, squeeze=True)
            )).to(torch.float)

            return relative_view

    def render_points(self, points, azim, elev, dist, view, aug=False, rot=False):
        """
        该函数用于将点云渲染为深度图像。

        参数说明：
        - points: 输入的点云数据，形状为 (B, N, 3)，B为batch size，N为点数。
        - azim: 方位角，形状为 (B, M)，M为视角数。
        - elev: 仰角，形状为 (B, M)。
        - dist: 相机距离，形状为 (B, M)。
        - view: 视角数量M。
        - aug: 是否进行数据增强（如视角翻倍和距离扰动）。
        - rot: 是否对点云进行旋转。

        主要流程：
        1. 如果aug为True，则将视角数量翻倍，并对距离dist进行扰动，模拟不同的观察距离。
        2. 如果rot为True，对点云进行两次旋转，分别绕X轴和Y轴旋转90度（0.5π），以获得不同的空间朝向。
        3. 构造Pointclouds对象，作为渲染输入。
        4. 通过look_at_view_transform函数，根据给定的dist、elev、azim计算相机的旋转矩阵R和平移向量T。
        5. 构造OpenGL正交相机对象。
        6. 设置点云渲染参数，包括图像大小、点半径、每像素点数等。
        7. 构造点云光栅化器PointsRasterizer。
        8. 将点云扩展到views个视角，并根据距离缩放点云。
        9. 渲染点云，得到深度图（zbuf），对最后一维取均值，归一化处理。
        10. 调整输出张量的维度顺序，便于后续处理。

        返回值：
        - rendered_images: 渲染得到的深度图像，形状为 (B, views, H, W)，H和W为图像高宽。
        """
        views = view * 2 if aug else view  # 如果增强，视角数量翻倍
        batch_size = points.shape[0]
        if aug:
            # 数据增强：复制视角，并对距离进行扰动
            azim = azim.repeat(1, 2)  # 将方位角重复一次，扩展维度
            elev = elev.repeat(1, 2)  # 将俯仰角重复一次，扩展维度
            rand_dist1 = dist * (1 + (torch.rand((batch_size, 1), device=points.device) - 0.5) / 5)  # 生成随机距离扰动
            rand_dist2 = dist * (1 + (torch.rand((batch_size, 1), device=points.device) - 0.5) / 5)  # 生成随机距离扰动
            dist = torch.cat([rand_dist1, rand_dist2], dim=1)  # 将两个随机距离扰动拼接在一起
        
        if rot:
            # 对点云进行两次旋转，分别绕X轴和Y轴
            rota1 = axis_angle_to_matrix(torch.tensor([0.5 * np.pi, 0, 0])).to(points.device)
            rota2 = axis_angle_to_matrix(torch.tensor([0, -0.5 * np.pi, 0])).to(points.device)
            points = points @ rota1 @ rota2
            
        # 视角对齐 MATLAB
        rota = axis_angle_to_matrix(torch.tensor([-0.5 * np.pi, 0.0, 0.0])).to(points.device)
        points = points @ rota.T

        # 构造点云对象
        point_cloud = Pointclouds(points=points.to(torch.float))

        # 计算相机的旋转和平移
        R, T = look_at_view_transform(
            dist=batch_tensor(dist.T, dim=1, squeeze=True),
            elev=batch_tensor(elev.T, dim=1, squeeze=True),
            azim=batch_tensor(azim.T, dim=1, squeeze=True),
            # up=((0, 0, 1),)
        )

        # 构造正交相机
        cameras = OpenGLOrthographicCameras(device=points.device, R=R, T=T, znear=0.01)
        # 设置点云渲染参数
        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=self.points_radius,
            points_per_pixel=self.points_per_pixel,
            bin_size=0
        )
        # 构造点云光栅化器
        renderer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        # 扩展点云到views个视角
        point_cloud = point_cloud.extend(views)
        # 根据距离缩放点云
        point_cloud.scale_(batch_tensor(1.0/dist.T, dim=1, squeeze=True)[..., None][..., None].to(points.device))

        # 渲染点云，得到深度图
        rendered_images = torch.mean(renderer(point_cloud).zbuf, dim=-1)
        # 归一化处理
        rendered_images = self.norm(rendered_images)
        # 调整输出张量的维度顺序
        rendered_images = unbatch_tensor(
            rendered_images, batch_size=views, dim=1, unsqueeze=True).transpose(0, 1)

        return rendered_images

    def forward(self, points, azim, elev, dist, view, mesh=None, aug=False, rot=False):
        """
        The main rendering function of the MVRenderer class. It can render meshes (if `self.pc_rendering` == `False`) or 3D point clouds(if `self.pc_rendering` == `True`).
        Arge:
            `meshes`: a list of B `Pytorch3D.Mesh` to be rendered , B batch size. In case not available, just pass `None`. 
            `points`: B * N * 3 tensor, a batch of B point clouds to be rendered where each point cloud has N points and each point has X,Y,Z property. In case not available, just pass `None` .
            `azim`: B * M tensor, a B batch of M azimth angles that represent the azimth angles of the M view-points to render the points or meshes from.
            `elev`: B * M tensor, a B batch of M elevation angles that represent the elevation angles of the M view-points to render the points or meshes from.
            `dist`:  B * M tensor, a B batch of M unit distances that represent the distances of the M view-points to render the points or meshes from.
            `color`: B * N * 3 tensor, The RGB colors of batch of point clouds/meshes with N is the number of points/vertices  and B batch size. Only if `self.object_color` == `custom`, otherwise this option not used

        """
        rendered_depthes = self.render_points(points=points, azim=azim, elev=elev, dist=dist, view=view, aug=aug, rot=rot)

        if mesh is not None:
            background_color = torch.tensor((1.0, 1.0, 1.0), device=points.device)            
            lights = DirectionalLights(device=points.device, direction=self.light_direction(azim, elev, dist))
            rendered_images = self.render_meshes(meshes=mesh, azim=azim, elev=elev, dist=dist * 2, view=view, lights=lights, background_color=background_color)
            return rendered_depthes, rendered_images
        
        return rendered_depthes
