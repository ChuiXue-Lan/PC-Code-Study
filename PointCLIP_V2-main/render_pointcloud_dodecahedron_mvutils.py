import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import trimesh

# ====== 视角设置 ======
def get_dodecahedron_views():
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi],
        [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
    ])
    azimuths = []
    elevations = []
    for v in vertices:
        x, y, z = v
        az = np.arctan2(y, x) * 180 / np.pi
        el = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
        azimuths.append(az)
        elevations.append(el)
    return azimuths, elevations

# ====== 点云归一化 ======
def normalize_points(points):
    center = torch.mean(points, dim=1, keepdim=True)
    points = points - center
    scale = torch.max(torch.norm(points, dim=2, keepdim=True), dim=1, keepdim=True)[0]
    points = points / scale
    return points

# ====== 点云转深度图核心 ======
from torch_scatter import scatter
import torch.nn as nn

params = {'maxpoolz':1, 'maxpoolxy':7, 'maxpoolpadz':0, 'maxpoolpadxy':2,
            'convz':1, 'convxy':3, 'convsigmaxy':3, 'convsigmaz':1, 'convpadz':0, 'convpadxy':1,
            'imgbias':0., 'depth_bias':0.2, 'obj_ratio':0.8, 'bg_clr':0.0,
            'resolution': 112, 'depth': 8}

class Grid2Image(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool3d((params['maxpoolz'], params['maxpoolxy'], params['maxpoolxy']), 
                                    stride=1, padding=(params['maxpoolpadz'], params['maxpoolpadxy'], 
                                    params['maxpoolpadxy']))
        self.conv = torch.nn.Conv3d(1, 1, kernel_size=(params['convz'], params['convxy'], params['convxy']),
                                    stride=1, padding=(params['convpadz'],params['convpadxy'],params['convpadxy']),
                                    bias=True)
        kn3d = get3DGaussianKernel(params['convxy'], params['convz'], sigma=params['convsigmaxy'], zsigma=params['convsigmaz'])
        self.conv.weight.data = torch.Tensor(kn3d).repeat(1,1,1,1,1)
        self.conv.bias.data.fill_(0)
    def forward(self, x):
        x = self.maxpool(x.unsqueeze(1))
        x = self.conv(x)
        img = torch.max(x, dim=2)[0]
        img = img / torch.max(torch.max(img, dim=-1)[0], dim=-1)[0][:,:,None,None]
        img = 1 - img
        img = img.repeat(1,3,1,1)
        return img

def get2DGaussianKernel(ksize, sigma=0):
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    return kernel

def get3DGaussianKernel(ksize, depth, sigma=2, zsigma=2):
    kernel2d = get2DGaussianKernel(ksize, sigma)
    zs = (np.arange(depth, dtype=np.float32) - depth//2)
    zkernel = np.exp(-(zs ** 2) / (2 * zsigma ** 2))
    kernel3d = np.repeat(kernel2d[None,:,:], depth, axis=0) * zkernel[:,None, None]
    kernel3d = kernel3d / torch.sum(kernel3d)
    return kernel3d

def euler2mat(angle):
    angle = np.array(angle)
    if len(angle.shape) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        Rx = np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
        Ry = np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
        Rz = np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        return Rz @ Ry @ Rx
    else:
        mats = []
        for a in angle:
            mats.append(euler2mat(a))
        return np.stack(mats)

def get_rotation_matrices(azimuths, elevations):
    # azimuth, elevation in degrees
    mats = []
    for az, el in zip(azimuths, elevations):
        # 先绕z轴azimuth，再绕y轴-elevation
        az_rad = np.deg2rad(az)
        el_rad = np.deg2rad(el)
        mat = euler2mat([0, el_rad, az_rad])
        mats.append(mat)
    return np.stack(mats)

def points2grid(points, resolution=params['resolution'], depth=params['depth']):
    batch, pnum, _ = points.shape
    pmax, pmin = points.max(dim=1)[0], points.min(dim=1)[0]
    pcent = (pmax + pmin) / 2
    pcent = pcent[:, None, :]
    prange = (pmax - pmin).max(dim=-1)[0][:, None, None]
    points = (points - pcent) / prange * 2.
    points[:, :, :2] = points[:, :, :2] * params['obj_ratio']
    depth_bias = params['depth_bias']
    _x = (points[:, :, 0] + 1) / 2 * resolution
    _y = (points[:, :, 1] + 1) / 2 * resolution
    _z = ((points[:, :, 2] + 1) / 2 + depth_bias) / (1+depth_bias) * (depth - 2)
    _x.ceil_()
    _y.ceil_()
    z_int = _z.ceil()
    _x = torch.clip(_x, 1, resolution - 2)
    _y = torch.clip(_y, 1, resolution - 2)
    _z = torch.clip(_z, 1, depth - 2)
    coordinates = z_int * resolution * resolution + _y * resolution + _x
    grid = torch.ones([batch, depth, resolution, resolution], device=points.device).view(batch, -1) * params['bg_clr']
    grid = scatter(_z, coordinates.long(), dim=1, out=grid, reduce="max")
    grid = grid.reshape((batch, depth, resolution, resolution)).permute((0,1,3,2))
    return grid

# ====== 渲染主函数 ======
def render_pointcloud_views(points, rot_mats, device):
    # points: [1, N, 3], rot_mats: [20, 3, 3]
    imgs = []
    for i in range(rot_mats.shape[0]):
        rot = torch.from_numpy(rot_mats[i]).float().to(device)
        pts_rot = torch.matmul(points, rot.T)
        grid = points2grid(pts_rot)
        grid2img = Grid2Image().to(device)
        with torch.no_grad():
            img = grid2img(grid)
        # 只取单通道深度
        img = img[0,0].cpu().numpy()
        imgs.append(img)
    return imgs

# ====== OFF/NPY读取 ======
def load_off(file_path):
    mesh = trimesh.load_mesh(file_path)
    points = mesh.vertices
    return torch.tensor(points).float()

def load_npy(file_path):
    arr = np.load(file_path)
    if arr.ndim == 2:
        return torch.tensor(arr).float().unsqueeze(0)
    elif arr.ndim == 3:
        return torch.tensor(arr).float()
    else:
        raise ValueError('npy格式不支持')

# ====== 批量处理 ======
def process_category(category_path, output_path, device, partition, file_ext='.off'):
    category_name = os.path.basename(category_path)
    output_category_path = os.path.join(output_path, category_name, partition)
    os.makedirs(output_category_path, exist_ok=True)
    files = [f for f in os.listdir(os.path.join(category_path, partition)) if f.endswith(file_ext)]
    azimuths, elevations = get_dodecahedron_views()
    rot_mats = get_rotation_matrices(azimuths, elevations)
    view_order = [1, 5, 0, 4, 3, 7, 2, 6, 13, 15, 12, 14, 16, 17, 18, 19, 9, 11, 8, 10]
    for file in tqdm(files, desc=f"Processing {category_name}-{partition}"):
        if file_ext == '.off':
            points = load_off(os.path.join(category_path, partition, file)).unsqueeze(0)
        else:
            points = load_npy(os.path.join(category_path, partition, file))
        points = normalize_points(points).to(device)
        imgs = render_pointcloud_views(points, rot_mats, device)
        base_name = os.path.splitext(file)[0]
        for view_idx, idx in enumerate(view_order):
            save_name = f"{base_name}_view{view_idx+1:02d}"
            save_path = os.path.join(output_category_path, f"{save_name}.npy")
            np.save(save_path, imgs[idx])

# ====== main入口 ======
def main():
    parser = argparse.ArgumentParser(description='Render depth maps for ModelNet40 dataset using dodecahedron views (mv_utils style)')
    parser.add_argument('--input_path', type=str, required=True, help='Path to ModelNet40 dataset')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save rendered depth maps')
    parser.add_argument('--file_ext', type=str, default='.off', help='File extension: .off or .npy')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories = [d for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))]
    for category in categories:
        print("==============================")
        print("当前处理类别：", category)
        category_path = os.path.join(args.input_path, category)
        process_category(category_path, args.output_path, device, 'train', file_ext=args.file_ext)
        process_category(category_path, args.output_path, device, 'test', file_ext=args.file_ext)

if __name__ == '__main__':
    main() 