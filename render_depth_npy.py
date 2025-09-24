import os
import numpy as np
import torch
import argparse
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer,
    look_at_view_transform, TexturesVertex
)
from pytorch3d.structures import Meshes
import trimesh

def load_mesh(path, device):
    ext = os.path.splitext(path)[-1].lower()
    if ext == '.obj':
        mesh = load_objs_as_meshes([path], device=device)
    elif ext == '.off':
        mesh_trimesh = trimesh.load(path)
        verts = torch.tensor(mesh_trimesh.vertices, dtype=torch.float32, device=device).unsqueeze(0)
        faces = torch.tensor(mesh_trimesh.faces, dtype=torch.int64, device=device).unsqueeze(0)
        textures = TexturesVertex(verts_features=torch.ones_like(verts))
        mesh = Meshes(verts=verts, faces=faces, textures=textures)
    else:
        raise ValueError("Unsupported mesh format.")
    return mesh

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    mesh = load_mesh(args.mesh_path, device)

    # Normalize to unit sphere
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(), 1e-5)
    mesh.offset_verts_(-center)
    mesh.scale_verts_(1.0 / scale.item())

    # Camera and rasterizer settings
    raster_settings = RasterizationSettings(
        image_size=args.resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    for i in range(args.views):
        azim = 360.0 * i / args.views
        R, T = look_at_view_transform(dist=args.camera_dist, elev=args.elev, azim=azim)
        camera = FoVPerspectiveCameras(device=device, R=R, T=T)

        rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)
        fragments = rasterizer(mesh)

        # Extract depth map
        depth = fragments.zbuf[0, ..., 0].cpu().numpy()  # (H, W)

        npy_path = os.path.join(args.output_dir, f'depth_{i:03d}.npy')
        np.save(npy_path, depth)
        print(f"[Saved] {npy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', type=str, default='/data/Lan/datasets/ShapeNet/test/model.obj', help='Path to .obj or .off file')
    parser.add_argument('--output_dir', type=str, default='/data/Lan/datasets/ShapeNet/test/rendered_images/')
    parser.add_argument('--views', type=int, default=30)
    parser.add_argument('--camera_dist', type=float, default=2.0)
    parser.add_argument('--elev', type=float, default=30.0)
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()
    main(args)
