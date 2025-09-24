# 示例 Python 检查脚本
import os
import open3d as o3d

off_dir = "/home/cls2024/ltx/Replicate/CrossMoST-main/data/modelnet40_rendered/ModelNet40-mini"
for fname in os.listdir(off_dir):
    if fname.endswith(".off"):
        mesh = o3d.io.read_triangle_mesh(os.path.join(off_dir, fname))
        if len(mesh.triangles) == 0:
            print(f"Invalid OFF file (no triangles): {fname}")