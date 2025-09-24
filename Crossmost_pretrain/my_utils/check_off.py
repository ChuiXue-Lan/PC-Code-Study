import os
import open3d as o3d

def check_off_files(root_dir):
    bad_files = []
    total = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".off"):
                print(f"正在检查%s文件..."%file)
                total += 1
                file_path = os.path.join(root, file)
                try:
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    if len(mesh.triangles) == 0:
                        print(f"[No Triangles] {file_path}")
                        bad_files.append(file_path)
                except Exception as e:
                    print(f"[Error] {file_path}: {e}")
                    bad_files.append(file_path)
    print("\n检查完成 ✅")
    print(f"共检查了 {total} 个 .off 文件，发现 {len(bad_files)} 个无三角形或加载失败的文件。")
    return bad_files

# 示例调用：替换为你自己的数据集根目录
if __name__ == "__main__":
    root_dir = "/home/cls2024/ltx/Replicate/CrossMoST-main/data/modelnet40_rendered/ModelNet40"  # 修改为你的 .off 文件根目录
    bad_off_files = check_off_files(root_dir)
