from vedo import Mesh, Plotter
import numpy as np
from PIL import Image
import os
from pathlib import Path
import trimesh
from tqdm import tqdm

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
        if x == 0:
            az = 90.0 if y > 0 else -90.0
        else:
            az = np.arctan(y / x) * 180 / np.pi
            if x < 0:
                az += 180
        el = np.arctan(z / np.sqrt(x**2 + y**2)) * 180 / np.pi
        azimuths.append(az)
        elevations.append(el)
    return azimuths, elevations

def render_mesh_views_vedo(mesh_path, output_dir, image_size=224):
    mesh = Mesh(mesh_path).color('silver').lighting('plastic')
    azimuths, elevations = get_dodecahedron_views()
    for idx, (az, el) in enumerate(zip(azimuths, elevations)):
        plt = Plotter(offscreen=True, size=(image_size, image_size), bg='white')
        plt.show(mesh, azimuth=az, elevation=el, zoom=1.2, interactive=False)
        img = plt.screenshot(asarray=True)
        img = Image.fromarray(img)
        view_num = f"{idx+1:03d}"
        out_path = os.path.join(output_dir, f"{Path(mesh_path).stem}_{view_num}.png")
        img.save(out_path)
        plt.close()

def main(input_dir, output_dir, image_size=224):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    categories = [d for d in input_dir.iterdir() if d.is_dir()]
    # print(f"找到的类别数量: {len(categories)}")
    for category in categories:
        # print(f"\n处理类别: {category.name}")
        for split in ['train', 'test']:
            current_input_dir = category / split
            current_output_dir = output_dir / category.name / split
            if not current_input_dir.exists():
                print(f"警告: 目录不存在: {current_input_dir}")
                continue
            current_output_dir.mkdir(parents=True, exist_ok=True)
            model_files = list(current_input_dir.glob('*.off'))
            # print(f"在 {current_input_dir} 中找到 {len(model_files)} 个模型文件")
            for model_file in tqdm(model_files, desc=f"{category.name}-{split}"):
                model_name = model_file.stem
                all_views_exist = True
                for v in range(1, 21):
                    view_num = f"{v:03d}"
                    view_file = current_output_dir / f"{model_name}_{view_num}.png"
                    if not view_file.exists():
                        all_views_exist = False
                        break
                # if all_views_exist:
                #     continue
                try:
                    render_mesh_views_vedo(str(model_file), str(current_output_dir), image_size)
                except Exception as e:
                    print(f"处理失败: {model_file.name}\n错误信息: {e}")
    print("所有处理完成！")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='多视角渲染OFF模型')
    parser.add_argument('--input_dir', type=str, required=True, help='输入数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图片根目录')
    parser.add_argument('--image_size', type=int, default=224, help='输出图片尺寸')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.image_size) 