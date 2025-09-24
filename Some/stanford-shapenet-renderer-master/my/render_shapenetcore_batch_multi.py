import os
import subprocess
from multiprocessing import Pool, cpu_count

# 配置
SHAPENETCORE_ROOT = 'F:/Datasets/ShapeNet/ShapeNetCore/ShapeNetCore.v1.withoutimages'
OUTPUT_ROOT = 'F:/Datasets/ShapeNet/ShapeNet55/rendered_images'
BLENDER_PATH = 'E:/Software/blender-2.92.0-windows64/blender.exe'
RENDER_SCRIPT = os.path.abspath('Some/stanford-shapenet-renderer-master/my/render_blender_rgb_depth.py')
VIEWS = 30
RESOLUTION = 224

# 你的GPU列表（如[0,1,2,3]，根据实际服务器GPU数量填写）
GPU_LIST = [0, 1, 2, 3]

# 收集模型
model_list = []
for cat in os.listdir(SHAPENETCORE_ROOT):
    cat_dir = os.path.join(SHAPENETCORE_ROOT, cat, cat)
    if not os.path.isdir(cat_dir):
        continue
    for model_id in os.listdir(cat_dir):
        obj_path = os.path.join(cat_dir, model_id, 'model.obj')
        if os.path.isfile(obj_path):
            model_list.append((cat, model_id, obj_path))

print(f'共找到 {len(model_list)} 个模型')

def render_one(args):
    idx, (cat, model_id, obj_path), gpu_id = args
    tag = f'{cat}-{model_id}'
    output_dir = os.path.join(OUTPUT_ROOT, tag)
    os.makedirs(output_dir, exist_ok=True)

    # 检查是否已渲染完成
    rendered = True
    for i in range(VIEWS):
        base_name = f"{model_id}_r_{i*int(360/VIEWS):03d}"
        rgb_path = os.path.join(output_dir, base_name + ".png")
        depth_path = os.path.join(output_dir, base_name + "_depth0001.png")
        if not (os.path.isfile(rgb_path) and os.path.isfile(depth_path)):
            rendered = False
            break
    if rendered:
        print(f"[{idx+1}] 已存在: {tag}，跳过渲染")
        return

    print(f'[{idx+1}] 渲染: {tag} (GPU {gpu_id})')
    cmd = [
        BLENDER_PATH, '--background', '--python', RENDER_SCRIPT, '--',
        '--views', str(VIEWS),
        os.path.abspath(obj_path),
        '--output_folder', os.path.abspath(output_dir),
        '--resolution', str(RESOLUTION),
        '--format', 'PNG',
        '--color_depth', '8',
        '--engine', 'BLENDER_EEVEE'
    ]
    # 设置环境变量，指定GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    try:
        with open('nul', 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    except subprocess.CalledProcessError as e:
        print(f'渲染失败: {tag}, 错误: {e}')

if __name__ == '__main__':
    from itertools import cycle
    # 分配GPU
    args_list = [(idx, item, gpu_id) for (idx, item), gpu_id in zip(enumerate(model_list), cycle(GPU_LIST))]
    # 进程数建议等于GPU数
    with Pool(len(GPU_LIST)) as pool:
        pool.map(render_one, args_list)