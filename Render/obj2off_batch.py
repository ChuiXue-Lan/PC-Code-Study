import os
import sys
import subprocess
from multiprocessing import Pool, cpu_count

# 配置路径（输入根目录）
SHAPENETCORE_ROOT = 'F:/Datasets/ShapeNet/ShapeNetCore/ShapeNetCore.v1.withoutimages'
RENDER_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'obj2off.py'))

# 遍历ShapeNetCore目录，收集所有model.obj路径及其类别、模型ID
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


def convert_one(args):
    idx, (cat, model_id, obj_path) = args
    model_dir = os.path.dirname(obj_path)
    off_path = os.path.join(model_dir, 'model.off')

    if os.path.isfile(off_path):
        print(f'[{idx+1}] 已存在: {cat}-{model_id}，跳过转换')
        return

    cmd = [sys.executable, RENDER_SCRIPT, os.path.abspath(obj_path), os.path.abspath(off_path)]
    devnull = 'nul' if os.name == 'nt' else '/dev/null'
    try:
        with open(devnull, 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f'[{idx+1}] 转换完成: {cat}-{model_id},剩余文件{len(model_list)-idx-1}')
    except subprocess.CalledProcessError as e:
        print(f'[{idx+1}] 转换失败: {cat}-{model_id}, 错误: {e}')


if __name__ == '__main__':
    with Pool(cpu_count()) as pool:
        pool.map(convert_one, list(enumerate(model_list)))