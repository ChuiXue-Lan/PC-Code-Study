import os
import subprocess

# 配置路径
# SHAPENETCORE_ROOT = 'F:/Temp/test/ShapeNetCore'  # 原始ShapeNetCore根目录
# OUTPUT_ROOT = 'F:/Temp/test/rendered_images'     # 输出根目录
SHAPENETCORE_ROOT = 'F:/Datasets/ShapeNet/ShapeNetCore/ShapeNetCore.v1.withoutimages'
OUTPUT_ROOT = 'F:/Datasets/ShapeNet/ShapeNet55/rendered_images'
BLENDER_PATH = 'E:/Software/blender-2.92.0-windows64/blender.exe'            # blender命令（如有需要可写绝对路径）
RENDER_SCRIPT = os.path.abspath('Some/stanford-shapenet-renderer-master/my/render_blender_rgb_depth.py')
VIEWS = 30                          # 渲染视角数
RESOLUTION = 224                    # 输出分辨率

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

for idx, (cat, model_id, obj_path) in enumerate(model_list):
    tag = f'{cat}-{model_id}'
    output_dir = os.path.join(OUTPUT_ROOT, tag)
    os.makedirs(output_dir, exist_ok=True)

    # 检查是否已渲染完成（RGB和depth图片都存在）
    rendered = True
    for i in range(VIEWS):
        base_name = f"{model_id}_r_{i*int(360/VIEWS):03d}"
        
        rgb_path = os.path.join(output_dir, base_name + ".png").replace('\\', '/')
        depth_path = os.path.join(output_dir, base_name + "_depth0001.png").replace('\\', '/')
        if not (os.path.isfile(rgb_path) and os.path.isfile(depth_path)):
            rendered = False
            break
    if rendered:
        print(f"[{idx+1}/{len(model_list)}] 已存在: {tag}，跳过渲染")
        continue

    print(f'[{idx+1}/{len(model_list)}] 渲染: {tag}')
    # 调用blender渲染
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
    
    try:
        # subprocess.run(cmd, check=True)
        
        # # 输出到日志文件
        # with open('blender.log', 'w') as f:
        #     subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

        # 或者完全丢弃输出（Windows）
        with open('nul', 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        
    except subprocess.CalledProcessError as e:
        print(f'渲染失败: {tag}, 错误: {e}') 