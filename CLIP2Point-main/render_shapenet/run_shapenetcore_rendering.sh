#!/bin/bash

# ShapeNetCore渲染脚本
# 使用方法: ./run_shapenetcore_rendering.sh

# 设置路径
INPUT_PATH="/path/to/ShapeNetCore"  # 请修改为实际的ShapeNetCore路径
OUTPUT_PATH="./rendered_images"      # 输出路径
GPU_ID=0                            # GPU设备ID

# 创建输出目录
mkdir -p $OUTPUT_PATH

# 运行渲染脚本
python render_shapenetcore_dodecahedron.py \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --image_size 224 \
    --points_radius 0.02 \
    --points_per_pixel 1 \
    --gpu $GPU_ID

# 如果只想处理特定类别，可以使用以下命令：
# python render_shapenetcore_dodecahedron.py \
#     --input_path $INPUT_PATH \
#     --output_path $OUTPUT_PATH \
#     --image_size 224 \
#     --points_radius 0.02 \
#     --points_per_pixel 1 \
#     --gpu $GPU_ID \
#     --categories 02691156 02747177 04554684

echo "渲染完成！输出保存在: $OUTPUT_PATH"
