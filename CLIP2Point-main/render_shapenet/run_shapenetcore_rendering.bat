@echo off
REM ShapeNetCore渲染脚本 (Windows版本)
REM 使用方法: run_shapenetcore_rendering.bat

REM 设置路径
set INPUT_PATH=D:\ShapeNetCore
set OUTPUT_PATH=.\rendered_images
set GPU_ID=0

REM 创建输出目录
if not exist %OUTPUT_PATH% mkdir %OUTPUT_PATH%

echo 开始渲染ShapeNetCore数据集...
echo 输入路径: %INPUT_PATH%
echo 输出路径: %OUTPUT_PATH%
echo GPU设备: %GPU_ID%

REM 运行渲染脚本
python render_shapenetcore_dodecahedron.py ^
    --input_path %INPUT_PATH% ^
    --output_path %OUTPUT_PATH% ^
    --image_size 224 ^
    --points_radius 0.02 ^
    --points_per_pixel 1 ^
    --gpu %GPU_ID%

REM 如果只想处理特定类别，可以使用以下命令：
REM python render_shapenetcore_dodecahedron.py ^
REM     --input_path %INPUT_PATH% ^
REM     --output_path %OUTPUT_PATH% ^
REM     --image_size 224 ^
REM     --points_radius 0.02 ^
REM     --points_per_pixel 1 ^
REM     --gpu %GPU_ID% ^
REM     --categories 02691156 02747177 04554684

echo.
echo 渲染完成！输出保存在: %OUTPUT_PATH%
pause

