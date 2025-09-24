# ShapeNetCore 深度图渲染脚本

这个脚本用于将ShapeNetCore数据集中的3D模型渲染为深度图，使用十二面体的20个视角。

## 功能特点

- 支持OBJ和OFF格式的3D模型文件
- 使用十二面体的20个均匀分布视角
- 自动归一化点云到单位球内
- 保存为PNG格式的深度图
- 支持断点续传（跳过已存在的文件）
- 支持指定特定类别进行处理

## 目录结构

### 输入目录结构
```
ShapeNetCore/
│   ├── 02691156/
│   │   └── 02691156/
│   │       ├── 1a04e3eab45ca15dd86060f189eb133/
│   │       │   ├── model.obj
│   │       ├── ...
│   │       └── fff513f407e00e85a9ced22d91ad7027/
│   │           ├── model.obj
│   ├── 02747177/
│   │   └── 02747177/
│   │       ├── 1b7d468a27208ee3dad910e221d16b18/
│   │       │   ├── model.obj
│   │       ├── ...
│   │       └── ffe5f0ef45769204cb2a965e75be701c/
│   │           ├── model.off
│   └── ...
```

### 输出目录结构
```
rendered_images/
│   ├── 02691156-1a04e3eab45ca15dd86060f189eb133/
│   │   ├── 02691156-1a04e3eab45ca15dd86060f189eb133_view01.png
│   │   ├── 02691156-1a04e3eab45ca15dd86060f189eb133_view02.png
│   │   ├── ...
│   │   └── 02691156-1a04e3eab45ca15dd86060f189eb133_view20.png
│   ├── 02691156-1a6ad7a24bb89733f412783097373bdc/
│   │   ├── 02691156-1a6ad7a24bb89733f412783097373bdc_view01.png
│   │   ├── 02691156-1a6ad7a24bb89733f412783097373bdc_view02.png
│   │   ├── ...
│   │   └── 02691156-1a6ad7a24bb89733f412783097373bdc_view20.png
│   └── ...
```

## 使用方法

### 1. 基本用法

```bash
python render_shapenetcore_dodecahedron.py \
    --input_path /path/to/ShapeNetCore \
    --output_path ./rendered_images \
    --gpu 0
```

### 2. 指定特定类别

```bash
python render_shapenetcore_dodecahedron.py \
    --input_path /path/to/ShapeNetCore \
    --output_path ./rendered_images \
    --gpu 0 \
    --categories 02691156 02747177 04554684
```

### 3. 使用bash脚本

```bash
# 修改脚本中的路径
vim run_shapenetcore_rendering.sh

# 运行脚本
chmod +x run_shapenetcore_rendering.sh
./run_shapenetcore_rendering.sh
```

## 参数说明

- `--input_path`: ShapeNetCore数据集的根目录路径
- `--output_path`: 渲染结果的保存路径
- `--image_size`: 渲染图像的大小，默认224
- `--points_radius`: 渲染时点的半径，默认0.02
- `--points_per_pixel`: 每个像素的点数，默认1
- `--gpu`: GPU设备ID，默认0
- `--categories`: 指定要处理的类别ID列表（可选）

## 依赖要求

- PyTorch
- PyTorch3D
- trimesh
- PIL (Pillow)
- numpy
- tqdm

## 注意事项

1. 确保有足够的GPU内存来处理渲染任务
2. 脚本会自动跳过已存在的文件，支持断点续传
3. 如果某个模型渲染失败，会记录错误并继续处理下一个模型
4. 深度图会进行归一化处理，保存为0-255的灰度值
5. 每个模型会生成20个视角的深度图（十二面体的顶点数）

## 常见问题

### Q: 如何处理内存不足的问题？
A: 可以尝试减小`image_size`参数或使用更小的`points_radius`值。

### Q: 如何只处理部分数据？
A: 使用`--categories`参数指定特定的类别ID。

### Q: 渲染过程中断怎么办？
A: 重新运行脚本，它会自动跳过已存在的文件。

### Q: 如何修改视角数量？
A: 修改`get_dodecahedron_views()`函数中的顶点定义。

