# ShapeNetCore渲染脚本总结

基于原有的ModelNet40渲染脚本，我创建了一套完整的ShapeNetCore数据集渲染工具。

## 创建的文件列表

### 1. 主要渲染脚本
- **`render_shapenetcore_dodecahedron.py`** - 主要的渲染脚本
  - 支持OBJ和OFF格式的3D模型文件
  - 使用十二面体的20个均匀分布视角
  - 自动归一化点云到单位球内
  - 保存为PNG格式的深度图
  - 支持断点续传和错误处理

### 2. 运行脚本
- **`run_shapenetcore_rendering.sh`** - Linux/Mac bash脚本
- **`run_shapenetcore_rendering.bat`** - Windows批处理脚本

### 3. 测试和配置
- **`test_shapenetcore_rendering.py`** - 功能测试脚本
- **`config_shapenetcore.yaml`** - 配置文件示例

### 4. 文档
- **`README_ShapeNetCore_Rendering.md`** - 详细使用说明
- **`ShapeNetCore_Rendering_Summary.md`** - 本总结文档

## 主要功能特点

### 1. 目录结构适配
- 自动识别ShapeNetCore的目录结构
- 支持类别ID-模型ID的命名方式
- 自动创建对应的输出目录结构

### 2. 文件格式支持
- 支持OBJ格式（model.obj）
- 支持OFF格式（model.off）
- 自动检测文件类型并选择合适的加载方法

### 3. 渲染功能
- 使用十二面体的20个视角
- 深度图归一化处理
- PNG格式输出
- 视角对齐（MATLAB兼容）

### 4. 错误处理
- 自动跳过已存在的文件
- 错误计数和限制
- 详细的错误日志
- 断点续传支持

### 5. 性能优化
- GPU加速渲染
- 内存管理
- 批处理支持

## 使用方法

### 快速开始
```bash
# 1. 测试功能
python test_shapenetcore_rendering.py --input_path /path/to/ShapeNetCore

# 2. 运行渲染
python render_shapenetcore_dodecahedron.py \
    --input_path /path/to/ShapeNetCore \
    --output_path ./rendered_images \
    --gpu 0

# 3. 或使用脚本
./run_shapenetcore_rendering.sh  # Linux/Mac
run_shapenetcore_rendering.bat   # Windows
```

### 高级用法
```bash
# 指定特定类别
python render_shapenetcore_dodecahedron.py \
    --input_path /path/to/ShapeNetCore \
    --output_path ./rendered_images \
    --gpu 0 \
    --categories 02691156 02747177 04554684
```

## 输出结构

```
rendered_images/
│   ├── 02691156-1a04e3eab45ca15dd86060f189eb133/
│   │   ├── 02691156-1a04e3eab45ca15dd86060f189eb133_view01.png
│   │   ├── 02691156-1a04e3eab45ca15dd86060f189eb133_view02.png
│   │   ├── ...
│   │   └── 02691156-1a04e3eab45ca15dd86060f189eb133_view20.png
│   └── ...
```

## 与原脚本的主要区别

1. **目录结构适配**：从ModelNet40的类别/train/test结构改为ShapeNetCore的类别/模型ID结构
2. **文件格式支持**：增加了OBJ格式支持，同时保留OFF格式
3. **输出格式**：从NPY格式改为PNG格式，便于查看和使用
4. **命名方式**：使用"类别ID-模型ID"的命名方式
5. **错误处理**：增强了错误处理和日志记录
6. **测试功能**：添加了完整的功能测试脚本

## 依赖要求

- PyTorch
- PyTorch3D
- trimesh
- PIL (Pillow)
- numpy
- tqdm
- yaml (可选，用于配置文件)

## 注意事项

1. 确保有足够的GPU内存
2. 首次运行建议先使用测试脚本验证功能
3. 可以根据需要调整渲染参数
4. 支持断点续传，可以安全地中断和恢复
5. 建议在处理大量数据前先处理少量样本进行测试

## 扩展性

脚本设计具有良好的扩展性：
- 可以轻松添加新的视角配置
- 可以支持更多的3D文件格式
- 可以添加更多的渲染选项
- 可以集成配置文件支持

