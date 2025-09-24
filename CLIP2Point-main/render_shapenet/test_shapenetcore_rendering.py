#!/usr/bin/env python3
"""
测试ShapeNetCore渲染脚本的功能
"""

import os
import sys
import argparse
from pathlib import Path

def test_directory_structure(input_path):
    """测试输入目录结构是否正确"""
    print("测试目录结构...")
    
    if not os.path.exists(input_path):
        print(f"错误：输入路径不存在: {input_path}")
        return False
    
    # 检查是否有类别目录
    categories = [d for d in os.listdir(input_path) 
                 if os.path.isdir(os.path.join(input_path, d))]
    
    if not categories:
        print(f"错误：在 {input_path} 中未找到类别目录")
        return False
    
    print(f"找到 {len(categories)} 个类别: {categories[:5]}...")
    
    # 检查第一个类别的结构
    first_category = categories[0]
    category_path = os.path.join(input_path, first_category)
    
    # 检查是否有子目录
    subdirs = [d for d in os.listdir(category_path) 
              if os.path.isdir(os.path.join(category_path, d))]
    
    if not subdirs:
        print(f"错误：类别 {first_category} 中没有子目录")
        return False
    
    print(f"类别 {first_category} 有 {len(subdirs)} 个子目录")
    
    # 检查第一个模型目录
    first_model = subdirs[0]
    model_path = os.path.join(category_path, first_model)
    
    # 检查模型文件
    obj_file = os.path.join(model_path, "model.obj")
    off_file = os.path.join(model_path, "model.off")
    
    if os.path.exists(obj_file):
        print(f"找到OBJ文件: {obj_file}")
        return True
    elif os.path.exists(off_file):
        print(f"找到OFF文件: {off_file}")
        return True
    else:
        print(f"错误：在 {model_path} 中未找到model.obj或model.off文件")
        return False

def test_single_model_rendering(input_path, output_path, gpu_id=0):
    """测试单个模型的渲染"""
    print("\n测试单个模型渲染...")
    
    # 导入必要的模块
    try:
        import torch
        from render.render import Renderer
        from render_shapenetcore_dodecahedron import process_model, load_obj_file, load_off_file
    except ImportError as e:
        print(f"导入错误: {e}")
        return False
    
    # 设置设备
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"使用GPU {gpu_id}")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    # 初始化渲染器
    try:
        renderer = Renderer(
            image_size=224,
            points_radius=0.02,
            points_per_pixel=1
        ).to(device)
        print("渲染器初始化成功")
    except Exception as e:
        print(f"渲染器初始化失败: {e}")
        return False
    
    # 找到第一个模型
    categories = [d for d in os.listdir(input_path) 
                 if os.path.isdir(os.path.join(input_path, d))]
    
    if not categories:
        print("未找到类别目录")
        return False
    
    category_id = categories[0]
    category_path = os.path.join(input_path, category_id)
    model_dirs = [d for d in os.listdir(category_path) 
                 if os.path.isdir(os.path.join(category_path, d))]
    
    if not model_dirs:
        print("未找到模型目录")
        return False
    
    model_id = model_dirs[0]
    model_path = os.path.join(category_path, model_id)
    
    print(f"测试模型: {category_id}-{model_id}")
    
    # 测试渲染
    try:
        success = process_model(model_path, output_path, renderer, device, category_id, model_id)
        if success:
            print("单个模型渲染测试成功！")
            return True
        else:
            print("单个模型渲染测试失败")
            return False
    except Exception as e:
        print(f"渲染测试出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='测试ShapeNetCore渲染功能')
    parser.add_argument('--input_path', type=str, required=True,
                        help='ShapeNetCore数据集路径')
    parser.add_argument('--output_path', type=str, default='./test_output',
                        help='测试输出路径')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("ShapeNetCore渲染功能测试")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 测试1：目录结构
    if not test_directory_structure(args.input_path):
        print("目录结构测试失败")
        return
    
    # 测试2：单个模型渲染
    if not test_single_model_rendering(args.input_path, args.output_path, args.gpu):
        print("单个模型渲染测试失败")
        return
    
    print("\n" + "=" * 50)
    print("所有测试通过！可以开始批量渲染")
    print("=" * 50)

if __name__ == '__main__':
    main()

