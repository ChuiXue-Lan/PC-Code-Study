import torch
import argparse
from collections import OrderedDict
import numpy as np

def analyze_tensor(tensor):
    """分析张量的基本信息"""
    return {
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'mean': float(tensor.float().mean()) if tensor.numel() > 0 else 'N/A',
        'std': float(tensor.float().std()) if tensor.numel() > 0 else 'N/A',
        'min': float(tensor.float().min()) if tensor.numel() > 0 else 'N/A',
        'max': float(tensor.float().max()) if tensor.numel() > 0 else 'N/A'
    }

def analyze_state_dict(state_dict, prefix=''):
    """递归分析状态字典的结构"""
    info = OrderedDict()
    for key, value in state_dict.items():
        if isinstance(value, (dict, OrderedDict)):
            info[key] = analyze_state_dict(value, prefix + '  ')
        elif isinstance(value, torch.Tensor):
            info[key] = analyze_tensor(value)
        else:
            info[key] = f"Type: {type(value)}, Value: {str(value)}"
    return info

def print_dict(d, indent=0):
    """格式化打印字典内容"""
    for key, value in d.items():
        print('  ' * indent + str(key) + ':')
        if isinstance(value, dict):
            print_dict(value, indent + 1)
        else:
            print('  ' * (indent + 1) + str(value))

def main():
    parser = argparse.ArgumentParser(description='Analyze PyTorch checkpoint file')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint file')
    parser.add_argument('--save_analysis', action='store_true', default=True, help='Save analysis to a text file')
    args = parser.parse_args()

    print(f"\n正在加载checkpoint文件: {args.checkpoint_path}")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"加载checkpoint时出错: {str(e)}")
        return

    print("\n=== Checkpoint 基本信息 ===")
    print(f"包含的键: {list(checkpoint.keys())}")
    
    if 'epoch' in checkpoint:
        print(f"\n训练轮数: {checkpoint['epoch']}")
    
    print("\n=== 详细分析 ===")
    analysis = {}
    
    # 分析模型状态
    if 'model' in checkpoint:
        print("\n模型状态分析:")
        model_keys = list(checkpoint['model'].keys())
        print(f"模型参数数量: {len(model_keys)}")
        print(f"参数键名示例 (前5个): {model_keys[:5]}")
        analysis['model'] = analyze_state_dict(checkpoint['model'])
    
    # 分析优化器状态
    if 'optimizer' in checkpoint:
        print("\n优化器状态分析:")
        optimizer_state = checkpoint['optimizer']
        print(f"优化器状态键: {list(optimizer_state.keys())}")
        analysis['optimizer'] = analyze_state_dict(optimizer_state)
    
    # 分析学习率调度器状态
    if 'scheduler' in checkpoint:
        print("\n学习率调度器状态分析:")
        scheduler_state = checkpoint['scheduler']
        print(f"调度器状态键: {list(scheduler_state.keys())}")
        analysis['scheduler'] = analyze_state_dict(scheduler_state)
    
    # 分析EMA状态
    if 'model_ema' in checkpoint:
        print("\nEMA模型状态分析:")
        ema_state = checkpoint['model_ema']
        print(f"EMA状态键: {list(ema_state.keys())}")
        analysis['model_ema'] = analyze_state_dict(ema_state)
    
    # 分析混合精度训练状态
    if 'scaler' in checkpoint:
        print("\n混合精度训练状态分析:")
        scaler_state = checkpoint['scaler']
        print(f"Scaler状态键: {list(scaler_state.keys())}")
        analysis['scaler'] = analyze_state_dict(scaler_state)
    
    # 分析其他信息
    other_keys = [k for k in checkpoint.keys() if k not in 
                  ['model', 'optimizer', 'scheduler', 'model_ema', 'scaler', 'epoch']]
    if other_keys:
        print("\n其他信息:")
        for key in other_keys:
            print(f"{key}: {type(checkpoint[key])}")
            if not isinstance(checkpoint[key], torch.Tensor):
                print(f"值: {checkpoint[key]}")
            analysis[key] = analyze_state_dict({key: checkpoint[key]})

    if args.save_analysis:
        output_file = args.checkpoint_path + '.analysis.txt'
        print(f"\n保存分析结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            def write_dict(d, indent=0):
                for key, value in d.items():
                    f.write('  ' * indent + str(key) + ':\n')
                    if isinstance(value, dict):
                        write_dict(value, indent + 1)
                    else:
                        f.write('  ' * (indent + 1) + str(value) + '\n')
            write_dict(analysis)

    print("\n分析完成!")

if __name__ == '__main__':
    main() 