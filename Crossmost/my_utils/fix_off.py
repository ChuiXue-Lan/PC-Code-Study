import os

def fix_off_header(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if not lines[0].strip().startswith('OFF'):
        return False
    # 如果第一行是 'OFF3102 4146 0'
    if lines[0].startswith('OFF') and len(lines[0].split()) == 1:
        return False  # 正常的 OFF 文件
    # 拆分头部数据
    header = lines[0][3:].strip()  # 去掉 'OFF'
    new_lines = ['OFF\n', header + '\n'] + lines[1:]
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    return True

def fix_folder(folder):
    fixed = 0
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.off'):
                path = os.path.join(root, file)
                if fix_off_header(path):
                    print(f"Fixed: {path}")
                    fixed += 1
    print(f"修复完成，共修复 {fixed} 个文件。")

# 示例调用
if __name__ == '__main__':
    fix_folder('/home/cls2024/ltx/Replicate/CrossMoST-main/data/modelnet40_rendered/ModelNet40_mini/')