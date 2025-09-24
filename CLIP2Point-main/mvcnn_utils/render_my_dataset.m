% 设置路径
addpath('D:/Pycharm/Replicate/SHREC2017_track3-master/mvcnn_utils');

% 设置输入输出路径
input_dir = 'F:\Temp\test\ModelNet40';
output_dir = 'F:\Temp\test\render_images';

% 检查输入目录是否存在
if ~exist(input_dir, 'dir')
    error('输入目录不存在: %s', input_dir);
end

% 获取所有类别文件夹
categories = dir(input_dir);
categories = categories([categories.isdir]);  % 只保留文件夹
categories = categories(~ismember({categories.name}, {'.', '..'}));  % 移除 . 和 ..

% 显示找到的类别
fprintf('找到的类别数量: %d\n', length(categories));
for i = 1:length(categories)
    fprintf('类别 %d: %s\n', i, categories(i).name);
end

% 遍历每个类别
for c = 1:length(categories)
    category = categories(c).name;
    fprintf('\n处理类别: %s\n', category);
    
    % 处理训练集和测试集
    for split = {'train', 'test'}
        split_name = split{1};
        
        % 设置输入输出路径
        current_input_dir = fullfile(input_dir, category, split_name);
        current_output_dir = fullfile(output_dir, category, split_name);
        
        % 检查输入目录是否存在
        if ~exist(current_input_dir, 'dir')
            fprintf('警告: 目录不存在: %s\n', current_input_dir);
            continue;
        end
        
        % 创建输出目录
        if ~exist(current_output_dir, 'dir')
            mkdir(current_output_dir);
        end
        
        % 获取当前目录下的所有.off文件
        model_files = dir(fullfile(current_input_dir, '*.off'));
        
        % 显示找到的模型数量
        fprintf('在 %s 中找到 %d 个模型文件\n', current_input_dir, length(model_files));
        
        % 遍历处理每个模型
        for i = 1:length(model_files)
            % 获取当前模型路径
            current_model = fullfile(current_input_dir, model_files(i).name);
            
            % 获取模型名称（不含扩展名）
            [~, model_name] = fileparts(model_files(i).name);
            
            % 严格检查是否所有视角都已渲染
            all_views_exist = true;
            missing_views = [];
            for v = 1:20  % 检查20个视角
                view_num = sprintf('%03d', v);
                view_file = fullfile(current_output_dir, sprintf('%s_%s.png', model_name, view_num));
                if ~exist(view_file, 'file')
                    all_views_exist = false;
                    missing_views = [missing_views, v];
                end
            end
            
            if all_views_exist
                fprintf('跳过已完全渲染的模型: %s\n', model_files(i).name);
                continue;
       
            end
            
            % 显示处理进度
            fprintf('正在处理: %s (%d/%d)\n', model_files(i).name, i, length(model_files));
            
            try
                % 渲染视图
                views = render_views(current_model, ...
                    'use_dodecahedron_views', true, ...  % 使用十二面体视角
                    'outputSize', 224, ...               % 输出图像大小
                    'colorMode', 'rgb');                 % 输出彩色图像
                
                % 保存渲染结果
                for v = 1:length(views)
                    % 使用三位数字格式化视角编号
                    view_num = sprintf('%03d', v);
                    output_file = fullfile(current_output_dir, ...
                        sprintf('%s_%s.png', model_name, view_num));
                    imwrite(views{v}, output_file);
                end
                
                fprintf('处理完成: %s\n', model_files(i).name);
            catch e
                fprintf('处理失败: %s\n错误信息: %s\n', model_files(i).name, e.message);
            end
        end
    end
end

fprintf('所有处理完成！\n');