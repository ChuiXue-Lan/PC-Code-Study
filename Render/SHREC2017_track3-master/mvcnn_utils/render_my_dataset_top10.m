% 设置路径
addpath('D:/Pycharm/Replicate/SHREC2017_track3-master/mvcnn_utils');

% 设置输入输出路径
input_dir = 'F:/Datasets/ModelNet40/ModelNet40';
output_dir = 'F:/Replicate/CrossMoST/render_images_mini';

% 检查输入目录
if ~exist(input_dir, 'dir')
    error('输入目录不存在: %s', input_dir);
end

% 创建主输出目录
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 获取所有类别文件夹
categories = dir(input_dir);
categories = categories([categories.isdir]);
categories = categories(~ismember({categories.name}, {'.', '..'}));

% 显示类别信息
fprintf('找到的类别数量: %d\n', length(categories));
for i = 1:length(categories)
    fprintf('类别 %d: %s\n', i, categories(i).name);
end

% 遍历每个类别
for c = 1:length(categories)
    category = categories(c).name;
    fprintf('\n开始处理类别: %s\n', category);
    
    % 处理train和test子目录
    for split = {'train', 'test'}
        split_name = split{1};
        
        % 设置当前处理的目录路径
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
        
        % 获取.off文件
        model_files = dir(fullfile(current_input_dir, '*.off'));
        
        % 确定要处理的文件数量（最多10个）
        num_files_to_process = min(10, length(model_files));
        
        fprintf('在 %s 中处理前 %d 个文件（共有 %d 个文件）\n', ...
            current_input_dir, num_files_to_process, length(model_files));
        
        % 只处理前10个文件
        for i = 1:num_files_to_process
            % 获取当前模型信息
            current_model = fullfile(current_input_dir, model_files(i).name);
            [~, model_name] = fileparts(model_files(i).name);
            
            % 检查所有视角文件是否都已存在
            all_views_exist = true;
            missing_views = [];
            for v = 1:12  % 假设有12个视角
                view_num = sprintf('%03d', v);
                view_file = fullfile(current_output_dir, ...
                    sprintf('%s_%s.png', model_name, view_num));
                if ~exist(view_file, 'file')
                    all_views_exist = false;
                    missing_views = [missing_views, v];
                end
            end
            
            % 如果所有视角都已存在，跳过此模型
            if all_views_exist
                fprintf('跳过已完全渲染的模型: %s\n', model_files(i).name);
                continue;
            
            end
            
            fprintf('正在处理 %d/%d: %s\n', i, num_files_to_process, model_files(i).name);
            
            try
                % 渲染模型
                views = render_views(current_model, ...
                    'use_dodecahedron_views', true, ...
                    'outputSize', 224, ...
                    'colorMode', 'rgb');
                
                % 保存渲染结果
                for v = 1:length(views)
                    view_num = sprintf('%03d', v);
                    output_file = fullfile(current_output_dir, ...
                        sprintf('%s_%s.png', model_name, view_num));
                    
                    % 检查文件是否已存在
                    if ~exist(output_file, 'file')
                        imwrite(views{v}, output_file);
                    else
                        fprintf('视角 %s 已存在，跳过\n', view_num);
                    end
                end
                
                fprintf('完成渲染: %s\n', model_files(i).name);
                
            catch e
                fprintf('渲染失败 %s: %s\n', model_files(i).name, e.message);
            end
        end
        
        fprintf('完成 %s/%s 目录的处理\n', category, split_name);
    end
end

fprintf('\n所有渲染任务完成！\n');
fprintf('渲染结果保存在: %s\n', output_dir);