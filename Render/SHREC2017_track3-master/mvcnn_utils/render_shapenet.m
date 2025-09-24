% 设置路径
addpath('D:/Pycharm/Replicate/Render/SHREC2017_track3-master/mvcnn_utils');
% 设置输入输出路径
input_root = 'F:/Datasets/ShapeNet/ShapeNetCore/ShapeNetCore.v1.withoutimages'; % ShapeNetCore根目录
output_dir = 'F:/Datasets/ShapeNet/rendered_images';
% input_root = 'F:/Temp/test/ShapeNetCore'; % ShapeNetCore根目录
% output_dir = 'F:/Temp/test/rendered_images';
% temp_dir = 'F:/Temp/test/temp_off';  % 临时目录存放转换后的OFF文件

% 检查并创建临时目录
% if ~exist(temp_dir, 'dir')
%     mkdir(temp_dir);
% end

% 获取所有类别目录
cat_dirs = dir(input_root);
cat_dirs = cat_dirs([cat_dirs.isdir] & ~startsWith({cat_dirs.name}, '.'));
opengl('save','hardware')    
model_list = {};
for c = 1:length(cat_dirs)
    cat_name = cat_dirs(c).name; 
    cat_path = fullfile(input_root, cat_name, cat_name);
    if ~exist(cat_path, 'dir'), continue; end
    model_dirs = dir(cat_path);
    model_dirs = model_dirs([model_dirs.isdir] & ~startsWith({model_dirs.name}, '.'));
    for m = 1:length(model_dirs)
        model_id = model_dirs(m).name;
        obj_file = fullfile(cat_path, model_id, 'model.obj');
        if exist(obj_file, 'file')
            model_list{end+1,1} = cat_name;
            model_list{end,2} = model_id;
            model_list{end,3} = obj_file;
        end
    end
end
fprintf('找到 %d 个模型\n', size(model_list,1));

for i = 1:size(model_list,1)
    cat_name = model_list{i,1};
    model_id = model_list{i,2};
    obj_file = model_list{i,3};
    model_tag = [cat_name '-' model_id];
    fprintf('正在处理: %s (%d/%d)\n', model_tag, i, size(model_list,1));
    
    % 设置输出目录
    current_output_dir = fullfile(output_dir, model_tag);
    if ~exist(current_output_dir, 'dir')
        mkdir(current_output_dir);
    end
    
    % 检查是否所有视角都已渲染
    all_views_exist = true;
    for v = 1:20
        view_num = sprintf('%02d', v);
        view_file = fullfile(current_output_dir, sprintf('%s_view%s.png', model_tag, view_num));
        if ~exist(view_file, 'file')
            all_views_exist = false;
            break;
        end
    end
    if all_views_exist
        fprintf('跳过已完全渲染的模型: %s\n', model_tag);
        continue;
    end
    try
        % 直接渲染OBJ文件
        views = render_views(obj_file, ...
            'use_dodecahedron_views', true, ...
            'outputSize', 224, ...
            'colorMode', 'rgb');
        for v = 1:length(views)
            view_num = sprintf('%02d', v);
            output_file = fullfile(current_output_dir, ...
                sprintf('%s_view%s.png', model_tag, view_num));
            imwrite(views{v}, output_file);
        end
        fprintf('处理完成: %s\n', model_tag);
    catch e
        fprintf('处理失败: %s\n错误信息: %s\n', model_tag, e.message);
    end
end
fprintf('所有处理完成！\n');