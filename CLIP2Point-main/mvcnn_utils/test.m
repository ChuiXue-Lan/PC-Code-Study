

% 测试代码
test_model = 'F:\Datasets\ModelNet40\ModelNet40\airplane\test\airplane_0627.off';
fprintf('测试第一个脚本的设置:\n');
views1 = render_views(test_model, 'use_dodecahedron_views', true);
fprintf('生成的视角数量: %d\n', length(views1));

fprintf('\n测试第二个脚本的设置:\n');
views2 = render_views(test_model, 'use_dodecahedron_views', true);
fprintf('生成的视角数量: %d\n', length(views2));