% 指定模型文件路径
filePath = 'F:/Temp/test/ModelNet40/airplane/train/airplane_0001.off';

% 加载网格模型
[mesh, ~, ~] = loadMesh(filePath);
if isempty(mesh)
    error('无法加载模型文件！');
end

% 定义所有视角的方位角和仰角
azimuths = [45, 45, -45, -45, 135, 135, 225, 225, 90, 90, -90, -90, 0, 0, 180, 180, 69.09, 110.91, -69.09, 249.09];
elevations = [35.26, -35.26, 35.26, -35.26, 35.26, -35.26, 35.26, -35.26, 69.09, -69.09, 69.09, -69.09, 20.91, -20.91, 20.91, -20.91, 0, 0, 0, 0];

% 绘制模型
figure('Position', [100, 100, 800, 600]);
trisurf(mesh.F', mesh.V(1,:), mesh.V(2,:), mesh.V(3,:), 'FaceColor', [0.8, 0.8, 0.8], 'EdgeColor', 'none');
axis equal;
axis vis3d;
grid on;
hold on;

% 绘制主轴方向（单位向量）
origin = [0, 0, 0];
mainAxisVector = [-0.0333, -0.9692, -0.2440];
quiver3(origin(1), origin(2), origin(3), mainAxisVector(1)*10, mainAxisVector(2)*10, mainAxisVector(3)*10, ...
    'Color', 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
text(mainAxisVector(1)*10, mainAxisVector(2)*10, mainAxisVector(3)*10, 'Main Axis', 'Color', 'r', 'FontSize', 12);

% 依次显示每个视角（按任意键切换）
for i = 1:length(azimuths)
    view(azimuths(i), elevations(i));
    title(sprintf('视角 %d - 方位角: %.2f°, 仰角: %.2f°', i, azimuths(i), elevations(i)));
    drawnow;
    pause; % 按任意键继续
end