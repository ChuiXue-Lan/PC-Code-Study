function mesh = loadMesh_obj(filename)
% 只适配 .obj 文件，返回 mesh 结构体，含 V, F, Nv

file = fopen(filename, 'rt');
if file == -1
    error(['无法打开文件: ' filename]);
end

V = [];
F = [];
Nv = [];
while ~feof(file)
    line = strtrim(fgetl(file));
    if startsWith(line, 'v ')
        V = [V; sscanf(line(3:end), '%f %f %f')'];
    elseif startsWith(line, 'vn ')
        Nv = [Nv; sscanf(line(4:end), '%f %f %f')'];
    elseif startsWith(line, 'f ')
        face = sscanf(line(3:end), '%d//%d %d//%d %d//%d');
        if isempty(face)
            face = sscanf(line(3:end), '%d %d %d');
        end
        F = [F; face(1:2:end)'];
    end
end
fclose(file);

mesh.V = V';
mesh.F = F';
if ~isempty(Nv)
    mesh.Nv = Nv';
end
end