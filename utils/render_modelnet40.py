def get_dodecahedron_views():
    """生成十二面体的20个顶点作为视角，并转换为方位角和仰角（匹配MATLAB实现）"""
    phi = (1 + np.sqrt(5)) / 2
    points = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi],
        [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
    ])
    
    # 归一化到单位球面
    points = points / np.linalg.norm(points, axis=1)[:, None]
    
    # 将3D点转换为方位角和仰角（匹配MATLAB实现）
    azimuths = []
    elevations = []
    for point in points:
        x, y, z = point
        # 计算方位角（azimuth）：匹配MATLAB的计算方式
        azimuth = np.arctan2(y, x) * 180 / np.pi
        if x < 0:
            azimuth += 180
        
        # 计算仰角（elevation）：匹配MATLAB的计算方式
        elevation = np.arctan2(z, np.sqrt(x * x + y * y)) * 180 / np.pi
        
        azimuths.append(azimuth)
        elevations.append(elevation)
    
    return azimuths, elevations 