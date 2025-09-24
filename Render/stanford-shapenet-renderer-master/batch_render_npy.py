import argparse, sys, os, math
import bpy
import numpy as np
from glob import glob
from tqdm import tqdm

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_pointcloud_as_spheres(points, radius=0.01):
    print(f"点云点数: {len(points)}")
    spheres = []
    for idx, pt in enumerate(points):
        if idx % 100 == 0:
            print(f"正在生成第{idx}个球体")
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(float(pt[0]), float(pt[1]), float(pt[2])))
        spheres.append(bpy.context.active_object)
    # 合并所有小球为一个对象
    print("开始合并球体")
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for s in spheres:
        s.select_set(True)
    bpy.context.view_layer.objects.active = spheres[0]
    bpy.ops.object.join()
    print("合并完成")
    return bpy.context.active_object

def setup_camera_light():
    scene = bpy.context.scene
    cam = scene.objects['Camera']
    cam.location = (0, 1, 0.6)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    cam.parent = cam_empty
    scene.collection.objects.link(cam_empty)
    bpy.context.view_layer.objects.active = cam_empty
    cam_constraint.target = cam_empty
    # 灯光
    light = bpy.data.lights['Light']
    light.type = 'SUN'
    light.use_shadow = False
    light.specular_factor = 1.0
    light.energy = 10.0
    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.data.lights['Sun']
    light2.use_shadow = False
    light2.specular_factor = 1.0
    light2.energy = 0.015
    bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
    bpy.data.objects['Sun'].rotation_euler[0] += 180
    return cam_empty

def setup_output_nodes(args):
    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    render_layers = nodes.new('CompositorNodeRLayers')
    # 深度输出
    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = ''
    depth_file_output.file_slots[0].use_node_format = True
    depth_file_output.format.file_format = args.format
    depth_file_output.format.color_depth = args.color_depth
    if args.format == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        depth_file_output.format.color_mode = "BW"
        map = nodes.new(type="CompositorNodeMapValue")
        map.offset = [-0.7]
        map.size = [args.depth_scale]
        map.use_min = True
        map.min = [0]
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
    # 法线输出
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = 'MULTIPLY'
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs['Normal'], scale_node.inputs[1])
    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = 'ADD'
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])
    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.base_path = ''
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = args.format
    links.new(bias_node.outputs[0], normal_file_output.inputs[0])
    # Albedo输出
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
    links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = 'Albedo Output'
    albedo_file_output.base_path = ''
    albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = args.format
    albedo_file_output.format.color_mode = 'RGBA'
    albedo_file_output.format.color_depth = args.color_depth
    links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
    # ID输出
    id_file_output = nodes.new(type="CompositorNodeOutputFile")
    id_file_output.label = 'ID Output'
    id_file_output.base_path = ''
    id_file_output.file_slots[0].use_node_format = True
    id_file_output.format.file_format = args.format
    id_file_output.format.color_depth = args.color_depth
    if args.format == 'OPEN_EXR':
        links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])
    else:
        id_file_output.format.color_mode = 'BW'
        divide_node = nodes.new(type='CompositorNodeMath')
        divide_node.operation = 'DIVIDE'
        divide_node.use_clamp = False
        divide_node.inputs[1].default_value = 2**int(args.color_depth)
        links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
        links.new(divide_node.outputs[0], id_file_output.inputs[0])
    return depth_file_output, normal_file_output, albedo_file_output, id_file_output

def render_views(obj, cam_empty, output_prefix, views, depth_file_output, normal_file_output, albedo_file_output, id_file_output):
    scene = bpy.context.scene
    stepsize = 360.0 / views
    for i in range(views):
        render_file_path = f"{output_prefix}_r_{i:03d}"
        scene.render.filepath = render_file_path
        depth_file_output.file_slots[0].path = render_file_path + "_depth"
        normal_file_output.file_slots[0].path = render_file_path + "_normal"
        albedo_file_output.file_slots[0].path = render_file_path + "_albedo"
        id_file_output.file_slots[0].path = render_file_path + "_id"
        bpy.ops.render.render(write_still=True)
        cam_empty.rotation_euler[2] += math.radians(stepsize)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='npy文件所在目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图片目录')
    parser.add_argument('--views', type=int, default=30, help='每个模型渲染视角数')
    parser.add_argument('--radius', type=float, default=0.01, help='点云小球半径')
    parser.add_argument('--format', type=str, default='PNG', help='输出图片格式PNG/OPEN_EXR')
    parser.add_argument('--color_depth', type=str, default='8', help='输出色深8/16')
    parser.add_argument('--depth_scale', type=float, default=1.4, help='深度缩放')
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    npy_files = glob(os.path.join(args.input_dir, "*.npy"))
    for npy_path in tqdm(npy_files, desc="Batch Rendering npy files"):
        clear_scene()
        points = np.load(npy_path)
        obj = import_pointcloud_as_spheres(points, radius=args.radius)
        cam_empty = setup_camera_light()
        model_name = os.path.splitext(os.path.basename(npy_path))[0]
        output_prefix = os.path.join(args.output_dir, model_name, model_name)
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        depth_file_output, normal_file_output, albedo_file_output, id_file_output = setup_output_nodes(args)
        render_views(obj, cam_empty, output_prefix, args.views, depth_file_output, normal_file_output, albedo_file_output, id_file_output)

if __name__ == "__main__":
    main() 