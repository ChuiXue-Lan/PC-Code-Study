import argparse, sys, os, math
import bpy
import numpy as np
from glob import glob
from tqdm import tqdm

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def import_pointcloud_as_particles(points):
    # 创建一个平面作为粒子发射体
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    emitter = bpy.context.active_object
    # 添加粒子系统
    ps = emitter.modifiers.new("particles", type='PARTICLE_SYSTEM')
    psettings = ps.particle_system.settings
    psettings.count = len(points)
    psettings.frame_start = 1
    psettings.frame_end = 1
    psettings.lifetime = 250
    psettings.emit_from = 'VERT'
    psettings.physics_type = 'NO'
    psettings.render_type = 'OBJECT'
    # 创建一个小球作为粒子对象
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=(0, 0, 0))
    sphere = bpy.context.active_object
    sphere.name = "ParticleSphere"
    psettings.instance_object = sphere
    # 设置粒子位置
    # 先将发射体顶点数设置为点数
    bpy.ops.object.select_all(action='DESELECT')
    emitter.select_set(True)
    bpy.context.view_layer.objects.active = emitter
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    # 新建顶点
    mesh = emitter.data
    for pt in points:
        mesh.vertices.add(1)
        mesh.vertices[-1].co = (float(pt[0]), float(pt[1]), float(pt[2]))
    return emitter, sphere

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

# 复用setup_output_nodes和render_views与原脚本一致
from batch_render_npy import setup_output_nodes, render_views

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
    for npy_path in tqdm(npy_files, desc="Batch Rendering npy files (particles)"):
        clear_scene()
        points = np.load(npy_path)
        emitter, sphere = import_pointcloud_as_particles(points)
        cam_empty = setup_camera_light()
        model_name = os.path.splitext(os.path.basename(npy_path))[0]
        output_prefix = os.path.join(args.output_dir, model_name, model_name)
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        depth_file_output, normal_file_output, albedo_file_output, id_file_output = setup_output_nodes(args)
        render_views(emitter, cam_empty, output_prefix, args.views, depth_file_output, normal_file_output, albedo_file_output, id_file_output)
        # 渲染后删除粒子球体，避免下一个模型重复
        bpy.data.objects.remove(sphere, do_unlink=True)

if __name__ == "__main__":
    main()
