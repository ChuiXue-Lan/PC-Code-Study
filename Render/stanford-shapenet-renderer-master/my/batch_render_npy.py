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

def render_views(obj, cam_empty, output_prefix, views=30):
    scene = bpy.context.scene
    stepsize = 360.0 / views
    for i in range(views):
        render_file_path = f"{output_prefix}_r_{i:03d}"
        scene.render.filepath = render_file_path
        bpy.ops.render.render(write_still=True)
        cam_empty.rotation_euler[2] += math.radians(stepsize)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='npy文件所在目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出图片目录')
    parser.add_argument('--views', type=int, default=30, help='每个模型渲染视角数')
    parser.add_argument('--radius', type=float, default=0.01, help='点云小球半径')
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
        render_views(obj, cam_empty, output_prefix, views=args.views)

if __name__ == "__main__":
    main()