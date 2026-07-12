import argparse
import os
import sys
import math
import bpy
import mathutils


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)


def compute_bounds():
    min_v = mathutils.Vector((1e9, 1e9, 1e9))
    max_v = mathutils.Vector((-1e9, -1e9, -1e9))
    has_mesh = False
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        has_mesh = True
        for v in obj.bound_box:
            world_v = obj.matrix_world @ mathutils.Vector(v)
            min_v.x = min(min_v.x, world_v.x)
            min_v.y = min(min_v.y, world_v.y)
            min_v.z = min(min_v.z, world_v.z)
            max_v.x = max(max_v.x, world_v.x)
            max_v.y = max(max_v.y, world_v.y)
            max_v.z = max(max_v.z, world_v.z)
    if not has_mesh:
        return mathutils.Vector((0, 0, 0)), 1.0
    center = (min_v + max_v) * 0.5
    extent = (max_v - min_v).length
    return center, extent


def ensure_camera():
    cam = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            cam = obj
            break
    if cam is None:
        cam_data = bpy.data.cameras.new(name="Camera")
        cam = bpy.data.objects.new("Camera", cam_data)
        bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def ensure_light():
    light = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            light = obj
            break
    if light is None:
        light_data = bpy.data.lights.new(name="Sun", type='SUN')
        light = bpy.data.objects.new("Sun", light_data)
        bpy.context.scene.collection.objects.link(light)
    return light


def look_at(camera, target, up_axis='Z'):
    """
    Point camera at target

    Args:
        camera: Blender camera object
        target: mathutils.Vector target position
        up_axis: 'Z' for Z-up scenes (default), 'Y' for Y-up
    """
    direction = target - camera.location
    rot_quat = direction.to_track_quat('-Z', up_axis)
    camera.rotation_euler = rot_quat.to_euler()


def render_views(output_dir, views):
    os.makedirs(output_dir, exist_ok=True)

    center, extent = compute_bounds()
    if extent <= 0:
        extent = 1.0

    view_positions = {
        "front": mathutils.Vector((center.x, center.y - extent * 1.5, center.z + extent * 0.3)),
        "top": mathutils.Vector((center.x, center.y, center.z + extent * 2.0)),
        "perspective": mathutils.Vector((center.x + extent * 1.0, center.y - extent * 1.0, center.z + extent * 0.8)),
    }

    scene = bpy.context.scene
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    # Blender 4.x uses EEVEE_NEXT
    available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    if "BLENDER_EEVEE_NEXT" in available_engines:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in available_engines:
        scene.render.engine = "BLENDER_EEVEE"
    else:
        scene.render.engine = "CYCLES"
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    cam = ensure_camera()
    light = ensure_light()

    for view in views:
        if view not in view_positions:
            continue

        # Position camera
        cam.location = view_positions[view]
        look_at(cam, center, up_axis='Y')  # Scene is rotated to Y-up for rendering

        # Position light relative to camera (avoid shadows blocking view)
        # Place light above and to the side of camera
        light.location = cam.location + mathutils.Vector((extent * 0.5, 0, extent * 1.0))

        # Render
        scene.render.filepath = os.path.join(output_dir, f"{view}.png")
        bpy.ops.render.render(write_still=True)
        if not os.path.exists(scene.render.filepath):
            print(f"[WARN] Render output missing: {scene.render.filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glb", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--views", default="front,top,perspective")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)

    clear_scene()
    import_glb(args.input_glb)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    render_views(args.output_dir, views)


if __name__ == "__main__":
    main()
