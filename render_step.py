"""
Blender Rendering Script

Standalone script executed by Blender to render B-rep geometry.
Handles scene setup, materials, lighting, and camera configuration.
"""

import bpy
import numpy as np
import sys
import math
from pathlib import Path
from dataclasses import dataclass

# Color presets
COLOR_PRESETS = {
    "blue": [x / 255.0 for x in [135, 206, 235, 255]],
    "pink": [x / 255.0 for x in [255, 155, 188, 255]],
    "orange": [x / 255.0 for x in [255, 178, 111, 255]],
    "green": [x / 255.0 for x in [147, 197, 114, 255]],
    "silver": [x / 255.0 for x in [220, 220, 220, 255]],
}


@dataclass(frozen=True)
class Arguments:
    """
    Arguments for Blender rendering script.

    All values are provided by vis_step.py - no defaults here.
    """

    input_file: Path
    output_file: Path
    color: str
    rotation_angle: float
    resolution: int
    flip_z: bool
    stand_upright: bool
    color_mode: str
    camera_distance: float
    camera_height: float
    camera_base_angle: float


def arg_parser() -> Arguments:
    """
    Parse command-line arguments passed to Blender script.

    Expected format:
        blender --background --python render_step.py -- input.npz output.png color rotation_angle resolution [flags...]

    All parameters are required (provided by vis_step.py).

    Flags can be:
        - flip: Enable flip_z transformation
        - stand_upright: Enable stand_upright transformation
        - rgb/rgba: Color mode
        - camdist=X: Camera distance override
        - camheight=X: Camera height override
        - camangle=X: Camera base angle override

    Returns:
        Arguments dataclass with parsed values
    """
    parameters = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []

    if len(parameters) < 5:
        raise ValueError(
            "Usage: blender --background --python render_step.py -- "
            "input.npz output.png color rotation_angle resolution [flags...]"
        )

    # Required positional arguments (all provided by vis_step.py)
    input_file = Path(parameters[0])
    output_file = Path(parameters[1])
    color = parameters[2]
    rotation_angle = float(parameters[3])
    resolution = int(parameters[4])

    # Parse flags from position 5 onwards
    flags = parameters[5:] if len(parameters) > 5 else []
    flip_z = "flip" in flags
    stand_upright = "stand_upright" in flags
    color_mode = "rgba" if "rgba" in flags else "rgb"

    # Camera parameters (default to oblique/isometric view, can be overridden by flags)
    camera_distance = 3.5
    camera_height = 2.5
    camera_base_angle = -35.0

    # Parse camera parameter overrides from flags
    for flag in flags:
        if flag.startswith("camdist="):
            try:
                camera_distance = float(flag.split("=")[1])
            except ValueError:
                pass  # Invalid float, keep default
        elif flag.startswith("camheight="):
            try:
                camera_height = float(flag.split("=")[1])
            except ValueError:
                pass
        elif flag.startswith("camangle="):
            try:
                camera_base_angle = float(flag.split("=")[1])
            except ValueError:
                pass

    # Validate color preset
    if color not in COLOR_PRESETS:
        print(f"Warning: Unknown color '{color}', using 'blue'")
        color = "blue"

    return Arguments(
        input_file=input_file,
        output_file=output_file,
        color=color,
        rotation_angle=rotation_angle,
        resolution=resolution,
        flip_z=flip_z,
        stand_upright=stand_upright,
        color_mode=color_mode,
        camera_distance=camera_distance,
        camera_height=camera_height,
        camera_base_angle=camera_base_angle,
    )


def create_edge_object(edges, name="STEP_Edges"):
    curve_data = bpy.data.curves.new(name="EdgeCurve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 24
    curve_data.bevel_depth = 0.008
    curve_data.bevel_resolution = 1

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)

    for edge_points in edges:
        spline = curve_data.splines.new("NURBS")
        spline.points.add(len(edge_points) - 1)

        for i, point in enumerate(edge_points):
            spline.points[i].co = (*point, 1)

        spline.use_endpoint_u = True
        spline.order_u = 3

    return obj


def create_mesh_object(vertices, triangles, name="STEP_Mesh"):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], triangles.tolist())
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def create_edge_material():
    mat = bpy.data.materials.new(name="Edge_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")

    principled.inputs["Base Color"].default_value = [x / 255 for x in [25, 25, 25, 255]]
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular"].default_value = 0.0
    principled.inputs["Roughness"].default_value = 1.0

    mat.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return mat


def create_mesh_material(rgba: float, alpha: float = 1.0):
    mat = bpy.data.materials.new(name="Mesh_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes["Principled BSDF"]

    principled.inputs["Base Color"].default_value = rgba
    principled.inputs["Metallic"].default_value = 0.1
    principled.inputs["Roughness"].default_value = 0.8
    principled.inputs["Specular"].default_value = 0.1
    principled.inputs["Sheen"].default_value = 0.1

    if alpha < 1.0:
        mat.blend_method = "BLEND"
        principled.inputs["Alpha"].default_value = alpha
        mat.use_backface_culling = False

    return mat


def create_ground_plane():
    bpy.ops.mesh.primitive_plane_add(size=20)
    plane = bpy.context.active_object

    mat = bpy.data.materials.new(name="Ground_Material")
    mat.use_nodes = True
    principled = mat.node_tree.nodes["Principled BSDF"]
    principled.inputs["Base Color"].default_value = (1, 1, 1, 1)
    principled.inputs["Roughness"].default_value = 0.3

    plane.data.materials.append(mat)
    plane.is_shadow_catcher = True
    return plane


def set_white_background():
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGBA"
    bpy.context.scene.use_nodes = True

    compositor = bpy.context.scene.node_tree
    compositor.nodes.clear()

    render_layers = compositor.nodes.new(type="CompositorNodeRLayers")
    white_background = compositor.nodes.new(type="CompositorNodeRGB")
    white_background.outputs["RGBA"].default_value = (1, 1, 1, 1)
    alpha_over = compositor.nodes.new(type="CompositorNodeAlphaOver")
    composite_output = compositor.nodes.new(type="CompositorNodeComposite")

    compositor.links.new(render_layers.outputs["Image"], alpha_over.inputs[2])
    compositor.links.new(white_background.outputs["RGBA"], alpha_over.inputs[1])
    compositor.links.new(alpha_over.outputs["Image"], composite_output.inputs["Image"])

    bpy.context.scene.view_settings.view_transform = "Standard"


def create_lights():
    bpy.ops.object.light_add(type="AREA", location=(-1.5, -1.5, 2.5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 35.0
    fill_light.scale = (3, 3, 3)

    bpy.ops.object.light_add(type="AREA", location=(2, -1, 2))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 35.0
    rim_light.scale = (2, 2, 2)

    bpy.ops.object.light_add(type="SUN", location=(0, 0, 20))
    top_light = bpy.context.active_object
    top_light.data.energy = 0.7

    bpy.ops.object.light_add(type="POINT", location=(1, -3, 4))
    point_light = bpy.context.active_object
    point_light.data.energy = 350

    for light in bpy.data.lights:
        if light.type in ("SUN", "AREA"):
            light.shadow_soft_size = 0.3
            light.cycles.cast_shadow = False


def main():
    args = arg_parser()

    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for device in bpy.context.preferences.addons["cycles"].preferences.devices:
        if device.type in ("CUDA", "OPTIX"):
            device.use = True
    bpy.context.scene.cycles.device = "GPU"

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    data = np.load(args.input_file, allow_pickle=True)

    if "vertices" in data and "triangles" in data and "edges" in data:
        vertices = data["vertices"]
        triangles = data["triangles"]
        edges = data["edges"]
        mesh_obj = create_mesh_object(vertices, triangles)
        edge_obj = create_edge_object(edges)
    elif "edges" in data:
        edges = data["edges"]
        edge_obj = create_edge_object(edges)
        mesh_obj = None
    else:
        raise ValueError(
            "NPZ file must contain either 'vertices', 'triangles', and 'edges', or just 'edges'."
        )

    create_lights()

    if mesh_obj:
        edge_obj.location.z += 0.0002

    ground = create_ground_plane()
    ground.location.z = -0.5

    if mesh_obj:
        mesh_obj.data.materials.append(create_mesh_material(COLOR_PRESETS[args.color]))
    edge_obj.data.materials.append(create_edge_material())

    radius = args.camera_distance
    base_angle = math.radians(args.camera_base_angle)
    theta = math.radians(args.rotation_angle)
    total_angle = base_angle + theta

    cam_x = radius * math.cos(total_angle)
    cam_y = radius * math.sin(total_angle)
    cam_z = args.camera_height

    bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
    camera = bpy.context.active_object

    empty = bpy.data.objects.new("CameraTarget", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = (0, 0, 0)

    track = camera.constraints.new(type="TRACK_TO")
    track.target = empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.film_transparent = True
    scene.cycles.samples = 256

    # World settings for ambient lighting
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    mix = nodes.new("ShaderNodeMixShader")
    output = nodes.new("ShaderNodeOutputWorld")

    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1)
    bg.inputs["Strength"].default_value = 0.5
    ao.inputs["Distance"].default_value = 0.5
    ao.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1)

    world.node_tree.links.new(ao.outputs["Color"], mix.inputs[1])
    world.node_tree.links.new(bg.outputs["Background"], mix.inputs[2])
    world.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])

    if args.color_mode == "rgb":
        set_white_background()
    else:
        bpy.context.scene.view_settings.view_transform = "Standard"

    scene.render.filepath = str(args.output_file)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
