"""
BrepVis: B-rep Visualization Tool

Main CLI application for rendering and viewing CAD B-rep models in STEP format.
Provides commands for static rendering, video generation, and interactive viewing.
"""

import numpy as np
import subprocess
import sys
import tempfile
import trimesh
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import typer

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Compound

from vis_util import (
    # Geometry functions
    extract_faces,
    apply_transformations,
    normalize_vertices,
    offset_edges_from_surface,
    # Edge packing (pickle-free NPZ storage)
    pack_edges,
    unpack_edges,
    # Helper functions
    _read_step_file,
    _count_geometry,
    _discretize_all_edges,
    _build_partial_compound,
    _setup_directories,
    _discover_step_files,
    _render_video_frames,
    _create_video_with_ffmpeg,
)

app = typer.Typer(pretty_exceptions_enable=False)


# =============================================================================
# Default Parameters
# =============================================================================

# Mesh Quality Defaults
DEFAULT_EDGE_DEFLECTION = 0.0005
DEFAULT_MESH_LINEAR_DEFLECTION = 0.01
DEFAULT_MESH_ANGULAR_DEFLECTION = 0.1

# Fast Mode Quality (10x coarser for speed)
FAST_EDGE_DEFLECTION = 0.005
FAST_MESH_LINEAR_DEFLECTION = 0.1
FAST_MESH_ANGULAR_DEFLECTION = 1.0

# Rendering Defaults
DEFAULT_RESOLUTION = 256
DEFAULT_COLOR = "blue"
DEFAULT_COLOR_MODE = "rgb"
DEFAULT_ROTATION_ANGLE = 0.0

# Camera Defaults (oblique/isometric view)
DEFAULT_CAMERA_DISTANCE = 3.5
DEFAULT_CAMERA_HEIGHT = 2.5
DEFAULT_CAMERA_BASE_ANGLE = -35.0

# Video Defaults
DEFAULT_VIDEO_FPS = 24
DEFAULT_VIDEO_DURATION = 3.0
DEFAULT_VIDEO_FORMAT = "mp4"

# Execution Defaults
DEFAULT_MAX_WORKERS = 4
DEFAULT_N_STEPS = 1000

# Ground Plane Positioning
GROUND_PLANE_Z = -0.5  # Z coordinate for ground plane after normalization

# Blender Configuration
BLENDER_EXECUTABLE = "./blender"
BLENDER_RENDER_SCRIPT = "render_step.py"

# Color Presets (kept in sync with render_step.py)
COLOR_PRESETS = {
    "blue": [x / 255.0 for x in [135, 206, 235, 255]],
    "pink": [x / 255.0 for x in [255, 155, 188, 255]],
    "orange": [x / 255.0 for x in [255, 178, 111, 255]],
    "green": [x / 255.0 for x in [147, 197, 114, 255]],
    "silver": [x / 255.0 for x in [220, 220, 220, 255]],
}


# =============================================================================
# Processing Functions
# =============================================================================


def _transform_and_normalize(vertices, edges, stand_upright, flip_z, no_normalize, ground_plane_z=GROUND_PLANE_Z, normalization_params=None):
    """
    Apply transformations and normalization to vertices and edges.

    Args:
        vertices: Nx3 array of mesh vertices
        edges: List of Mx3 arrays of edge points
        stand_upright: Whether to rotate 90° around X
        flip_z: Whether to flip along Z axis
        no_normalize: Whether to skip normalization
        ground_plane_z: Z position for ground plane after normalization
        normalization_params: Optional dict with pre-computed 'center', 'scale', 'z_shift'
                            If provided, uses these instead of computing new ones

    Returns:
        Tuple of (vertices, edges_rescaled, norm_params) where:
        - vertices: Normalized mesh vertices
        - edges_rescaled: Normalized edge points
        - norm_params: Dict with 'center', 'scale', 'z_shift' for reuse
    """
    # Apply transformations to vertices BEFORE normalization
    vertices = apply_transformations(vertices, stand_upright, flip_z)

    # Apply same transformations to edges BEFORE normalization
    edges_transformed = []
    for edge in edges:
        edge = apply_transformations(edge, stand_upright, flip_z)
        edges_transformed.append(edge)

    # Normalize vertices and get/use transform parameters
    if not no_normalize:
        if normalization_params is not None:
            # Use pre-computed normalization parameters
            center = normalization_params['center']
            scale = normalization_params['scale']
            z_shift = normalization_params['z_shift']
        else:
            # Compute normalization parameters from this geometry
            v_min, v_max = vertices.min(axis=0, keepdims=True), vertices.max(
                axis=0, keepdims=True
            )
            center = ((v_max + v_min) / 2).flatten()
            scale = np.max(v_max - v_min)

            # Compute z_shift based on normalized geometry
            vertices_normalized = (vertices - center) / scale
            z_min = vertices_normalized[:, 2].min()
            z_shift = z_min - ground_plane_z

        # Apply normalization (only if there are vertices)
        if len(vertices) > 0:
            vertices = (vertices - center) / scale
            vertices[:, 2] -= z_shift
    else:
        scale = 1.0
        center = np.zeros(3)
        z_shift = 0.0

    # Apply same normalization to edges
    edges_rescaled = []
    for edge in edges_transformed:
        if not no_normalize:
            edge = (edge - center) / scale
            edge[:, 2] -= z_shift
        edges_rescaled.append(edge)

    # Return normalization params for potential reuse
    norm_params = {'center': center, 'scale': scale, 'z_shift': z_shift}
    return vertices, edges_rescaled, norm_params


def process_step_file(
    step_file: Path,
    output_dir: Path,
    edge_deflection: float,
    mesh_linear_deflection: float,
    mesh_angular_deflection: float,
    stand_upright: bool,
    flip_z: bool,
    no_normalize: bool,
    partial_faces: list[int] = None,
    verbose: bool = False,
    ground_plane_z: float = GROUND_PLANE_Z,
    normalization_params: dict = None,
) -> Path:
    """
    Process a STEP file to NPZ format with edges and mesh.

    Args:
        step_file: Path to input STEP file
        output_dir: Directory for output NPZ file
        edge_deflection: Deflection tolerance for edge discretization
        mesh_linear_deflection: Linear deflection for mesh tessellation
        mesh_angular_deflection: Angular deflection for mesh tessellation
        stand_upright: Whether to rotate 90° around X axis
        flip_z: Whether to flip along Z axis
        no_normalize: Whether to skip normalization
        partial_faces: Optional list of face indices to process (if None, process all)
        verbose: Whether to print detailed information
        ground_plane_z: Z position for ground plane after normalization
        normalization_params: Optional pre-computed normalization parameters

    Returns:
        Path to output NPZ file, or None if processing failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if partial_faces is not None and len(partial_faces) > 0:
        brep_npz_path = output_dir / f"{step_file.stem}_partial_{partial_faces}.npz"
    else:
        suffix = "_flip" if flip_z else ""
        brep_npz_path = output_dir / f"{step_file.stem}{suffix}.npz"

    # Read STEP file
    shape = _read_step_file(step_file)
    if shape is None:
        return None

    # If partial faces specified, build compound with only those faces
    if partial_faces is not None and len(partial_faces) > 0:
        shape = _build_partial_compound(shape, partial_faces)

    # Print basic info if verbose
    if verbose:
        face_count, edge_count = _count_geometry(shape)
        print(f"[{step_file.name}] Faces: {face_count}, Edges: {edge_count}")

    # Discretize all edges
    all_edges = _discretize_all_edges(shape, edge_deflection)

    # Extract mesh
    vertices, triangles = extract_faces(
        shape, mesh_linear_deflection, mesh_angular_deflection
    )

    # Apply transformations and normalization consistently
    vertices, all_edges_rescaled, norm_params = _transform_and_normalize(
        vertices, all_edges, stand_upright, flip_z, no_normalize, ground_plane_z, normalization_params
    )

    # Offset edges from surface to prevent z-fighting (only for full processing)
    if partial_faces is None or len(partial_faces) == 0:
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        all_edges_rescaled = offset_edges_from_surface(all_edges_rescaled, mesh)

    # Save to NPZ file (pickle-free: edges stored as flat arrays + offsets)
    edge_points, edge_offsets = pack_edges(all_edges_rescaled)
    np.savez(
        brep_npz_path,
        edge_points=edge_points,
        edge_offsets=edge_offsets,
        vertices=vertices,
        triangles=triangles,
        scale=norm_params['scale'],
        center=norm_params['center'],
    )

    return brep_npz_path


def explode_step_file(
    step_file: Path,
    output_dir: Path,
    edge_deflection: float,
    mesh_linear_deflection: float,
    mesh_angular_deflection: float,
    stand_upright: bool,
    flip_z: bool,
    no_normalize: bool,
    blender_script: Path,
    color: str,
    rotation_angle: float,
    color_mode: str,
    camera_distance: float = None,
    camera_height: float = None,
    camera_base_angle: float = None,
    ground_plane_z: float = GROUND_PLANE_Z,
):
    """
    Create exploded view renderings of a STEP file.

    For each face, generates three renders:
    1. Face with mesh and all its edges
    2. Edge loop only (no mesh)
    3. Individual edges separately

    Args:
        step_file: Path to STEP file
        output_dir: Base output directory (creates exploded/ subdirectory)
        edge_deflection: Edge discretization tolerance
        mesh_linear_deflection: Mesh linear deflection
        mesh_angular_deflection: Mesh angular deflection
        stand_upright: Whether to rotate 90° around X
        flip_z: Whether to flip along Z axis
        no_normalize: Whether to skip normalization
        blender_script: Path to Blender rendering script
        color: Color preset name
        rotation_angle: Camera rotation angle in degrees
        color_mode: Color mode (rgb/rgba)
    """
    exploded_dir = output_dir / "exploded"
    exploded_dir.mkdir(parents=True, exist_ok=True)

    # Read STEP file
    shape = _read_step_file(step_file)
    if shape is None:
        return

    # Pre-compute normalization parameters from the full model
    # This ensures all faces use the same coordinate space
    normalization_params = None
    if not no_normalize:
        # Extract full model geometry
        full_edges = _discretize_all_edges(shape, edge_deflection)
        full_vertices, full_triangles = extract_faces(
            shape, mesh_linear_deflection, mesh_angular_deflection
        )

        # Compute normalization parameters (but don't save the full model)
        _, _, normalization_params = _transform_and_normalize(
            full_vertices, full_edges, stand_upright, flip_z, False, ground_plane_z
        )

    # Process each face
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0

    while face_explorer.More():
        face = face_explorer.Current()

        # Render the face as mesh with edges (using partial processing)
        # Use pre-computed normalization params so all faces are in same coordinate space
        npz_path = process_step_file(
            step_file,
            exploded_dir,
            edge_deflection,
            mesh_linear_deflection,
            mesh_angular_deflection,
            stand_upright,
            flip_z,
            no_normalize=no_normalize,  # Respect user's no_normalize setting
            partial_faces=[face_idx],
            ground_plane_z=ground_plane_z,
            normalization_params=normalization_params,  # Use pre-computed params
        )

        render_path = render_blender_step(
            npz_path,
            exploded_dir,
            blender_script,
            color,
            rotation_angle,
            flip_z,
            stand_upright,
            color_mode,
            DEFAULT_RESOLUTION,
            camera_distance,
            camera_height,
            camera_base_angle,
        )
        if render_path:
            render_path.rename(exploded_dir / f"face_{face_idx}.png")

        # Clean up NPZ file after rendering
        if npz_path and npz_path.exists():
            npz_path.unlink()

        # Render edge loop (all edges of this face, no mesh)
        edge_loop_edges = _discretize_all_edges(face, edge_deflection)

        # Apply same transformations and normalization as face mesh
        _, edge_loop_edges_normalized, _ = _transform_and_normalize(
            np.zeros((0, 3)),  # No vertices for edge-only rendering
            edge_loop_edges,
            stand_upright,
            flip_z,
            no_normalize,
            ground_plane_z,
            normalization_params,  # Use same normalization as face mesh
        )

        # Save and render edge loop (pickle-free)
        edge_loop_npz = exploded_dir / f"face_{face_idx}_loop.npz"
        edge_loop_points, edge_loop_offsets = pack_edges(edge_loop_edges_normalized)
        np.savez(
            edge_loop_npz,
            edge_points=edge_loop_points,
            edge_offsets=edge_loop_offsets,
        )

        render_path = render_blender_step(
            edge_loop_npz,
            exploded_dir,
            blender_script,
            color,
            rotation_angle,
            flip_z,
            stand_upright,
            color_mode,
            DEFAULT_RESOLUTION,
            camera_distance,
            camera_height,
            camera_base_angle,
        )
        if render_path:
            render_path.rename(exploded_dir / f"face_{face_idx}_loop.png")

        if edge_loop_npz.exists():
            edge_loop_npz.unlink()

        # Render each edge individually (use normalized edges)
        for edge_idx, single_edge in enumerate(edge_loop_edges_normalized):
            edge_npz = exploded_dir / f"face_{face_idx}_edge_{edge_idx}.npz"
            single_edge_points, single_edge_offsets = pack_edges([single_edge])
            np.savez(
                edge_npz,
                edge_points=single_edge_points,
                edge_offsets=single_edge_offsets,
            )

            render_path = render_blender_step(
                edge_npz,
                exploded_dir,
                blender_script,
                color,
                rotation_angle,
                flip_z,
                stand_upright,
                color_mode,
                DEFAULT_RESOLUTION,
                camera_distance,
                camera_height,
                camera_base_angle,
            )
            if render_path:
                render_path.rename(
                    exploded_dir / f"face_{face_idx}_edge_{edge_idx}.png"
                )

            if edge_npz.exists():
                edge_npz.unlink()

        face_idx += 1
        face_explorer.Next()


def render_blender_step(
    npz_file: Path,
    output_dir: Path,
    blender_script: Path,
    color: str = DEFAULT_COLOR,
    rotation_angle: float = DEFAULT_ROTATION_ANGLE,
    flip_z: bool = False,
    stand_upright: bool = False,
    color_mode: str = DEFAULT_COLOR_MODE,
    resolution: int = DEFAULT_RESOLUTION,
    camera_distance: float = None,
    camera_height: float = None,
    camera_base_angle: float = None,
    output_name: str = None,
) -> Path:
    """
    Render a processed NPZ file using Blender.

    Args:
        npz_file: Path to NPZ file containing geometry
        output_dir: Output directory for rendered image
        blender_script: Path to Blender Python script
        color: Color preset name
        rotation_angle: Camera rotation angle in degrees
        flip_z: Whether geometry is flipped along Z
        stand_upright: Whether geometry is rotated upright
        color_mode: Color mode (rgb/rgba)
        resolution: Rendering resolution in pixels
        camera_distance: Optional camera distance override
        camera_height: Optional camera height override
        camera_base_angle: Optional camera base angle override
        output_name: Optional custom output filename

    Returns:
        Path to rendered image file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name:
        output_image = output_dir / output_name
    else:
        output_image = output_dir / f"{npz_file.stem}.png"

    cmd = [
        BLENDER_EXECUTABLE,
        "--background",
        "--python",
        str(blender_script),
        "--",
        str(npz_file),
        str(output_image),
        color,
        str(rotation_angle),
        str(resolution),
    ]

    if flip_z:
        cmd.append("flip")
    if stand_upright:
        cmd.append("stand_upright")
    cmd.append(color_mode)

    # Add camera parameters if specified
    if camera_distance is not None:
        cmd.append(f"camdist={camera_distance}")
    if camera_height is not None:
        cmd.append(f"camheight={camera_height}")
    if camera_base_angle is not None:
        cmd.append(f"camangle={camera_base_angle}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    log = result.stdout + "\n" + result.stderr

    if result.returncode != 0 or "Traceback (most recent call last)" in log:
        print(f"\n[BLENDER LOG]\n{log}")
        raise RuntimeError(f"Blender failed while rendering {npz_file}")

    return output_image


def process_file(args):
    """
    Process a single STEP file: convert to NPZ and render.

    Args:
        args: Tuple of all processing parameters

    Returns:
        Tuple of (npz_path, render_path) or (None, None) if processing failed
    """
    (
        step_file,
        output_npz_dir,
        output_render_dir,
        blender_script,
        edge_deflection,
        mesh_linear_deflection,
        mesh_angular_deflection,
        color,
        stand_upright,
        flip_z,
        no_normalize,
        rotation_angle,
        color_mode,
        resolution,
        partial_faces,
        verbose,
        output_name,
        camera_distance,
        camera_height,
        camera_base_angle,
        ground_plane_z,
    ) = args

    npz_path = process_step_file(
        step_file,
        output_npz_dir,
        edge_deflection,
        mesh_linear_deflection,
        mesh_angular_deflection,
        stand_upright,
        flip_z,
        no_normalize,
        partial_faces,
        verbose,
        ground_plane_z,
    )

    if npz_path:
        render_path = render_blender_step(
            npz_path,
            output_render_dir,
            blender_script,
            color,
            rotation_angle,
            flip_z,
            stand_upright,
            color_mode,
            resolution,
            camera_distance,
            camera_height,
            camera_base_angle,
            output_name=output_name,
        )
        return npz_path, render_path

    return None, None


@app.command()
def render(
    input_dir_or_path: Path = typer.Argument(
        ..., help="Directory or file path containing STEP file(s)"
    ),
    output_dir: Path = typer.Option(None, help="Base output directory"),
    n_steps: int = typer.Option(DEFAULT_N_STEPS, help="Number of step files to render"),
    edge_deflection: float = typer.Option(
        DEFAULT_EDGE_DEFLECTION, help="Edge discretization deflection"
    ),
    mesh_linear_deflection: float = typer.Option(
        DEFAULT_MESH_LINEAR_DEFLECTION, help="Mesh linear deflection"
    ),
    mesh_angular_deflection: float = typer.Option(
        DEFAULT_MESH_ANGULAR_DEFLECTION, help="Mesh angular deflection"
    ),
    color: str = typer.Option(
        DEFAULT_COLOR, help="Color for rendering (blue/pink/orange/green/silver). Use --colors for multi-color output."
    ),
    colors: str = typer.Option(
        None, help="Comma-separated list of colors to render each file in multiple colors (e.g. 'blue,pink,orange')"
    ),
    max_workers: int = typer.Option(
        DEFAULT_MAX_WORKERS, help="Maximum number of parallel workers"
    ),
    stand_upright: bool = typer.Option(
        False, help="Stand model upright (rotate 90° around X axis)"
    ),
    flip_z: bool = typer.Option(False, help="Flip model along Z axis (XZ plane)"),
    no_normalize: bool = typer.Option(
        False, help="Skip normalization (keep original scale and position)"
    ),
    rotation_angle: float = typer.Option(
        DEFAULT_ROTATION_ANGLE, help="Camera rotation angle in degrees"
    ),
    color_mode: str = typer.Option(DEFAULT_COLOR_MODE, help="Color mode (rgb/rgba)"),
    partial: str = typer.Option(
        None, help="Comma-separated face indices to keep (e.g. '1,2,3')"
    ),
    explode: bool = typer.Option(
        False,
        help="If set, renders exploded views of each face, its edge loop, and each edge.",
    ),
    resolution: int = typer.Option(
        DEFAULT_RESOLUTION, help="Rendering resolution in pixels (width and height)"
    ),
    keep_intermediate: bool = typer.Option(
        False, help="Keep intermediate processed files (NPZ meshes)"
    ),
    intermediate_dir: Path = typer.Option(
        None,
        help="Directory for intermediate files (default: temp or output_dir/intermediate)",
    ),
    output_name: str = typer.Option(
        None,
        help="Custom output filename (e.g., 'test_basic.png'). If not provided, uses input filename.",
    ),
    verbose: bool = typer.Option(
        False, help="Print detailed information about each STEP file being processed"
    ),
    fast: bool = typer.Option(
        False,
        help="Fast mode: reduce quality for faster processing (coarser mesh)",
    ),
    camera_distance: float = typer.Option(
        None, help="Camera distance from model (default: 3.0)"
    ),
    camera_height: float = typer.Option(
        None, help="Camera height (Z position, default: 2.0)"
    ),
    camera_base_angle: float = typer.Option(
        None, help="Camera base angle in degrees (default: -45.0)"
    ),
    ground_plane_z: float = typer.Option(
        GROUND_PLANE_Z, help="Ground plane Z position after normalization (default: -0.5)"
    ),
):
    """
    Process STEP file(s):
    1. Convert STEP files to NPZ format
    2. Generate Blender renders for each file
    """
    # Validate input path
    if not input_dir_or_path.exists():
        typer.echo(f"Error: Input path {input_dir_or_path} does not exist")
        raise typer.Exit(1)

    # Set up directories
    output_dir, output_npz_dir, use_temp_dir = _setup_directories(
        input_dir_or_path, output_dir, keep_intermediate, intermediate_dir
    )
    output_render_dir = output_dir

    # Discover STEP files
    step_files = _discover_step_files(input_dir_or_path, n_steps)

    # Get path to Blender script
    blender_script = Path(BLENDER_RENDER_SCRIPT)
    if not blender_script.exists():
        typer.echo(f"Error: Blender script not found at {blender_script}")
        raise typer.Exit(1)

    # Parse partial faces
    if partial is not None:
        partial_faces = [int(idx.strip()) for idx in partial.split(",") if idx.strip()]
    else:
        partial_faces = None

    # Apply fast mode settings
    if fast:
        if verbose:
            print(
                "Fast mode enabled: reducing mesh quality for speed (resolution unchanged)"
            )
        edge_deflection = FAST_EDGE_DEFLECTION
        mesh_linear_deflection = FAST_MESH_LINEAR_DEFLECTION
        mesh_angular_deflection = FAST_MESH_ANGULAR_DEFLECTION

    # Parse multi-color list
    color_list = [color]
    if colors is not None:
        color_list = [c.strip() for c in colors.split(",") if c.strip()]
        # Validate all colors
        for c in color_list:
            if c not in COLOR_PRESETS:
                typer.echo(f"Error: Unknown color '{c}'. Available: {', '.join(COLOR_PRESETS.keys())}")
                raise typer.Exit(1)

    if explode:
        for step_file in step_files:
            for c in color_list:
                explode_output = output_render_dir
                if len(color_list) > 1:
                    explode_output = output_render_dir / c
                explode_step_file(
                    step_file,
                    explode_output,
                    edge_deflection,
                    mesh_linear_deflection,
                    mesh_angular_deflection,
                    stand_upright,
                    flip_z,
                    no_normalize,
                    blender_script,
                    c,
                    rotation_angle,
                    color_mode,
                    camera_distance,
                    camera_height,
                    camera_base_angle,
                    ground_plane_z,
                )
        return

    # Process files in parallel, for each color
    successful_npz = []
    successful_renders = []

    for c in color_list:
        # When multiple colors, create subdirectory for each color
        render_dir_for_color = output_render_dir
        if len(color_list) > 1:
            render_dir_for_color = output_render_dir / c
            render_dir_for_color.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Prepare arguments for each file
            args_list = [
                (
                    step_file,
                    output_npz_dir,
                    render_dir_for_color,
                    blender_script,
                    edge_deflection,
                    mesh_linear_deflection,
                    mesh_angular_deflection,
                    c,
                    stand_upright,
                    flip_z,
                    no_normalize,
                    rotation_angle,
                    color_mode,
                    resolution,
                    partial_faces,
                    verbose,
                    output_name,
                    camera_distance,
                    camera_height,
                    camera_base_angle,
                    ground_plane_z,
                )
                for step_file in step_files
            ]

            # Submit all tasks and track progress
            futures = [executor.submit(process_file, args) for args in args_list]

            # Process results with progress bar
            desc = f"Processing files ({c})" if len(color_list) > 1 else "Processing files"
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=desc
            ):
                npz_path, render_path = future.result()
                if npz_path:
                    successful_npz.append(npz_path)
                if render_path:
                    successful_renders.append(render_path)

    # Clean up temporary directory if used
    if use_temp_dir:
        import shutil

        try:
            shutil.rmtree(output_npz_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temp directory {output_npz_dir}: {e}")

    # Print summary
    print("\nProcessing complete!")
    print(f"Processed {len(successful_npz)} files")
    print(f"Rendered {len(successful_renders)} images")
    if len(color_list) > 1:
        print(f"Colors rendered: {', '.join(color_list)}")
    print(f"\nOutput locations:")
    if keep_intermediate or intermediate_dir is not None:
        print(f"Intermediate files: {output_npz_dir}")
    print(f"Renders: {output_render_dir}")


@app.command()
def view(
    step_file: Path = typer.Argument(..., help="Path to STEP file"),
    edge_deflection: float = typer.Option(
        DEFAULT_EDGE_DEFLECTION, help="Edge discretization deflection"
    ),
    mesh_linear_deflection: float = typer.Option(
        DEFAULT_MESH_LINEAR_DEFLECTION, help="Mesh linear deflection"
    ),
    mesh_angular_deflection: float = typer.Option(
        DEFAULT_MESH_ANGULAR_DEFLECTION, help="Mesh angular deflection"
    ),
    stand_upright: bool = typer.Option(
        False, help="Stand model upright (rotate 90° around X axis)"
    ),
    flip_z: bool = typer.Option(False, help="Flip model along Z axis (XZ plane)"),
    no_normalize: bool = typer.Option(
        False, help="Skip normalization (keep original scale and position)"
    ),
    color: str = typer.Option(
        DEFAULT_COLOR, help="Color for mesh (blue/pink/orange/green/silver)"
    ),
    show_edges: bool = typer.Option(True, help="Show edge curves"),
    show_mesh: bool = typer.Option(True, help="Show surface mesh"),
    fast: bool = typer.Option(
        False, help="Fast mode: reduce quality for faster processing"
    ),
):
    """
    Launch interactive Polyscope viewer for STEP file.

    Provides real-time rotation, zoom, and inspection of CAD geometry.
    """
    try:
        import polyscope as ps
    except ImportError:
        typer.echo("Error: polyscope not installed")
        typer.echo("Install with: pip install polyscope")
        raise typer.Exit(1)

    # Validate input
    if not step_file.exists():
        typer.echo(f"Error: STEP file {step_file} does not exist")
        raise typer.Exit(1)

    # Apply fast mode settings
    if fast:
        edge_deflection = FAST_EDGE_DEFLECTION
        mesh_linear_deflection = FAST_MESH_LINEAR_DEFLECTION
        mesh_angular_deflection = FAST_MESH_ANGULAR_DEFLECTION

    # Process STEP file to NPZ
    typer.echo(f"Processing {step_file.name}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = process_step_file(
            step_file,
            Path(tmpdir),
            edge_deflection,
            mesh_linear_deflection,
            mesh_angular_deflection,
            stand_upright,
            flip_z,
            no_normalize,
            verbose=True,
            ground_plane_z=ground_plane_z,
        )

        if not npz_path:
            typer.echo("Error: Failed to process STEP file")
            raise typer.Exit(1)

        # Load NPZ data (pickle-free)
        data = np.load(npz_path)
        vertices = data["vertices"]
        triangles = data["triangles"]
        edges = unpack_edges(data["edge_points"], data["edge_offsets"])

        typer.echo(f"Launching Polyscope viewer...")
        typer.echo(
            f"  Vertices: {len(vertices)}, Triangles: {len(triangles)}, Edges: {len(edges)}"
        )

        # Initialize Polyscope
        ps.init()
        ps.set_ground_plane_mode("shadow_only")
        ps.set_ground_plane_height_factor(ground_plane_z, is_relative=False)

        # Register surface mesh if requested
        if show_mesh and len(vertices) > 0 and len(triangles) > 0:
            mesh_vis = ps.register_surface_mesh(
                "model",
                vertices,
                triangles,
            )
            # Set color (RGB only, polyscope doesn't use alpha)
            color_rgb = COLOR_PRESETS[color][:3]
            mesh_vis.set_color(color_rgb)

        # Register edges as curve network if requested
        if show_edges and len(edges) > 0:
            # Convert edge list to vertex array + segment indices
            all_edge_verts = []
            all_edge_segs = []
            vert_offset = 0

            for edge_points in edges:
                n_points = len(edge_points)
                all_edge_verts.extend(edge_points.tolist())

                # Create segments connecting consecutive points
                segments = [
                    (i + vert_offset, i + 1 + vert_offset) for i in range(n_points - 1)
                ]
                all_edge_segs.extend(segments)
                vert_offset += n_points

            if len(all_edge_verts) > 0:
                edge_verts = np.array(all_edge_verts)
                edge_segs = np.array(all_edge_segs)

                edge_vis = ps.register_curve_network(
                    "edges", edge_verts, edge_segs, enabled=True
                )
                edge_vis.set_color((0.1, 0.1, 0.1))  # Dark edges
                edge_vis.set_radius(0.005, relative=False)

        typer.echo("\n✓ Viewer launched. Close window to exit.")
        typer.echo("  Controls: Left-click drag to rotate, scroll to zoom")

        # Launch viewer (blocks until window closed)
        ps.show()


@app.command()
def render_video(
    step_file: Path = typer.Argument(..., help="Path to STEP file"),
    output_path: Path = typer.Option(
        None, help="Output video path (default: same as input)"
    ),
    fps: int = typer.Option(DEFAULT_VIDEO_FPS, help="Frames per second"),
    duration: float = typer.Option(DEFAULT_VIDEO_DURATION, help="Duration in seconds"),
    video_format: str = typer.Option(
        DEFAULT_VIDEO_FORMAT, help="Output format (mp4 or gif)"
    ),
    color: str = typer.Option(
        DEFAULT_COLOR, help="Color for rendering (blue/pink/orange/green/silver)"
    ),
    resolution: int = typer.Option(
        DEFAULT_RESOLUTION, help="Rendering resolution in pixels"
    ),
    edge_deflection: float = typer.Option(
        DEFAULT_EDGE_DEFLECTION, help="Edge discretization deflection"
    ),
    mesh_linear_deflection: float = typer.Option(
        DEFAULT_MESH_LINEAR_DEFLECTION, help="Mesh linear deflection"
    ),
    mesh_angular_deflection: float = typer.Option(
        DEFAULT_MESH_ANGULAR_DEFLECTION, help="Mesh angular deflection"
    ),
    stand_upright: bool = typer.Option(
        False, help="Stand model upright (rotate 90° around X axis)"
    ),
    flip_z: bool = typer.Option(False, help="Flip model along Z axis (XZ plane)"),
    no_normalize: bool = typer.Option(
        False, help="Skip normalization (keep original scale and position)"
    ),
    color_mode: str = typer.Option(DEFAULT_COLOR_MODE, help="Color mode (rgb or rgba)"),
    camera_distance: float = typer.Option(
        None, help="Camera distance from model (default: 3.0)"
    ),
    camera_height: float = typer.Option(
        None, help="Camera height (Z position, default: 2.0)"
    ),
    camera_base_angle: float = typer.Option(
        None, help="Camera base angle in degrees (default: -45.0)"
    ),
    ground_plane_z: float = typer.Option(
        GROUND_PLANE_Z, help="Ground plane Z position after normalization (default: -0.5)"
    ),
):
    """
    Create a rotating video animation of a STEP file.
    """
    import shutil

    # Validate input
    if not step_file.exists():
        typer.echo(f"Error: STEP file {step_file} does not exist")
        raise typer.Exit(1)

    # Set output path and infer format from extension if provided
    if output_path is None:
        output_path = step_file.with_suffix(f".{video_format}")
    else:
        # Infer format from output path extension
        ext = output_path.suffix.lower().lstrip(".")
        if ext in ["mp4", "gif"]:
            video_format = ext

    if video_format not in ["mp4", "gif"]:
        typer.echo("Error: video_format must be 'mp4' or 'gif'")
        raise typer.Exit(1)

    # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        typer.echo("Error: ffmpeg is not installed or not in PATH")
        typer.echo("Please install ffmpeg: sudo apt install ffmpeg")
        raise typer.Exit(1)

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        npz_dir = tmpdir_path / "npz"
        frames_dir = tmpdir_path / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        typer.echo(f"Creating rotating video of {step_file.name}...")
        typer.echo(f"Duration: {duration}s, FPS: {fps}, Format: {video_format}")

        # Calculate number of frames
        n_frames = int(fps * duration)
        angle_step = 360.0 / n_frames

        # Process STEP file to NPZ
        typer.echo("Converting STEP to NPZ...")
        npz_path = process_step_file(
            step_file,
            npz_dir,
            edge_deflection,
            mesh_linear_deflection,
            mesh_angular_deflection,
            stand_upright,
            flip_z,
            no_normalize,
            ground_plane_z=ground_plane_z,
        )

        if not npz_path:
            typer.echo("Error: Failed to process STEP file")
            raise typer.Exit(1)

        # Get blender script path
        blender_script = Path(BLENDER_RENDER_SCRIPT)
        if not blender_script.exists():
            typer.echo(f"Error: Blender script not found at {blender_script}")
            raise typer.Exit(1)

        # Render each frame
        typer.echo(f"Rendering {n_frames} frames...")
        _render_video_frames(
            npz_path,
            frames_dir,
            n_frames,
            blender_script,
            color,
            resolution,
            flip_z,
            stand_upright,
            color_mode,
            blender_executable=BLENDER_EXECUTABLE,
            camera_distance=camera_distance,
            camera_height=camera_height,
            camera_base_angle=camera_base_angle,
        )

        # Create video using ffmpeg
        typer.echo(f"Creating {video_format.upper()} video...")
        _create_video_with_ffmpeg(frames_dir, output_path, fps, video_format)

        typer.echo(f"✓ Video saved to: {output_path}")
        typer.echo(f"  Frames: {n_frames}, Duration: {duration}s, FPS: {fps}")


@app.command()
def batch(
    input_dir: Path = typer.Argument(
        ..., help="Directory containing STEP file(s) to batch process"
    ),
    output_dir: Path = typer.Option(None, help="Base output directory (default: same as input_dir)"),
    colors: str = typer.Option(
        "blue,pink,orange,green,silver",
        help="Comma-separated list of colors to render (e.g. 'blue,pink,orange')",
    ),
    n_steps: int = typer.Option(DEFAULT_N_STEPS, help="Maximum number of STEP files to process"),
    resolution: int = typer.Option(
        DEFAULT_RESOLUTION, help="Rendering resolution in pixels (width and height)"
    ),
    max_workers: int = typer.Option(
        DEFAULT_MAX_WORKERS, help="Maximum number of parallel workers"
    ),
    stand_upright: bool = typer.Option(
        False, help="Stand model upright (rotate 90° around X axis)"
    ),
    flip_z: bool = typer.Option(False, help="Flip model along Z axis (XZ plane)"),
    no_normalize: bool = typer.Option(
        False, help="Skip normalization (keep original scale and position)"
    ),
    color_mode: str = typer.Option(DEFAULT_COLOR_MODE, help="Color mode (rgb/rgba)"),
    fast: bool = typer.Option(
        False, help="Fast mode: reduce quality for faster processing (coarser mesh)"
    ),
    camera_distance: float = typer.Option(
        None, help="Camera distance from model (default: 3.5)"
    ),
    camera_height: float = typer.Option(
        None, help="Camera height (Z position, default: 2.5)"
    ),
    camera_base_angle: float = typer.Option(
        None, help="Camera base angle in degrees (default: -35.0)"
    ),
    ground_plane_z: float = typer.Option(
        GROUND_PLANE_Z, help="Ground plane Z position after normalization (default: -0.5)"
    ),
    verbose: bool = typer.Option(
        False, help="Print detailed information about each STEP file being processed"
    ),
):
    """
    Batch process all STEP files in a directory, rendering each in multiple colors.

    This is a convenience command that processes all .step files in the input directory
    and renders each file in every specified color. Output is organized by color subdirectories.

    Example:
        python vis_step.py batch samples/ --colors "blue,pink,orange" --resolution 512
    """
    # Validate input
    if not input_dir.exists() or not input_dir.is_dir():
        typer.echo(f"Error: {input_dir} is not a valid directory")
        raise typer.Exit(1)

    # Parse and validate colors
    color_list = [c.strip() for c in colors.split(",") if c.strip()]
    for c in color_list:
        if c not in COLOR_PRESETS:
            typer.echo(f"Error: Unknown color '{c}'. Available: {', '.join(COLOR_PRESETS.keys())}")
            raise typer.Exit(1)

    # Set up output directory
    if output_dir is None:
        output_dir = input_dir / "batch_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover STEP files
    step_files = _discover_step_files(input_dir, n_steps)

    typer.echo(f"Batch processing {len(step_files)} STEP file(s)")
    typer.echo(f"Colors: {', '.join(color_list)}")
    typer.echo(f"Output: {output_dir}")
    typer.echo("")

    # Get blender script
    blender_script = Path(BLENDER_RENDER_SCRIPT)
    if not blender_script.exists():
        typer.echo(f"Error: Blender script not found at {blender_script}")
        raise typer.Exit(1)

    # Apply fast mode
    edge_deflection = DEFAULT_EDGE_DEFLECTION
    mesh_linear_deflection = DEFAULT_MESH_LINEAR_DEFLECTION
    mesh_angular_deflection = DEFAULT_MESH_ANGULAR_DEFLECTION
    if fast:
        edge_deflection = FAST_EDGE_DEFLECTION
        mesh_linear_deflection = FAST_MESH_LINEAR_DEFLECTION
        mesh_angular_deflection = FAST_MESH_ANGULAR_DEFLECTION

    total_renders = 0

    for c in color_list:
        # Create color subdirectory
        color_output_dir = output_dir / c
        color_output_dir.mkdir(parents=True, exist_ok=True)

        # Use temp dir for intermediate NPZ files
        with tempfile.TemporaryDirectory(prefix="brepvis_batch_") as tmpdir:
            npz_dir = Path(tmpdir)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                args_list = [
                    (
                        step_file,
                        npz_dir,
                        color_output_dir,
                        blender_script,
                        edge_deflection,
                        mesh_linear_deflection,
                        mesh_angular_deflection,
                        c,
                        stand_upright,
                        flip_z,
                        no_normalize,
                        DEFAULT_ROTATION_ANGLE,
                        color_mode,
                        resolution,
                        None,  # partial_faces
                        verbose,
                        None,  # output_name
                        camera_distance,
                        camera_height,
                        camera_base_angle,
                        ground_plane_z,
                    )
                    for step_file in step_files
                ]

                futures = [executor.submit(process_file, args) for args in args_list]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Rendering ({c})",
                ):
                    npz_path, render_path = future.result()
                    if render_path:
                        total_renders += 1

    typer.echo(f"\nBatch processing complete!")
    typer.echo(f"Total renders: {total_renders}")
    typer.echo(f"Files: {len(step_files)}, Colors: {len(color_list)}")
    typer.echo(f"Output: {output_dir}")
    for c in color_list:
        typer.echo(f"  {c}/: {output_dir / c}")


if __name__ == "__main__":
    app()
