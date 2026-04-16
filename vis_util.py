"""
BrepVis Utility Functions

Consolidated utility module containing:
- Geometry processing: edge discretization, mesh tessellation, transformations
- Edge offset computation (z-fighting prevention with normal-based approach)
- STEP file I/O and geometry counting
- Directory setup and file discovery
- Video frame rendering and FFmpeg integration
"""

import os
import subprocess
import tempfile
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
import typer

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Extend.DataExchange import write_stl_file


# =============================================================================
# Edge Offset Parameters
# =============================================================================

# Edge offset parameters (for z-fighting prevention)
EDGE_NORMAL_OFFSET = 5e-4
EDGE_NORMAL_NOISE_STD = 5e-3
EDGE_NORMAL_NOISE_ITERATIONS = 10


# =============================================================================
# Geometry Processing Functions
# =============================================================================


def discretize_edge(edge, deflection=0.01):
    """
    Convert an edge to a series of 3D points with given deflection tolerance.

    Args:
        edge: OpenCASCADE edge object
        deflection: Maximum distance between approximation and true curve

    Returns:
        List of (x, y, z) tuples representing discretized edge points,
        or empty list if edge is degenerate or invalid
    """
    curve_tuple = BRep_Tool.Curve(edge)

    # BRep_Tool.Curve returns different tuple lengths depending on edge type:
    # - len=3: (curve_handle, first_param, last_param) - normal edge with parameters
    # - len=2: (first_param, last_param) - degenerate edge, no geometry
    # - len=1: (curve_handle,) - edge without explicit parameters

    if len(curve_tuple) == 2:
        # Degenerate edge with no geometry (just parameter bounds), skip it
        return []
    elif len(curve_tuple) == 1:
        # Curve handle without parameters - use default parameterization
        curve_handle = curve_tuple[0]
        if curve_handle is None:
            return []
        try:
            curve = GeomAdaptor_Curve(curve_handle)
        except Exception as e:
            print(f"Error discretizing edge: {e}")
            return []
    elif len(curve_tuple) == 3:
        # Standard case: curve with explicit parameter range
        curve_handle = curve_tuple[0]
        first_param = curve_tuple[1]
        last_param = curve_tuple[2]

        if curve_handle is None:
            return []  # Invalid curve

        try:
            curve = GeomAdaptor_Curve(curve_handle, first_param, last_param)
        except Exception as e:
            print(f"Error discretizing edge: {e}")
            return []
    else:
        # Unknown format, skip
        return []

    # Discretize the curve using quasi-uniform deflection
    discretizer = GCPnts_QuasiUniformDeflection(curve, deflection)
    if not discretizer.IsDone():
        return []

    points = []
    for i in range(1, discretizer.NbPoints() + 1):
        p = discretizer.Value(i)
        points.append((p.X(), p.Y(), p.Z()))
    return points


def extract_faces(shape, linear_deflection=0.1, angular_deflection=0.1):
    """
    Extract faces from a shape as a triangulated mesh.

    Uses OpenCASCADE's STL export for tessellation with specified deflection parameters.

    Args:
        shape: OpenCASCADE shape object
        linear_deflection: Maximum linear distance between mesh and surface
        angular_deflection: Maximum angular deviation in radians

    Returns:
        Tuple of (vertices, triangles) where:
        - vertices: Nx3 numpy array of vertex coordinates
        - triangles: Mx3 numpy array of triangle vertex indices
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        stl_path = os.path.join(tmpdir, "brep.stl")
        write_stl_file(
            shape,
            stl_path,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
        mesh = trimesh.load_mesh(stl_path)
    return np.array(mesh.vertices), np.array(mesh.faces)


def apply_transformations(vertices, stand_upright=False, flip_z=False):
    """
    Apply geometric transformations to vertices.

    Args:
        vertices: Nx3 numpy array of vertex coordinates
        stand_upright: If True, rotate 90° around X axis (makes Z-up models Y-up)
        flip_z: If True, flip along Z axis (mirror across XZ plane)

    Returns:
        Transformed Nx3 numpy array
    """
    result = vertices.copy()

    if stand_upright:
        # Rotate 90° around X axis: [x, y, z] -> [x, -z, y]
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        result = result @ rotation_matrix.T

    if flip_z:
        # Flip Z coordinate (mirror across XZ plane)
        result[:, 2] = -result[:, 2]

    return result


def normalize_vertices(vertices, no_normalize=False):
    """
    Normalize vertices to [-1, 1] bounding box and position ground plane at z=-0.5.

    Args:
        vertices: Nx3 numpy array of vertex coordinates
        no_normalize: If True, skip normalization (keep original scale)

    Returns:
        Tuple of (normalized_vertices, scale, center) where:
        - normalized_vertices: Transformed Nx3 array
        - scale: Scalar used for normalization (1.0 if no_normalize=True)
        - center: 3D point used as center (zeros if no_normalize=True)
    """
    v_min, v_max = vertices.min(axis=0, keepdims=True), vertices.max(axis=0, keepdims=True)

    if not no_normalize:
        # Center and scale to [-1, 1] range
        center = (v_max + v_min) / 2
        scale = np.max(v_max - v_min)
        vertices_normalized = (vertices - center) / scale

        # Position ground plane at z=-0.5
        z_min = vertices_normalized[:, 2].min()
        vertices_normalized[:, 2] -= z_min + 0.5

        return vertices_normalized, scale, center.flatten()
    else:
        # No normalization - return original with identity transform
        scale = 1.0
        center = np.zeros(3)
        return vertices, scale, center


def compute_edge_offset_normals(edge_points, mesh):
    """
    Compute averaged normals for edge points to offset them slightly from mesh surface.

    This prevents z-fighting artifacts by moving edges slightly outward along the
    surface normal direction. Uses noisy sampling to robustly estimate normals.

    NOTE: This complexity is necessary because the camera orbits 360° around the model.
    A simple Z-offset would only work for horizontal surfaces and fail at side angles.

    Args:
        edge_points: Nx3 numpy array of edge point coordinates
        mesh: Trimesh object of the surface mesh

    Returns:
        Nx3 numpy array of normalized normal vectors at edge points
    """
    # Average normals over multiple noisy samples for robustness
    avg_normals = np.zeros_like(edge_points)

    for i in range(EDGE_NORMAL_NOISE_ITERATIONS):
        # Add small random noise to edge points
        noisy_edge = edge_points + EDGE_NORMAL_NOISE_STD * np.random.randn(*edge_points.shape)

        # Find closest triangle on mesh for each noisy point
        closest_points, distances, triangle_ids = trimesh.proximity.closest_point(mesh, noisy_edge)

        # Ensure integer indices for array indexing
        triangle_ids = np.asarray(triangle_ids, dtype=np.int64)

        # Accumulate normals from closest triangles
        avg_normals += mesh.face_normals[triangle_ids]

    # Average accumulated normals
    avg_normals = avg_normals / EDGE_NORMAL_NOISE_ITERATIONS

    # Normalize to unit vectors
    avg_normals = avg_normals / np.linalg.norm(avg_normals, keepdims=True, axis=1)

    return avg_normals


def offset_edges_from_surface(edges, mesh):
    """
    Offset edge points slightly outward from mesh surface to prevent z-fighting.

    NOTE: This normal-based approach is necessary (not just simple z-offset) because
    the camera can view the model from any angle during 360° rotation.

    Args:
        edges: List of Nx3 numpy arrays, each representing an edge's points
        mesh: Trimesh object of the surface mesh

    Returns:
        List of offset Nx3 numpy arrays (same structure as input)
    """
    offset_edges = []

    for edge in edges:
        # Compute averaged normals at edge points
        normals = compute_edge_offset_normals(edge, mesh)

        # Offset edge points slightly along normals
        offset_edge = edge + EDGE_NORMAL_OFFSET * normals
        offset_edges.append(offset_edge)

    return offset_edges


# =============================================================================
# Edge Packing Utilities (pickle-free NPZ storage)
# =============================================================================


def pack_edges(edges):
    """
    Pack a list of variable-length edge arrays into flat arrays for pickle-free NPZ storage.

    Instead of storing edges as a ragged object array (which requires pickle and
    enables arbitrary code execution on load), this stores edges as a single
    concatenated point array plus an offsets array to reconstruct individual edges.

    Args:
        edges: List of Nx3 numpy arrays, each representing an edge's points

    Returns:
        Tuple of (edge_points, edge_offsets) where:
        - edge_points: Concatenated Mx3 array of all edge points
        - edge_offsets: 1D int64 array of length len(edges)+1 with cumulative offsets
    """
    if not edges:
        return np.empty((0, 3), dtype=np.float64), np.array([0], dtype=np.int64)
    edge_points = np.concatenate(edges, axis=0)
    edge_lengths = np.array([len(e) for e in edges], dtype=np.int64)
    edge_offsets = np.concatenate([[0], np.cumsum(edge_lengths)]).astype(np.int64)
    return edge_points, edge_offsets


def unpack_edges(edge_points, edge_offsets):
    """
    Unpack flat edge arrays back to a list of variable-length edge arrays.

    Inverse of pack_edges(). Reconstructs the original list of edge arrays
    from the concatenated point array and offset indices.

    Args:
        edge_points: Concatenated Mx3 array of all edge points
        edge_offsets: 1D int64 array of cumulative offsets (length = num_edges + 1)

    Returns:
        List of Nx3 numpy arrays, one per edge
    """
    edges = []
    for i in range(len(edge_offsets) - 1):
        edges.append(edge_points[edge_offsets[i]:edge_offsets[i + 1]])
    return edges


# =============================================================================
# STEP File I/O and Processing
# =============================================================================


def _read_step_file(step_file: Path):
    """
    Read STEP file and return shape, or None if failed.

    Args:
        step_file: Path to STEP file

    Returns:
        OpenCASCADE shape object, or None if reading failed
    """
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(str(step_file))

    if status != IFSelect_RetDone:
        print(f"Error: Failed to read STEP file {step_file}")
        return None

    step_reader.TransferRoots()
    return step_reader.OneShape()


def _count_geometry(shape):
    """
    Count faces and edges in a shape.

    Args:
        shape: OpenCASCADE shape object

    Returns:
        Tuple of (face_count, edge_count)
    """
    face_count = 0
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face_count += 1
        face_explorer.Next()

    edge_count = 0
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge_count += 1
        edge_explorer.Next()

    return face_count, edge_count


def _discretize_all_edges(shape, edge_deflection: float):
    """
    Discretize all edges in a shape to point arrays.

    Args:
        shape: OpenCASCADE shape object
        edge_deflection: Deflection tolerance for discretization

    Returns:
        List of Nx3 numpy arrays, one per edge
    """
    all_edges = []
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = edge_explorer.Current()
        points = discretize_edge(edge, edge_deflection)
        if points:
            edge_array = np.stack(points, axis=0)
            all_edges.append(edge_array)
        edge_explorer.Next()
    return all_edges


def _build_partial_compound(shape, face_indices: list[int]):
    """
    Build compound containing only specified faces from shape.

    Args:
        shape: OpenCASCADE shape object
        face_indices: List of face indices to include

    Returns:
        TopoDS_Compound containing only specified faces
    """
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    current_face_idx = 0
    while face_explorer.More():
        if current_face_idx in face_indices:
            builder.Add(compound, face_explorer.Current())
        current_face_idx += 1
        face_explorer.Next()

    return compound


def _setup_directories(input_dir_or_path: Path, output_dir: Path,
                       keep_intermediate: bool, intermediate_dir: Path):
    """
    Set up output and intermediate directories.

    Args:
        input_dir_or_path: Input file or directory
        output_dir: Output directory (may be None)
        keep_intermediate: Whether to keep intermediate files
        intermediate_dir: Custom intermediate directory (may be None)

    Returns:
        Tuple of (output_dir, output_npz_dir, use_temp_dir)
    """
    # Set up output directory
    if output_dir is None:
        output_dir = (
            input_dir_or_path.parent
            if input_dir_or_path.is_file()
            else input_dir_or_path
        )

    # Set up intermediate files directory
    if keep_intermediate:
        # If keeping files, use specified dir or default to output_dir/intermediate
        if intermediate_dir is None:
            output_npz_dir = output_dir / "intermediate"
        else:
            output_npz_dir = intermediate_dir
        output_npz_dir.mkdir(parents=True, exist_ok=True)
        use_temp_dir = False
    else:
        # If not keeping, use temp directory that gets cleaned up
        if intermediate_dir is not None:
            # User specified a directory, respect their choice
            output_npz_dir = intermediate_dir
            output_npz_dir.mkdir(parents=True, exist_ok=True)
            use_temp_dir = False
        else:
            # Use temporary directory
            temp_dir = tempfile.mkdtemp(prefix="brepvis_")
            output_npz_dir = Path(temp_dir)
            use_temp_dir = True

    return output_dir, output_npz_dir, use_temp_dir


def _discover_step_files(input_dir_or_path: Path, n_steps: int):
    """
    Find STEP files to process.

    Args:
        input_dir_or_path: Input file or directory path
        n_steps: Maximum number of files to process

    Returns:
        List of Path objects for STEP files to process

    Raises:
        typer.Exit: If no valid STEP files found
    """
    if input_dir_or_path.is_file():
        if not input_dir_or_path.suffix.lower() == ".step":
            typer.echo("Input file must be a STEP file")
            raise typer.Exit(1)
        step_files = [input_dir_or_path]
    else:
        step_files = list(sorted(input_dir_or_path.glob("*.step")))
        if len(step_files) > n_steps:
            print(
                f"Warning: {len(step_files)} STEP files found, only rendering {n_steps} steps"
            )
            step_files = step_files[:n_steps]

    if not step_files:
        typer.echo("No STEP files found")
        raise typer.Exit(1)

    return step_files


def _render_video_frames(npz_path: Path, frames_dir: Path, n_frames: int,
                         blender_script: Path, color: str, resolution: int,
                         flip_z: bool, stand_upright: bool, color_mode: str,
                         blender_executable: str = "./blender",
                         camera_distance: float = None,
                         camera_height: float = None,
                         camera_base_angle: float = None):
    """
    Render all frames for video animation.

    Args:
        npz_path: Path to processed NPZ file
        frames_dir: Directory to save frames
        n_frames: Number of frames to render
        blender_script: Path to Blender rendering script
        color: Color preset name
        resolution: Rendering resolution
        flip_z: Whether geometry is flipped
        stand_upright: Whether geometry is rotated upright
        color_mode: Color mode (rgb/rgba)
        blender_executable: Path to Blender executable

    Raises:
        typer.Exit: If frame rendering fails
    """
    angle_step = 360.0 / n_frames

    for i in tqdm(range(n_frames), desc="Rendering frames"):
        rotation_angle = i * angle_step
        frame_output = frames_dir / f"frame_{i:04d}.png"

        cmd = [
            blender_executable,
            "--background",
            "--python",
            str(blender_script),
            "--",
            str(npz_path),
            str(frame_output),
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

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            typer.echo(f"Error rendering frame {i}: {e.stderr.decode()}")
            raise typer.Exit(1)


def _create_video_with_ffmpeg(frames_dir: Path, output_path: Path,
                               fps: int, video_format: str):
    """
    Create video from frames using ffmpeg.

    Args:
        frames_dir: Directory containing frame images
        output_path: Path for output video file
        fps: Frames per second
        video_format: Output format (mp4 or gif)

    Raises:
        typer.Exit: If video creation fails
    """
    frame_pattern = str(frames_dir / "frame_%04d.png")

    if video_format == "mp4":
        # Create MP4 with H.264 codec
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "23",
            str(output_path),
        ]
    else:  # gif
        # Create GIF with palette for better quality
        palette_path = frames_dir.parent / "palette.png"

        # Generate palette
        palette_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-vf",
            "palettegen",
            str(palette_path),
        ]
        subprocess.run(palette_cmd, check=True, capture_output=True)

        # Create GIF using palette
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frame_pattern,
            "-i",
            str(palette_path),
            "-filter_complex",
            "paletteuse",
            str(output_path),
        ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error creating video: {e.stderr.decode()}")
        raise typer.Exit(1)
