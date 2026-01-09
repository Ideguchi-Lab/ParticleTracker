"""Conversion utilities for napari layer data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pt3d.config import VoxelSize


def to_napari_points(
    detections: pd.DataFrame,
    include_frame: bool = True,
) -> NDArray[np.float64]:
    """Convert detection DataFrame to napari Points format.

    Parameters
    ----------
    detections : pd.DataFrame
        DataFrame with columns [frame, z, y, x] or [z, y, x]
    include_frame : bool
        Whether to include frame dimension (for 4D data)

    Returns
    -------
    NDArray[np.float64]
        Array with shape (N, 4) for [t, z, y, x] or (N, 3) for [z, y, x]

    Notes
    -----
    napari Points layer expects coordinates in the order matching
    the image dimensions. For 4D data (t, z, y, x), points should
    be (t, z, y, x).
    """
    if len(detections) == 0:
        if include_frame:
            return np.empty((0, 4), dtype=np.float64)
        return np.empty((0, 3), dtype=np.float64)

    if include_frame and "frame" in detections.columns:
        return detections[["frame", "z", "y", "x"]].values.astype(np.float64)
    return detections[["z", "y", "x"]].values.astype(np.float64)


def to_napari_tracks(
    tracks: pd.DataFrame,
) -> NDArray[np.float64]:
    """Convert tracks DataFrame to napari Tracks format.

    Parameters
    ----------
    tracks : pd.DataFrame
        DataFrame with columns [particle, frame, z, y, x]

    Returns
    -------
    NDArray[np.float64]
        Array with shape (N, 5) for [track_id, t, z, y, x]

    Notes
    -----
    napari Tracks layer expects data in format:
    [track_id, t, z, y, x] where rows must be sorted by
    (track_id, t) for proper rendering.
    """
    if len(tracks) == 0:
        return np.empty((0, 5), dtype=np.float64)

    # Sort by particle and frame for proper track ordering
    sorted_tracks = tracks.sort_values(["particle", "frame"])
    return sorted_tracks[["particle", "frame", "z", "y", "x"]].values.astype(np.float64)


def get_napari_scale(
    voxel_size: VoxelSize,
    include_time: bool = True,
) -> tuple[float, ...]:
    """Get napari scale from voxel size.

    Creates a scale tuple for napari layers to display
    physically correct aspect ratios.

    Parameters
    ----------
    voxel_size : VoxelSize
        Physical voxel dimensions
    include_time : bool
        Whether to include time dimension (scale=1)

    Returns
    -------
    tuple[float, ...]
        Scale tuple for napari layer
        (1, z_um, y_um, x_um) for 4D or (z_um, y_um, x_um) for 3D
    """
    if include_time:
        return (1.0, voxel_size.z_um, voxel_size.y_um, voxel_size.x_um)
    return (voxel_size.z_um, voxel_size.y_um, voxel_size.x_um)


def points_properties_from_detections(
    detections: pd.DataFrame,
    property_columns: list[str] | None = None,
) -> dict[str, NDArray]:
    """Extract properties from detections for napari Points layer.

    Parameters
    ----------
    detections : pd.DataFrame
        Detection DataFrame
    property_columns : list[str] | None
        Columns to include as properties.
        If None, uses ["mass", "size"] if available.

    Returns
    -------
    dict[str, NDArray]
        Properties dict for napari Points layer
    """
    if len(detections) == 0:
        return {}

    if property_columns is None:
        # Default properties
        property_columns = []
        for col in ["mass", "size", "signal", "ecc"]:
            if col in detections.columns:
                property_columns.append(col)

    properties = {}
    for col in property_columns:
        if col in detections.columns:
            properties[col] = detections[col].values

    return properties


def tracks_properties_from_tracks(
    tracks: pd.DataFrame,
    property_columns: list[str] | None = None,
) -> dict[str, NDArray]:
    """Extract properties from tracks for napari Tracks layer.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks DataFrame
    property_columns : list[str] | None
        Columns to include as properties.

    Returns
    -------
    dict[str, NDArray]
        Properties dict for napari Tracks layer
    """
    if len(tracks) == 0:
        return {}

    # Sort to match tracks array order
    sorted_tracks = tracks.sort_values(["particle", "frame"])

    if property_columns is None:
        property_columns = []
        for col in ["mass", "size", "signal"]:
            if col in sorted_tracks.columns:
                property_columns.append(col)

    properties = {}
    for col in property_columns:
        if col in sorted_tracks.columns:
            properties[col] = sorted_tracks[col].values

    return properties


def tracks_visualization_properties(
    tracks: pd.DataFrame,
    track_stats: pd.DataFrame | None = None,
    color_by: str = "track_id",
) -> tuple[dict[str, NDArray], str]:
    """Generate properties for napari Tracks layer with color mapping.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks DataFrame with particle, frame, z, y, x columns
    track_stats : pd.DataFrame | None
        Optional track statistics from compute_track_stats()
    color_by : str
        Property to use for coloring: "track_id", "time", "length"

    Returns
    -------
    tuple[dict[str, NDArray], str]
        (properties dict, color_by key for napari)
    """
    if len(tracks) == 0:
        return {}, "track_id"

    # Sort to match tracks array order
    sorted_tracks = tracks.sort_values(["particle", "frame"]).copy()

    properties: dict[str, NDArray] = {}

    # Always include track_id for coloring
    properties["track_id"] = sorted_tracks["particle"].values.astype(np.float64)

    # Time (frame) for coloring
    properties["time"] = sorted_tracks["frame"].values.astype(np.float64)

    # If track_stats provided, merge statistics
    if track_stats is not None and len(track_stats) > 0:
        merged = merge_track_stats_to_tracks(sorted_tracks, track_stats)
        if "length" in merged.columns:
            properties["length"] = merged["length"].values.astype(np.float64)
        if "mean_velocity_um" in merged.columns:
            properties["velocity"] = merged["mean_velocity_um"].values.astype(np.float64)
        if "total_displacement_um" in merged.columns:
            properties["displacement"] = merged["total_displacement_um"].values.astype(
                np.float64
            )

    # Map color_by to actual property name
    color_by_map = {
        "track_id": "track_id",
        "time": "time",
        "length": "length",
        "velocity": "velocity",
        "displacement": "displacement",
    }
    actual_color_by = color_by_map.get(color_by, "track_id")

    # Fallback if requested property not available
    if actual_color_by not in properties:
        actual_color_by = "track_id"

    return properties, actual_color_by


def merge_track_stats_to_tracks(
    tracks: pd.DataFrame,
    track_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-track statistics into tracks DataFrame.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks DataFrame with particle column
    track_stats : pd.DataFrame
        Track statistics with particle, length, mean_velocity_um, etc.

    Returns
    -------
    pd.DataFrame
        Tracks with statistics columns added
    """
    if len(tracks) == 0 or len(track_stats) == 0:
        return tracks

    # Select columns to merge (exclude particle as it's the key)
    stat_cols = [
        col for col in track_stats.columns if col != "particle" and col not in tracks.columns
    ]
    if not stat_cols:
        return tracks

    merge_cols = ["particle", *stat_cols]
    return tracks.merge(track_stats[merge_cols], on="particle", how="left")


def compute_xy_max_projection(
    image_data: NDArray[np.floating],
    tracks: pd.DataFrame | None = None,
) -> tuple[NDArray[np.floating], pd.DataFrame | None]:
    """Compute XY maximum intensity projection along Z axis.

    Parameters
    ----------
    image_data : NDArray
        4D image data (T, Z, Y, X)
    tracks : pd.DataFrame | None
        Optional tracks to project (z coordinate set to 0)

    Returns
    -------
    tuple[NDArray, pd.DataFrame | None]
        Projected image (T, Y, X) and optionally projected tracks
    """
    if image_data.ndim != 4:
        raise ValueError(f"Expected 4D data (T, Z, Y, X), got {image_data.ndim}D")

    # Max projection along Z axis (axis=1)
    projected = np.max(image_data, axis=1)

    # Project tracks if provided
    projected_tracks = None
    if tracks is not None and len(tracks) > 0:
        projected_tracks = tracks.copy()
        # Set z to 0 for 2D display
        projected_tracks["z"] = 0.0

    return projected, projected_tracks
