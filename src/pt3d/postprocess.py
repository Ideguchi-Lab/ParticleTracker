"""Postprocessing module for track filtering and statistics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pt3d.exceptions import ProcessingError

if TYPE_CHECKING:
    from pt3d.config import PostprocessConfig, VoxelSize

logger = logging.getLogger(__name__)


def filter_stubs(
    tracks: pd.DataFrame,
    min_length: int,
) -> pd.DataFrame:
    """Remove tracks shorter than minimum length.

    Parameters
    ----------
    tracks : pd.DataFrame
        DataFrame with 'particle' column
    min_length : int
        Minimum number of frames for a valid track

    Returns
    -------
    pd.DataFrame
        DataFrame with short tracks removed
    """
    if "particle" not in tracks.columns:
        msg = "DataFrame must have 'particle' column"
        raise ProcessingError(msg)

    if len(tracks) == 0:
        return tracks.copy()

    # Count points per track
    track_lengths = tracks.groupby("particle").size()
    valid_tracks = track_lengths[track_lengths >= min_length].index

    filtered = tracks[tracks["particle"].isin(valid_tracks)].copy()

    n_removed = tracks["particle"].nunique() - filtered["particle"].nunique()
    if n_removed > 0:
        logger.info(f"Removed {n_removed} tracks shorter than {min_length} frames")

    return filtered


def compute_velocities(
    tracks: pd.DataFrame,
    voxel_size: VoxelSize,
) -> pd.DataFrame:
    """Compute instantaneous velocities for each track point.

    Parameters
    ----------
    tracks : pd.DataFrame
        DataFrame with 'particle', 'frame', 'z', 'y', 'x' columns
    voxel_size : VoxelSize
        Physical voxel dimensions for µm conversion

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'velocity_um' column (µm/frame)
    """
    if len(tracks) == 0:
        result = tracks.copy()
        result["velocity_um"] = pd.Series(dtype=float)
        return result

    df = tracks.copy().sort_values(["particle", "frame"])

    # Compute displacements in physical units
    df["dz_um"] = df.groupby("particle")["z"].diff() * voxel_size.z_um
    df["dy_um"] = df.groupby("particle")["y"].diff() * voxel_size.y_um
    df["dx_um"] = df.groupby("particle")["x"].diff() * voxel_size.x_um

    # Compute 3D velocity magnitude
    df["velocity_um"] = np.sqrt(
        df["dz_um"] ** 2 + df["dy_um"] ** 2 + df["dx_um"] ** 2
    )

    # Clean up intermediate columns
    df = df.drop(columns=["dz_um", "dy_um", "dx_um"])

    return df


def filter_by_velocity(
    tracks: pd.DataFrame,
    max_velocity_um: float,
    voxel_size: VoxelSize,
) -> pd.DataFrame:
    """Remove track points with velocity exceeding threshold.

    This can help remove tracking errors where a particle "jumps"
    to an incorrect detection.

    Parameters
    ----------
    tracks : pd.DataFrame
        DataFrame with 'particle', 'frame', 'z', 'y', 'x' columns
    max_velocity_um : float
        Maximum allowed velocity in µm/frame
    voxel_size : VoxelSize
        Physical voxel dimensions

    Returns
    -------
    pd.DataFrame
        DataFrame with high-velocity points removed
    """
    df = compute_velocities(tracks, voxel_size)

    # Keep points with velocity below threshold (or NaN for first points)
    mask = (df["velocity_um"] <= max_velocity_um) | df["velocity_um"].isna()
    filtered = df[mask].drop(columns=["velocity_um"])

    n_removed = len(df) - len(filtered)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} points exceeding velocity threshold")

    return filtered


def compute_track_stats(
    tracks: pd.DataFrame,
    voxel_size: VoxelSize,
) -> pd.DataFrame:
    """Compute statistics for each track.

    Parameters
    ----------
    tracks : pd.DataFrame
        DataFrame with 'particle', 'frame', 'z', 'y', 'x' columns
    voxel_size : VoxelSize
        Physical voxel dimensions

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per track containing:
        - particle: track ID
        - length: number of frames
        - duration: last_frame - first_frame
        - mean_velocity_um: average velocity in µm/frame
        - max_velocity_um: maximum velocity in µm/frame
        - total_displacement_um: start-to-end distance in µm
    """
    if len(tracks) == 0:
        return pd.DataFrame(
            columns=[
                "particle",
                "length",
                "duration",
                "mean_velocity_um",
                "max_velocity_um",
                "total_displacement_um",
            ]
        )

    # Add velocities
    df = compute_velocities(tracks, voxel_size)

    stats_list = []
    for particle_id, group in df.groupby("particle"):
        group = group.sort_values("frame")

        # Basic statistics
        length = len(group)
        duration = group["frame"].max() - group["frame"].min()

        # Velocity statistics (excluding first NaN)
        velocities = group["velocity_um"].dropna()
        mean_vel = velocities.mean() if len(velocities) > 0 else 0.0
        max_vel = velocities.max() if len(velocities) > 0 else 0.0

        # Total displacement (start to end)
        start = group.iloc[0]
        end = group.iloc[-1]
        total_disp = np.sqrt(
            ((end["z"] - start["z"]) * voxel_size.z_um) ** 2
            + ((end["y"] - start["y"]) * voxel_size.y_um) ** 2
            + ((end["x"] - start["x"]) * voxel_size.x_um) ** 2
        )

        stats_list.append(
            {
                "particle": particle_id,
                "length": length,
                "duration": duration,
                "mean_velocity_um": mean_vel,
                "max_velocity_um": max_vel,
                "total_displacement_um": total_disp,
            }
        )

    return pd.DataFrame(stats_list)


def postprocess(
    tracks: pd.DataFrame,
    config: PostprocessConfig,
    voxel_size: VoxelSize,
) -> pd.DataFrame:
    """Apply all postprocessing steps.

    Parameters
    ----------
    tracks : pd.DataFrame
        DataFrame with 'particle', 'frame', 'z', 'y', 'x' columns
    config : PostprocessConfig
        Postprocessing configuration
    voxel_size : VoxelSize
        Physical voxel dimensions

    Returns
    -------
    pd.DataFrame
        Postprocessed tracks
    """
    result = tracks.copy()

    # Filter by velocity if configured
    if config.max_velocity_um is not None:
        result = filter_by_velocity(result, config.max_velocity_um, voxel_size)

    # Filter short tracks
    result = filter_stubs(result, config.min_track_length)

    return result
