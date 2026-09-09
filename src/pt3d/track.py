"""Tracking module wrapping trackpy for particle linking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import trackpy as tp

from pt3d.exceptions import ProcessingError
from pt3d.utils import apply_coordinate_scaling, search_range_um_to_scaled

if TYPE_CHECKING:
    from pt3d.config import TrackingConfig, VoxelSize

logger = logging.getLogger(__name__)


def link_detections(
    detections: pd.DataFrame,
    config: TrackingConfig,
    voxel_size: VoxelSize,
) -> pd.DataFrame:
    """Link detections across frames to form tracks.

    Uses trackpy.link_df with coordinate scaling to handle anisotropic
    voxel sizes. The search_range is specified in micrometers and
    automatically converted to scaled coordinate units.
    adaptive_stop is forwarded unchanged in those scaled units; it is not
    converted from micrometers.

    Parameters
    ----------
    detections : pd.DataFrame
        DataFrame with columns [frame, z, y, x, ...]
    config : TrackingConfig
        Tracking parameters (search_range_um in micrometers)
    voxel_size : VoxelSize
        Physical voxel size for coordinate scaling

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'particle' column (track_id)

    Raises
    ------
    ProcessingError
        If linking fails
    """
    if len(detections) == 0:
        logger.warning("No detections to link")
        result = detections.copy()
        result["particle"] = pd.Series(dtype=int)
        return result

    required_cols = ["frame", "z", "y", "x"]
    missing = [c for c in required_cols if c not in detections.columns]
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ProcessingError(msg)

    try:
        # Remove rows with NaN in position columns (trackpy sometimes produces these)
        df = detections.dropna(subset=["z", "y", "x"]).copy()
        if len(df) < len(detections):
            n_dropped = len(detections) - len(df)
            logger.warning(f"Dropped {n_dropped} detections with NaN coordinates")

        if len(df) == 0:
            logger.warning("No valid detections to link after removing NaN values")
            result = detections.copy()
            result["particle"] = pd.Series(dtype=int)
            return result

        # Apply coordinate scaling for isotropic distance calculation
        df = apply_coordinate_scaling(df, voxel_size)

        # Convert search_range from um to scaled units
        search_range_scaled = search_range_um_to_scaled(config.search_range_um, voxel_size)

        logger.debug(f"Linking with search_range={config.search_range_um}um ({search_range_scaled:.2f} scaled units)")

        # Use scaled coordinates for linking
        pos_columns = ["z_scaled", "y_scaled", "x_scaled"]

        # Prepare adaptive search parameters
        kwargs = {}
        if config.adaptive_stop is not None:
            kwargs["adaptive_stop"] = config.adaptive_stop
            kwargs["adaptive_step"] = config.adaptive_step

        linked = tp.link_df(
            df,
            search_range=search_range_scaled,
            memory=config.memory,
            pos_columns=pos_columns,
            **kwargs,
        )

        # Remove scaled columns (keep original coordinates)
        linked = linked.drop(columns=["z_scaled", "y_scaled", "x_scaled"], errors="ignore")

        n_tracks = linked["particle"].nunique()
        logger.info(f"Linked {len(linked)} detections into {n_tracks} tracks")

        return linked

    except Exception as e:
        msg = f"Linking failed: {e}"
        raise ProcessingError(msg) from e


def relabel_tracks(
    tracks: pd.DataFrame,
    mode: str = "dense",
) -> pd.DataFrame:
    """Relabel track IDs for consistency.

    Parameters
    ----------
    tracks : pd.DataFrame
        DataFrame with 'particle' column
    mode : str
        Relabeling mode:
        - "dense": Relabel to consecutive integers starting from 0
        - "sorted": Relabel by order of first appearance

    Returns
    -------
    pd.DataFrame
        DataFrame with relabeled 'particle' column
    """
    if "particle" not in tracks.columns:
        msg = "DataFrame must have 'particle' column"
        raise ProcessingError(msg)

    if len(tracks) == 0:
        return tracks.copy()

    df = tracks.copy()

    if mode == "dense":
        # Simple consecutive relabeling
        unique_ids = df["particle"].unique()
        id_map = {old: new for new, old in enumerate(sorted(unique_ids))}
        df["particle"] = df["particle"].map(id_map)

    elif mode == "sorted":
        # Relabel by order of first appearance
        first_appearance = df.groupby("particle")["frame"].min().sort_values()
        id_map = {old: new for new, old in enumerate(first_appearance.index)}
        df["particle"] = df["particle"].map(id_map)

    else:
        msg = f"Unknown relabeling mode: {mode}"
        raise ValueError(msg)

    return df
