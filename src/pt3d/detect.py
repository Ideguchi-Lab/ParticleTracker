"""Detection module wrapping trackpy for 3D particle detection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import trackpy as tp

from pt3d.exceptions import ProcessingError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from pt3d.config import DetectionConfig, VoxelSize

logger = logging.getLogger(__name__)


def detect_frame(
    volume: NDArray[np.floating],
    config: DetectionConfig,
    voxel_size: VoxelSize | None = None,
) -> pd.DataFrame:
    """Detect particles in a single 3D volume.

    Uses trackpy.locate for particle detection in a single frame.

    Parameters
    ----------
    volume : NDArray[np.floating]
        3D array with shape (z, y, x)
    config : DetectionConfig
        Detection parameters
    voxel_size : VoxelSize | None
        Physical voxel dimensions (required if diameter_um is used)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [z, y, x, mass, ...]. trackpy supplies the
        additional columns; anisotropic diameters produce size_z/size_y/size_x
        and may produce ep_z/ep_y/ep_x instead of scalar size and ep.
        No detections returns an empty DataFrame with columns
        [z, y, x, mass, size, ecc, signal, raw_mass, ep].

    Raises
    ------
    ProcessingError
        If detection fails
    """
    if volume.ndim != 3:
        msg = f"Volume must be 3D, got {volume.ndim}D"
        raise ProcessingError(msg)

    try:
        # Get diameter in pixels (convert from um if necessary)
        if config.diameter_um is not None:
            if voxel_size is None:
                msg = "voxel_size is required when using diameter_um"
                raise ProcessingError(msg)
            diameter = list(config.get_diameter_pixels(voxel_size))
            logger.debug(f"Converted diameter_um={config.diameter_um} to pixels: {diameter}")
        else:
            assert config.diameter is not None
            diameter = list(config.diameter)

        # Prepare separation if provided
        separation = list(config.separation) if config.separation else None

        features = tp.locate(
            volume,
            diameter=diameter,
            minmass=config.minmass,
            threshold=config.threshold,
            separation=separation,
            invert=config.invert,
            preprocess=config.preprocess,
        )

        if features is None or len(features) == 0:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["z", "y", "x", "mass", "size", "ecc", "signal", "raw_mass", "ep"])

        logger.debug(f"Detected {len(features)} particles")
        return features

    except Exception as e:
        msg = f"Detection failed: {e}"
        raise ProcessingError(msg) from e


def detect_batch(
    frames: NDArray[np.floating],
    config: DetectionConfig,
    frame_range: tuple[int, int] | None = None,
    voxel_size: VoxelSize | None = None,
) -> pd.DataFrame:
    """Detect particles in multiple frames.

    Uses trackpy.batch for efficient multi-frame detection.

    Parameters
    ----------
    frames : NDArray[np.floating]
        4D array with shape (t, z, y, x)
    config : DetectionConfig
        Detection parameters
    frame_range : tuple[int, int] | None
        Optional (start, end) frame range for subset processing
        End is exclusive (like Python slicing)
    voxel_size : VoxelSize | None
        Physical voxel dimensions (required if diameter_um is used)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [frame, z, y, x, mass, ...]. Additional columns
        depend on trackpy and the diameter; see detect_frame.
        No detections returns an empty DataFrame with columns
        [frame, z, y, x, mass, size, ecc, signal, raw_mass, ep].

    Raises
    ------
    ProcessingError
        If detection fails
    """
    if frames.ndim != 4:
        msg = f"Frames must be 4D (t, z, y, x), got {frames.ndim}D"
        raise ProcessingError(msg)

    # Handle subset
    frame_offset = 0
    if frame_range is not None:
        start, end = frame_range
        if start < 0 or end > frames.shape[0] or start >= end:
            msg = f"Invalid frame_range {frame_range} for {frames.shape[0]} frames"
            raise ProcessingError(msg)
        frames = frames[start:end]
        frame_offset = start
        logger.info(f"Processing frames {start} to {end}")

    try:
        # Get diameter in pixels (convert from um if necessary)
        if config.diameter_um is not None:
            if voxel_size is None:
                msg = "voxel_size is required when using diameter_um"
                raise ProcessingError(msg)
            diameter = list(config.get_diameter_pixels(voxel_size))
            logger.info(f"Converted diameter_um={config.diameter_um} um to pixels: {diameter}")
        else:
            assert config.diameter is not None
            diameter = list(config.diameter)

        separation = list(config.separation) if config.separation else None

        features = tp.batch(
            frames,
            diameter=diameter,
            minmass=config.minmass,
            threshold=config.threshold,
            separation=separation,
            invert=config.invert,
            preprocess=config.preprocess,
        )

        if features is None or len(features) == 0:
            return pd.DataFrame(columns=["frame", "z", "y", "x", "mass", "size", "ecc", "signal", "raw_mass", "ep"])

        # Adjust frame numbers if processing a subset
        if frame_offset > 0:
            features["frame"] = features["frame"] + frame_offset

        logger.info(f"Detected {len(features)} particles across {features['frame'].nunique()} frames")
        return features

    except Exception as e:
        msg = f"Batch detection failed: {e}"
        raise ProcessingError(msg) from e


def detect_streaming(
    frame_iterator: Iterable[NDArray[np.floating]],
    config: DetectionConfig,
    voxel_size: VoxelSize | None = None,
) -> pd.DataFrame:
    """Detect particles from a streaming iterator (memory efficient).

    Processes frames one at a time using tp.locate, keeping memory usage low.

    Parameters
    ----------
    frame_iterator : Iterable[NDArray[np.floating]]
        Iterator or generator yielding 3D volumes with shape (z, y, x).
        Can be a generator function that yields frames one at a time.
    config : DetectionConfig
        Detection parameters
    voxel_size : VoxelSize | None
        Physical voxel dimensions (required if diameter_um is used)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [frame, z, y, x, mass, ...]. Additional columns
        depend on trackpy and the diameter; see detect_frame.
        Frames are numbered from zero in iterator order, including empty frames.
        No detections returns an empty DataFrame with columns
        [frame, z, y, x, mass, size, ecc, signal, raw_mass, ep].

    Raises
    ------
    ProcessingError
        If detection fails

    Notes
    -----
    This function is designed for large datasets. The iterator yields
    one 3D volume at a time, keeping memory usage low.
    Each frame is processed individually with tp.locate and results are concatenated.
    """
    try:
        # Get diameter in pixels (convert from um if necessary)
        if config.diameter_um is not None:
            if voxel_size is None:
                msg = "voxel_size is required when using diameter_um"
                raise ProcessingError(msg)
            diameter = list(config.get_diameter_pixels(voxel_size))
            logger.info(f"Converted diameter_um={config.diameter_um} um to pixels: {diameter}")
        else:
            assert config.diameter is not None
            diameter = list(config.diameter)

        separation = list(config.separation) if config.separation else None

        logger.info("Running streaming detection (frame by frame)...")

        all_features = []
        for frame_idx, volume in enumerate(frame_iterator):
            if volume.ndim != 3:
                msg = f"Volume must be 3D, got {volume.ndim}D at frame {frame_idx}"
                raise ProcessingError(msg)

            features = tp.locate(
                volume,
                diameter=diameter,
                minmass=config.minmass,
                threshold=config.threshold,
                separation=separation,
                invert=config.invert,
                preprocess=config.preprocess,
            )

            if features is not None and len(features) > 0:
                features["frame"] = frame_idx
                all_features.append(features)

            if (frame_idx + 1) % 10 == 0:
                logger.debug(f"Processed frame {frame_idx + 1}")

        if not all_features:
            return pd.DataFrame(columns=["frame", "z", "y", "x", "mass", "size", "ecc", "signal", "raw_mass", "ep"])

        result = pd.concat(all_features, ignore_index=True)
        logger.info(f"Detected {len(result)} particles across {result['frame'].nunique()} frames")
        return result

    except Exception as e:
        msg = f"Streaming detection failed: {e}"
        raise ProcessingError(msg) from e


def detect_single_frame(
    frames: NDArray[np.floating],
    frame_index: int,
    config: DetectionConfig,
    voxel_size: VoxelSize | None = None,
) -> pd.DataFrame:
    """Detect particles in a single frame from a 4D array.

    Convenience function for detecting in one frame of a time series.

    Parameters
    ----------
    frames : NDArray[np.floating]
        4D array with shape (t, z, y, x)
    frame_index : int
        Index of the frame to process
    config : DetectionConfig
        Detection parameters
    voxel_size : VoxelSize | None
        Physical voxel dimensions (required if diameter_um is used)

    Returns
    -------
    pd.DataFrame
        Nonempty results have columns [frame, z, y, x, mass, ...], with
        additional columns as described in detect_frame. If no particles are
        found, returns detect_frame's empty schema without a frame column.
    """
    if frame_index < 0 or frame_index >= frames.shape[0]:
        msg = f"Frame index {frame_index} out of range [0, {frames.shape[0]})"
        raise ProcessingError(msg)

    volume = frames[frame_index]
    features = detect_frame(volume, config, voxel_size)

    # Add frame column
    if len(features) > 0:
        features["frame"] = frame_index

    return features
