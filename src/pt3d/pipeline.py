"""Pipeline module for orchestrating the full tracking workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pt3d.detect import detect_batch, detect_streaming
from pt3d.exceptions import ProcessingError
from pt3d.export import export_config, export_detections, export_run_info, export_tracks
from pt3d.io import load_data
from pt3d.postprocess import compute_track_stats, postprocess
from pt3d.track import link_detections

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pt3d.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Results from running the tracking pipeline.

    Attributes
    ----------
    detections : pd.DataFrame
        All detected particles with columns [frame, z, y, x, mass, ...]
    tracks : pd.DataFrame
        Linked tracks with columns [particle, frame, z, y, x, ...]
    track_stats : pd.DataFrame
        Statistics for each track
    config : PipelineConfig
        Configuration used for this run
    input_info : dict[str, Any]
        Information about the input data
    start_time : datetime
        When the pipeline started
    end_time : datetime
        When the pipeline finished
    """

    detections: pd.DataFrame
    tracks: pd.DataFrame
    track_stats: pd.DataFrame
    config: PipelineConfig
    input_info: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def n_detections(self) -> int:
        """Total number of detections."""
        return len(self.detections)

    @property
    def n_tracks(self) -> int:
        """Number of tracks."""
        if "particle" not in self.tracks.columns:
            return 0
        return self.tracks["particle"].nunique()

    @property
    def duration_seconds(self) -> float:
        """Pipeline execution time in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    def summary(self) -> dict[str, Any]:
        """Get a summary of the results."""
        return {
            "n_detections": self.n_detections,
            "n_tracks": self.n_tracks,
            "n_frames": self.detections["frame"].nunique() if len(self.detections) > 0 else 0,
            "mean_track_length": self.track_stats["length"].mean() if len(self.track_stats) > 0 else 0,
            "duration_seconds": self.duration_seconds,
        }


def run_pipeline(
    data: NDArray[np.floating] | Path | str,
    config: PipelineConfig,
) -> PipelineResult:
    """Run the complete tracking pipeline.

    Parameters
    ----------
    data : NDArray[np.floating] | Path | str
        Array in (t, z, y, x) or (z, y, x) order, or a path to a file.
        Only file inputs use config.input.axis_order for normalization.
        For arrays in another order, call pt3d.io.load_array first.
    config : PipelineConfig
        Pipeline configuration. input.dtype and napari settings are stored
        but not applied. Convert dtype explicitly before calling.

    Returns
    -------
    PipelineResult
        Results including detections, tracks, and statistics

    Raises
    ------
    ProcessingError
        If dimension validation, detection, linking, or postprocessing fails.
        Loading errors may propagate as DataError or native I/O exceptions;
        export errors may also propagate without wrapping.
    """
    start_time = datetime.now(timezone.utc)
    logger.info("Starting tracking pipeline")

    # Load data
    if isinstance(data, np.ndarray):
        frames = data
        if frames.ndim == 3:
            # Single volume, add time dimension
            frames = frames[np.newaxis, ...]
        input_info = {
            "source": "array",
            "shape": frames.shape,
            "dtype": str(frames.dtype),
        }
    else:
        path = Path(data)
        frames = load_data(path, config.input.axis_order)
        input_info = {
            "source": str(path),
            "shape": frames.shape,
            "dtype": str(frames.dtype),
        }

    logger.info(f"Loaded data with shape {frames.shape}")

    # Validate dimensions
    if frames.ndim != 4:
        msg = f"Data must be 4D (t, z, y, x), got {frames.ndim}D"
        raise ProcessingError(msg)

    # Detection
    logger.info("Running detection...")
    detections = detect_batch(frames, config.detection, voxel_size=config.input.voxel_size)
    logger.info(f"Detected {len(detections)} particles")

    # Tracking
    logger.info("Running tracking...")
    tracks = link_detections(
        detections,
        config.tracking,
        config.input.voxel_size,
    )
    logger.info(f"Linked into {tracks['particle'].nunique()} tracks")

    # Postprocessing
    logger.info("Running postprocessing...")
    tracks = postprocess(tracks, config.postprocess, config.input.voxel_size)
    logger.info(f"After postprocessing: {tracks['particle'].nunique()} tracks")

    # Compute statistics
    track_stats = compute_track_stats(tracks, config.input.voxel_size)

    end_time = datetime.now(timezone.utc)

    result = PipelineResult(
        detections=detections,
        tracks=tracks,
        track_stats=track_stats,
        config=config,
        input_info=input_info,
        start_time=start_time,
        end_time=end_time,
    )

    # Export if configured
    if config.export is not None:
        export_results(result, config)

    logger.info(f"Pipeline completed in {result.duration_seconds:.2f}s")
    return result


def run_pipeline_streaming(
    frame_iterator,
    config: PipelineConfig,
    n_frames: int | None = None,
) -> PipelineResult:
    """Run the tracking pipeline with streaming input (memory efficient).

    This version accepts an iterator/generator instead of a 4D array,
    allowing processing of datasets that don't fit in memory.

    Parameters
    ----------
    frame_iterator : Iterable[NDArray[np.floating]]
        Iterator or generator yielding 3D volumes with shape (z, y, x).
    config : PipelineConfig
        Pipeline configuration. Input axis_order/dtype and napari preferences
        are not applied; yielded volumes must already have the desired dtype
        and (z, y, x) axis order.
    n_frames : int | None
        Number of frames (for logging). If None, not reported.

    Returns
    -------
    PipelineResult
        Results including detections, tracks, and statistics

    Raises
    ------
    ProcessingError
        If detection, linking, or postprocessing fails. Export errors may
        propagate without wrapping.

    Notes
    -----
    The frame iterator is consumed only once during detection.
    After detection, tracking and postprocessing work on the
    DataFrame of detections which requires less memory.
    """
    start_time = datetime.now(timezone.utc)
    logger.info("Starting streaming tracking pipeline")

    input_info = {
        "source": "iterator",
        "n_frames": n_frames,
    }

    if n_frames:
        logger.info(f"Processing {n_frames} frames in streaming mode")
    else:
        logger.info("Processing frames in streaming mode")

    # Detection (streaming)
    logger.info("Running streaming detection...")
    detections = detect_streaming(frame_iterator, config.detection, voxel_size=config.input.voxel_size)
    logger.info(f"Detected {len(detections)} particles")

    # Tracking
    logger.info("Running tracking...")
    tracks = link_detections(
        detections,
        config.tracking,
        config.input.voxel_size,
    )
    logger.info(f"Linked into {tracks['particle'].nunique()} tracks")

    # Postprocessing
    logger.info("Running postprocessing...")
    tracks = postprocess(tracks, config.postprocess, config.input.voxel_size)
    logger.info(f"After postprocessing: {tracks['particle'].nunique()} tracks")

    # Compute statistics
    track_stats = compute_track_stats(tracks, config.input.voxel_size)

    end_time = datetime.now(timezone.utc)

    result = PipelineResult(
        detections=detections,
        tracks=tracks,
        track_stats=track_stats,
        config=config,
        input_info=input_info,
        start_time=start_time,
        end_time=end_time,
    )

    # Export if configured
    if config.export is not None:
        export_results(result, config)

    logger.info(f"Streaming pipeline completed in {result.duration_seconds:.2f}s")
    return result


def export_results(
    result: PipelineResult,
    config: PipelineConfig,
) -> dict[str, Path]:
    """Export all results to files.

    Parameters
    ----------
    result : PipelineResult
        Pipeline results
    config : PipelineConfig
        Pipeline configuration (must have export config)

    Returns
    -------
    dict[str, Path]
        Paths to exported files
    """
    if config.export is None:
        msg = "Export configuration is required"
        raise ProcessingError(msg)

    output_dir = config.export.output_dir
    overwrite = config.export.overwrite

    paths = {}

    # Export detections
    paths["detections"] = export_detections(result.detections, config.export)

    # Export tracks
    paths["tracks"] = export_tracks(result.tracks, config.export)

    # Export configuration
    paths["config"] = export_config(config, output_dir, overwrite)

    # Export run info
    paths["run_info"] = export_run_info(
        output_dir=output_dir,
        config=config,
        input_info=result.input_info,
        results_summary=result.summary(),
        start_time=result.start_time,
        end_time=result.end_time,
        overwrite=overwrite,
    )

    logger.info(f"Exported results to {output_dir}")
    return paths
