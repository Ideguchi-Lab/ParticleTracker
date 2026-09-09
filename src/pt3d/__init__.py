"""pt3d - 3D+t Particle Tracking Library.

A library for particle detection and tracking in 3D time-series microscopy data,
using trackpy as the processing engine and napari for visualization.
"""

from pt3d.config import (
    DetectionConfig,
    ExportConfig,
    InputConfig,
    PipelineConfig,
    PostprocessConfig,
    TrackingConfig,
    VoxelSize,
)
from pt3d.detect import detect_batch, detect_frame, detect_streaming
from pt3d.pipeline import PipelineResult, run_pipeline, run_pipeline_streaming
from pt3d.track import link_detections

__version__ = "1.0.0"

__all__ = [
    # Config
    "VoxelSize",
    "InputConfig",
    "DetectionConfig",
    "TrackingConfig",
    "PostprocessConfig",
    "ExportConfig",
    "PipelineConfig",
    # Detection
    "detect_frame",
    "detect_batch",
    "detect_streaming",
    # Tracking
    "link_detections",
    # Pipeline
    "run_pipeline",
    "run_pipeline_streaming",
    "PipelineResult",
]
