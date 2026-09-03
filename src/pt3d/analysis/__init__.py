"""Diffusion analysis module for pt3d.

This module provides tools for analyzing diffusion properties from
particle tracking results, including MSD computation, diffusion
coefficient extraction (MSD = 6*D*t^alpha), and visualization.

Examples
--------
>>> from pt3d import run_pipeline, PipelineConfig
>>> from pt3d.analysis import analyze_diffusion, DiffusionAnalysisConfig
>>> from pt3d.analysis import (
...     plot_tracks_3d,
...     plot_diffusion_histograms,
...     plot_diffusion_scatter,
... )
>>>
>>> # Run tracking pipeline
>>> pipeline_config = PipelineConfig(...)  # configure as needed
>>> result = run_pipeline(data, pipeline_config)
>>>
>>> # Analyze diffusion
>>> analysis_config = DiffusionAnalysisConfig(dt=0.1, min_track_length=10)
>>> analysis = analyze_diffusion(result, analysis_config)
>>>
>>> print(f"Mean D = {analysis.mean_d:.4f} um^2/s^alpha")
>>> print(f"Mean alpha = {analysis.mean_alpha:.2f}")
>>>
>>> # Visualize
>>> fig, ax = plot_tracks_3d(analysis)
>>> fig, (ax_d, ax_a) = plot_diffusion_histograms(analysis)
>>> fig, (ax_sc, ax_ha) = plot_diffusion_scatter(analysis)
"""

from pt3d.analysis.brownian import (
    DiffusionAnalysisResult,
    ParticleDiffusionResult,
    analyze_diffusion,
    analyze_single_track,
    compute_ensemble_msd,
    compute_msd_with_gaps,
    interpolate_gaps,
    tracks_to_positions,
)
from pt3d.analysis.config import (
    DiffusionAnalysisConfig,
    DiffusionScatterConfig,
    HistogramConfig,
    MSDConfig,
    MSDPlotConfig,
    TrackVisualizationConfig,
)
from pt3d.analysis.visualization import (
    create_analysis_report,
    plot_diffusion_histograms,
    plot_diffusion_scatter,
    plot_msd_loglog,
    plot_tracks_3d,
)

__all__ = [
    "DiffusionAnalysisConfig",
    "DiffusionAnalysisResult",
    "DiffusionScatterConfig",
    "HistogramConfig",
    "MSDConfig",
    "MSDPlotConfig",
    "ParticleDiffusionResult",
    "TrackVisualizationConfig",
    "analyze_diffusion",
    "analyze_single_track",
    "compute_ensemble_msd",
    "compute_msd_with_gaps",
    "create_analysis_report",
    "interpolate_gaps",
    "plot_diffusion_histograms",
    "plot_diffusion_scatter",
    "plot_msd_loglog",
    "plot_tracks_3d",
    "tracks_to_positions",
]
