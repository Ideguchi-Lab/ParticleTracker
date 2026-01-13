"""Configuration models for diffusion analysis.

All configuration is managed through Pydantic models for validation
and serialization.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MSDConfig(BaseModel):
    """Configuration for MSD computation.

    Attributes
    ----------
    max_lag_fraction : float
        Maximum time lag as fraction of track length.
    fit_range_fraction : tuple[float, float]
        Fitting range as (start, end) fraction of max_lag.
    """

    max_lag_fraction: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Maximum time lag as fraction of track length",
    )
    fit_range_fraction: tuple[float, float] = Field(
        default=(0.1, 0.5),
        description="Fitting range as (start, end) fraction of max_lag",
    )


class DiffusionAnalysisConfig(BaseModel):
    """Configuration for diffusion analysis.

    Attributes
    ----------
    min_track_length : int
        Minimum number of frames required for analysis.
    dt : float
        Time step between frames in seconds.
    msd : MSDConfig
        MSD computation settings.
    interpolate_gaps : bool
        Whether to interpolate missing frames in trajectories.
    """

    min_track_length: int = Field(
        default=10,
        ge=3,
        description="Minimum frames required for analysis",
    )
    dt: float = Field(
        default=1.0,
        gt=0,
        description="Time step between frames in seconds",
    )
    msd: MSDConfig = Field(default_factory=MSDConfig)
    interpolate_gaps: bool = Field(
        default=False,
        description="Interpolate missing frames in trajectories",
    )


class TrackVisualizationConfig(BaseModel):
    """Configuration for 3D track visualization.

    Attributes
    ----------
    color_by : Literal["D", "alpha", "particle"]
        Property to use for color mapping.
    colormap : str
        Matplotlib colormap name.
    point_size : float
        Size of trajectory points.
    line_width : float
        Width of trajectory lines.
    alpha : float
        Transparency (0-1).
    figsize : tuple[int, int]
        Figure size in inches.
    show_colorbar : bool
        Whether to show colorbar.
    elevation : float
        3D view elevation angle in degrees.
    azimuth : float
        3D view azimuth angle in degrees.
    """

    color_by: Literal["D", "alpha", "particle"] = Field(
        default="D",
        description="Color mapping: diffusion coefficient, exponent, or particle ID",
    )
    colormap: str = Field(
        default="viridis",
        description="Matplotlib colormap name",
    )
    point_size: float = Field(default=20.0, gt=0)
    line_width: float = Field(default=1.0, gt=0)
    alpha: float = Field(default=0.7, ge=0, le=1)
    figsize: tuple[int, int] = Field(default=(10, 8))
    show_colorbar: bool = Field(default=True)
    elevation: float = Field(default=20.0, description="3D view elevation angle")
    azimuth: float = Field(default=45.0, description="3D view azimuth angle")


class HistogramConfig(BaseModel):
    """Configuration for histogram plots.

    Attributes
    ----------
    n_bins : int
        Number of histogram bins.
    log_scale_D : bool
        Use log scale for D (x-axis).
    show_stats : bool
        Show mean/median annotations.
    figsize : tuple[int, int]
        Figure size in inches.
    """

    n_bins: int = Field(default=20, ge=5)
    log_scale_d: bool = Field(
        default=True,
        description="Use log scale for x-axis (D values)",
    )
    show_stats: bool = Field(
        default=True,
        description="Show mean/median annotations",
    )
    figsize: tuple[int, int] = Field(default=(10, 4))


class MSDPlotConfig(BaseModel):
    """Configuration for MSD vs time plots.

    Attributes
    ----------
    show_individual : bool
        Show individual particle MSDs.
    show_ensemble : bool
        Show ensemble average MSD.
    show_fit : bool
        Show power-law fit line.
    individual_alpha : float
        Transparency for individual curves.
    figsize : tuple[int, int]
        Figure size in inches.
    """

    show_individual: bool = Field(
        default=True,
        description="Show individual particle MSDs",
    )
    show_ensemble: bool = Field(
        default=True,
        description="Show ensemble average MSD",
    )
    show_fit: bool = Field(
        default=True,
        description="Show power-law fit line",
    )
    individual_alpha: float = Field(default=0.3, ge=0, le=1)
    figsize: tuple[int, int] = Field(default=(8, 6))
