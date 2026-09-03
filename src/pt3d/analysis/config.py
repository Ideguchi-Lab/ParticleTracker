"""Configuration models for diffusion analysis.

All configuration is managed through Pydantic models for validation
and serialization.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("fit_range_fraction")
    @classmethod
    def validate_fit_range_fraction(
        cls,
        value: tuple[float, float],
    ) -> tuple[float, float]:
        start, end = value
        if not (0.0 <= start < end <= 1.0):
            msg = f"fit_range_fraction must satisfy 0.0 <= start < end <= 1.0, got (start={start}, end={end})"
            raise ValueError(msg)
        return value


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
        Property to use for color mapping. ``D`` refers to generalized
        diffusion coefficient from MSD = 6*D*t^alpha.
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
    tick_nbins : int
        Maximum number of ticks per axis to avoid text overlap.
    show_labels : bool
        Whether to show axis labels.
    show_title : bool
        Whether to show plot title.
    label_fontsize : float | None
        Font size for axis labels (None = matplotlib default).
    title_fontsize : float | None
        Font size for plot title (None = matplotlib default).
    tick_fontsize : float | None
        Font size for tick labels (None = matplotlib default).
    colorbar_fontsize : float | None
        Font size for colorbar label (None = matplotlib default).
    """

    color_by: Literal["D", "alpha", "particle"] = Field(
        default="D",
        description="Color mapping: generalized coefficient D, exponent, or particle ID",
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
    tick_nbins: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Maximum number of ticks per axis",
    )
    # Display options
    show_labels: bool = Field(default=True, description="Show axis labels")
    show_title: bool = Field(default=True, description="Show plot title")
    # Font size options
    label_fontsize: float | None = Field(default=None, description="Axis label font size")
    title_fontsize: float | None = Field(default=None, description="Title font size")
    tick_fontsize: float | None = Field(default=None, description="Tick label font size")
    colorbar_fontsize: float | None = Field(default=None, description="Colorbar label font size")


class HistogramConfig(BaseModel):
    """Configuration for histogram plots.

    Attributes
    ----------
    n_bins : int
        Number of histogram bins.
    log_scale_d : bool
        Use log scale for generalized D (x-axis).
    show_stats : bool
        Show mean/median annotations.
    show_alpha_reference_line : bool
        Show reference line at alpha=1 (normal diffusion boundary).
    figsize : tuple[int, int]
        Figure size in inches.
    show_labels : bool
        Whether to show axis labels.
    show_title : bool
        Whether to show plot title.
    label_fontsize : float | None
        Font size for axis labels.
    title_fontsize : float | None
        Font size for plot title.
    tick_fontsize : float | None
        Font size for tick labels.
    stats_fontsize : float | None
        Font size for statistics annotation.
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
    show_alpha_reference_line: bool = Field(
        default=True,
        description="Show reference line at alpha=1 (normal diffusion boundary)",
    )
    figsize: tuple[int, int] = Field(default=(10, 4))
    # Display options
    show_labels: bool = Field(default=True, description="Show axis labels")
    show_title: bool = Field(default=True, description="Show plot title")
    # Font size options
    label_fontsize: float | None = Field(default=None, description="Axis label font size")
    title_fontsize: float | None = Field(default=None, description="Title font size")
    tick_fontsize: float | None = Field(default=None, description="Tick label font size")
    stats_fontsize: float | None = Field(default=None, description="Statistics annotation font size")


class DiffusionScatterConfig(BaseModel):
    """Configuration for D-alpha scatter plots with right-side alpha histogram.

    Attributes
    ----------
    n_bins : int
        Number of bins for alpha histogram.
    figsize : tuple[int, int]
        Figure size in inches.
    show_stats : bool
        Show summary statistics annotation on the scatter axis.
    show_alpha_reference_line : bool
        Show reference line at alpha=1.
    show_labels : bool
        Whether to show axis labels.
    show_title : bool
        Whether to show plot title.
    label_fontsize : float | None
        Font size for axis labels.
    title_fontsize : float | None
        Font size for plot title.
    tick_fontsize : float | None
        Font size for tick labels.
    stats_fontsize : float | None
        Font size for statistics annotation.
    scatter_marker_size : float
        Marker area for scatter points.
    scatter_alpha : float
        Marker transparency.
    scatter_edgecolor : str
        Marker edge color.
    scatter_facecolor : str
        Marker face color.
    scatter_linewidth : float
        Marker edge line width.
    hist_color : str
        Marginal histogram bar color.
    hist_edgecolor : str
        Marginal histogram bar edge color.
    hist_alpha : float
        Marginal histogram bar transparency.
    """

    n_bins: int = Field(default=20, ge=5)
    figsize: tuple[int, int] = Field(default=(10, 5))
    show_stats: bool = Field(default=False, description="Show summary statistics annotation")
    show_alpha_reference_line: bool = Field(
        default=True,
        description="Show reference line at alpha=1",
    )
    # Display options
    show_labels: bool = Field(default=True, description="Show axis labels")
    show_title: bool = Field(default=True, description="Show plot title")
    # Font size options
    label_fontsize: float | None = Field(default=None, description="Axis label font size")
    title_fontsize: float | None = Field(default=None, description="Title font size")
    tick_fontsize: float | None = Field(default=None, description="Tick label font size")
    stats_fontsize: float | None = Field(default=None, description="Statistics annotation font size")
    # Scatter style
    scatter_marker_size: float = Field(default=25.0, gt=0)
    scatter_alpha: float = Field(default=1.0, ge=0, le=1)
    scatter_edgecolor: str = Field(default="#1f7a8c")
    scatter_facecolor: str = Field(default="#d0f0f5")
    scatter_linewidth: float = Field(default=1.0, ge=0)
    # Marginal histogram style
    hist_color: str = Field(default="#90be6d")
    hist_edgecolor: str = Field(default="#3a5a40")
    hist_alpha: float = Field(default=0.9, ge=0, le=1)


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
    show_labels : bool
        Whether to show axis labels.
    show_title : bool
        Whether to show plot title.
    show_legend : bool
        Whether to show legend.
    label_fontsize : float | None
        Font size for axis labels.
    title_fontsize : float | None
        Font size for plot title.
    tick_fontsize : float | None
        Font size for tick labels.
    legend_fontsize : float | None
        Font size for legend.
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
    # Display options
    show_labels: bool = Field(default=True, description="Show axis labels")
    show_title: bool = Field(default=True, description="Show plot title")
    show_legend: bool = Field(default=True, description="Show legend")
    # Font size options
    label_fontsize: float | None = Field(default=None, description="Axis label font size")
    title_fontsize: float | None = Field(default=None, description="Title font size")
    tick_fontsize: float | None = Field(default=None, description="Tick label font size")
    legend_fontsize: float | None = Field(default=None, description="Legend font size")
