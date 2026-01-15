"""Visualization functions for diffusion analysis results.

This module provides matplotlib-based visualization for diffusion
analysis results including 3D trajectory plots, histograms, and
MSD log-log plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pt3d.analysis.brownian import DiffusionAnalysisResult
    from pt3d.analysis.config import (
        HistogramConfig,
        MSDPlotConfig,
        TrackVisualizationConfig,
    )


def plot_tracks_3d(
    result: DiffusionAnalysisResult,
    config: TrackVisualizationConfig | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot 3D trajectories colored by diffusion properties.

    Parameters
    ----------
    result : DiffusionAnalysisResult
        Analysis result containing trajectories and D/alpha values.
    config : TrackVisualizationConfig | None
        Visualization configuration.
    ax : Axes | None
        Existing 3D axes. If None, creates new figure.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes objects.
    """
    from pt3d.analysis.config import TrackVisualizationConfig

    if config is None:
        config = TrackVisualizationConfig()

    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=config.figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if len(result.positions_um) == 0:
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
        ax.set_zlabel("Z (um)")
        ax.set_title("No tracks to display")
        return fig, ax

    # Get color values based on config
    particle_ids = list(result.positions_um.keys())

    if config.color_by == "D":
        color_values = [result.msd_per_particle[pid].D for pid in particle_ids]
        color_label = "D (um²/s)"
    elif config.color_by == "alpha":
        color_values = [result.msd_per_particle[pid].alpha for pid in particle_ids]
        color_label = "α"
    else:  # particle
        color_values = particle_ids
        color_label = "Particle ID"

    # Normalize colors
    vmin, vmax = min(color_values), max(color_values)
    if vmin == vmax:
        vmax = vmin + 1  # Avoid division by zero
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(config.colormap)

    # Plot each trajectory
    for pid, c_val in zip(particle_ids, color_values, strict=True):
        positions = result.positions_um[pid]
        color = cmap(norm(c_val))

        valid_mask = ~np.any(np.isnan(positions), axis=1)
        if not np.any(valid_mask):
            continue

        # Split into contiguous valid segments so gaps don't connect lines
        mask_int = valid_mask.astype(np.int8)
        changes = np.diff(mask_int)
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if valid_mask[0]:
            starts = np.concatenate(([0], starts))
        if valid_mask[-1]:
            ends = np.concatenate((ends, [len(valid_mask)]))

        for start_idx, end_idx in zip(starts, ends, strict=True):
            segment = positions[start_idx:end_idx]
            if len(segment) < 2:
                continue

            # Extract coordinates (positions are in [z, y, x] order)
            z = segment[:, 0]
            y = segment[:, 1]
            x = segment[:, 2]

            ax.plot(
                x,
                y,
                z,
                color=color,
                linewidth=config.line_width,
                alpha=config.alpha,
            )

        # Plot start point (first valid position)
        first_valid = int(np.flatnonzero(valid_mask)[0])
        z0, y0, x0 = positions[first_valid]
        ax.scatter(
            [x0],
            [y0],
            [z0],
            color=color,
            s=config.point_size,
            marker="o",
            alpha=config.alpha,
        )

    # Set labels
    ax.set_xlabel("X (um)")
    ax.set_ylabel("Y (um)")
    ax.set_zlabel("Z (um)")
    ax.set_title(f"3D Trajectories (colored by {color_label})")

    # Set view angle
    ax.view_init(elev=config.elevation, azim=config.azimuth)

    # Add colorbar
    if config.show_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label(color_label)

    return fig, ax


def plot_diffusion_histograms(
    result: DiffusionAnalysisResult,
    config: HistogramConfig | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot histograms of diffusion coefficient and exponent.

    Creates side-by-side histograms for D and alpha distributions.

    Parameters
    ----------
    result : DiffusionAnalysisResult
        Analysis result containing particle statistics.
    config : HistogramConfig | None
        Histogram configuration.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes]]
        Figure and (ax_D, ax_alpha) axes.
    """
    from pt3d.analysis.config import HistogramConfig

    if config is None:
        config = HistogramConfig()

    fig, (ax_d, ax_alpha) = plt.subplots(1, 2, figsize=config.figsize)

    if len(result.particle_results) == 0:
        ax_d.set_title("D Distribution (no data)")
        ax_alpha.set_title("α Distribution (no data)")
        return fig, (ax_d, ax_alpha)

    d_values = result.particle_results["D"].values
    alpha_values = result.particle_results["alpha"].values

    # Filter out non-positive D values for log scale
    d_positive = d_values[d_values > 0]

    # Plot D histogram
    if config.log_scale_d and len(d_positive) > 0:
        # Use log bins
        log_min = np.floor(np.log10(d_positive.min()))
        log_max = np.ceil(np.log10(d_positive.max()))
        if log_min == log_max:
            log_max = log_min + 1  # Ensure increasing bin edges
        bins = np.logspace(log_min, log_max, config.n_bins + 1)
        ax_d.hist(d_positive, bins=bins, edgecolor="black", alpha=0.7)
        ax_d.set_xscale("log")
    else:
        ax_d.hist(d_values, bins=config.n_bins, edgecolor="black", alpha=0.7)

    ax_d.set_xlabel("D (um²/s)")
    ax_d.set_ylabel("Count")
    ax_d.set_title("Diffusion Coefficient Distribution")

    # Add statistics annotation for D
    if config.show_stats and len(d_values) > 0:
        mean_d = np.mean(d_values)
        median_d = np.median(d_values)
        stats_text = f"Mean: {mean_d:.4f}\nMedian: {median_d:.4f}"
        ax_d.annotate(
            stats_text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    # Plot alpha histogram
    ax_alpha.hist(alpha_values, bins=config.n_bins, edgecolor="black", alpha=0.7)
    ax_alpha.set_xlabel("α (diffusion exponent)")
    ax_alpha.set_ylabel("Count")
    ax_alpha.set_title("Diffusion Exponent Distribution")

    # Add reference line at alpha=1 (normal diffusion)
    ax_alpha.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="α=1")
    ax_alpha.legend()

    # Add statistics annotation for alpha
    if config.show_stats and len(alpha_values) > 0:
        mean_alpha = np.mean(alpha_values)
        median_alpha = np.median(alpha_values)
        stats_text = f"Mean: {mean_alpha:.2f}\nMedian: {median_alpha:.2f}"
        ax_alpha.annotate(
            stats_text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    fig.tight_layout()
    return fig, (ax_d, ax_alpha)


def plot_msd_loglog(
    result: DiffusionAnalysisResult,
    config: MSDPlotConfig | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot MSD vs time delay on log-log scale.

    Shows individual particle MSDs, ensemble average, and power-law fit.

    Parameters
    ----------
    result : DiffusionAnalysisResult
        Analysis result containing MSD data.
    config : MSDPlotConfig | None
        Plot configuration.
    ax : Axes | None
        Existing axes. If None, creates new figure.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes objects.

    Notes
    -----
    The log-log plot is useful for identifying diffusion regimes:
    - Slope = 1: Normal diffusion
    - Slope < 1: Subdiffusion (confined)
    - Slope > 1: Superdiffusion (directed)
    """
    from pt3d.analysis.config import MSDPlotConfig

    if config is None:
        config = MSDPlotConfig()

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.get_figure()

    if len(result.msd_per_particle) == 0:
        ax.set_xlabel("Time lag (s)")
        ax.set_ylabel("MSD (um²)")
        ax.set_title("MSD vs Time (no data)")
        return fig, ax

    # Plot individual particle MSDs
    if config.show_individual:
        for _pid, particle_result in result.msd_per_particle.items():
            msd = particle_result.msd
            time_lags = particle_result.time_lags

            # Filter out zero/negative values for log scale
            valid_mask = (msd > 0) & (time_lags > 0)
            if not np.any(valid_mask):
                continue

            ax.plot(
                time_lags[valid_mask],
                msd[valid_mask],
                alpha=config.individual_alpha,
                linewidth=0.5,
                color="gray",
            )

    # Plot ensemble average
    if config.show_ensemble and result.ensemble_msd is not None:
        valid_mask = (result.ensemble_msd > 0) & (result.ensemble_time_lags > 0)
        if np.any(valid_mask):
            ax.plot(
                result.ensemble_time_lags[valid_mask],
                result.ensemble_msd[valid_mask],
                "b-",
                linewidth=2,
                label=f"Ensemble (n={result.n_particles})",
            )

    # Plot reference lines for different diffusion regimes
    if config.show_fit and result.ensemble_time_lags is not None:
        # Get valid time range
        valid_mask = result.ensemble_time_lags > 0
        if np.any(valid_mask):
            t_range = result.ensemble_time_lags[valid_mask]
            t_min, t_max = t_range.min(), t_range.max()

            # Reference line for alpha=1 (normal diffusion)
            mean_d = result.mean_d
            if mean_d > 0:
                t_ref = np.linspace(t_min, t_max, 100)
                msd_ref = 6 * mean_d * t_ref  # 3D diffusion: MSD = 6*D*t
                ax.plot(
                    t_ref,
                    msd_ref,
                    "r--",
                    linewidth=1.5,
                    label=f"α=1 fit (D={mean_d:.4f})",
                )

    # Set log scale
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Labels and title
    ax.set_xlabel("Time lag (s)")
    ax.set_ylabel("MSD (um²)")

    mean_alpha = result.mean_alpha
    ax.set_title(f"MSD vs Time (mean α = {mean_alpha:.2f})")

    ax.legend(loc="upper left")
    ax.grid(True, which="both", ls="-", alpha=0.3)

    return fig, ax


def create_analysis_report(
    result: DiffusionAnalysisResult,
    output_path: str | None = None,
) -> Figure:
    """Create a comprehensive multi-panel analysis report.

    Generates a figure with:
    - 3D trajectory plot (colored by D)
    - D and alpha histograms
    - MSD log-log plot
    - Summary statistics annotations

    Parameters
    ----------
    result : DiffusionAnalysisResult
        Complete analysis result.
    output_path : str | None
        If provided, saves figure to this path.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    from pt3d.analysis.config import (
        MSDPlotConfig,
        TrackVisualizationConfig,
    )

    fig = plt.figure(figsize=(16, 12))

    # 3D trajectories (top left)
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    track_config = TrackVisualizationConfig(figsize=(8, 6), show_colorbar=True)
    plot_tracks_3d(result, config=track_config, ax=ax_3d)

    # MSD log-log (top right)
    ax_msd = fig.add_subplot(2, 2, 2)
    msd_config = MSDPlotConfig()
    plot_msd_loglog(result, config=msd_config, ax=ax_msd)

    # D histogram (bottom left)
    ax_d = fig.add_subplot(2, 2, 3)
    if len(result.particle_results) > 0:
        d_values = result.particle_results["D"].values
        d_positive = d_values[d_values > 0]
        if len(d_positive) > 0:
            log_min = np.floor(np.log10(d_positive.min()))
            log_max = np.ceil(np.log10(d_positive.max()))
            if log_min == log_max:
                log_max = log_min + 1  # Ensure increasing bin edges
            bins = np.logspace(log_min, log_max, 21)
            ax_d.hist(d_positive, bins=bins, edgecolor="black", alpha=0.7)
            ax_d.set_xscale("log")
    ax_d.set_xlabel("D (um²/s)")
    ax_d.set_ylabel("Count")
    ax_d.set_title("Diffusion Coefficient Distribution")

    # Add statistics
    if len(result.particle_results) > 0:
        summary = result.summary()
        stats_text = f"Mean D: {summary['D_mean']:.4f} um²/s\nMedian D: {summary['D_median']:.4f} um²/s"
        ax_d.annotate(
            stats_text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    # Alpha histogram (bottom right)
    ax_alpha = fig.add_subplot(2, 2, 4)
    if len(result.particle_results) > 0:
        alpha_values = result.particle_results["alpha"].values
        ax_alpha.hist(alpha_values, bins=20, edgecolor="black", alpha=0.7)
        ax_alpha.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="α=1")
        ax_alpha.legend()

        # Add statistics
        stats_text = f"Mean α: {result.mean_alpha:.2f}\nMedian α: {summary['alpha_median']:.2f}"
        ax_alpha.annotate(
            stats_text,
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
    ax_alpha.set_xlabel("α (diffusion exponent)")
    ax_alpha.set_ylabel("Count")
    ax_alpha.set_title("Diffusion Exponent Distribution")

    # Add overall title
    n_particles = result.n_particles
    fig.suptitle(
        f"Diffusion Analysis Report (n={n_particles} particles)",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save if path provided
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
