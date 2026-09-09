"""Visualization functions for diffusion analysis results.

This module provides matplotlib-based visualization for diffusion
analysis results including 3D trajectory plots, histograms, scatter
plots, and MSD log-log plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from pt3d.analysis.brownian import DiffusionAnalysisResult
    from pt3d.analysis.config import (
        DiffusionScatterConfig,
        HistogramConfig,
        MSDPlotConfig,
        TrackVisualizationConfig,
    )

D_ALPHA_LABEL = r"$D_\alpha$ ($\mu$m$^2$/s$^\alpha$)"
D_ALPHA_MILLI_LABEL = r"$D_\alpha$ ($10^{-3}\,\mu$m$^2$/s$^\alpha$)"


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
        ax.set_xlabel(r"X ($\mu$m)")
        ax.set_ylabel(r"Y ($\mu$m)")
        ax.set_zlabel(r"Z ($\mu$m)")
        ax.set_title("No tracks to display")
        return fig, ax

    # Get color values based on config
    particle_ids = list(result.positions_um.keys())

    if config.color_by == "D":
        color_values = [result.msd_per_particle[pid].D for pid in particle_ids]
        color_label = D_ALPHA_LABEL
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
    if config.show_labels:
        label_fontsize = config.label_fontsize
        if label_fontsize is not None:
            ax.set_xlabel(r"X ($\mu$m)", fontsize=label_fontsize)
            ax.set_ylabel(r"Y ($\mu$m)", fontsize=label_fontsize)
            ax.set_zlabel(r"Z ($\mu$m)", fontsize=label_fontsize)
        else:
            ax.set_xlabel(r"X ($\mu$m)")
            ax.set_ylabel(r"Y ($\mu$m)")
            ax.set_zlabel(r"Z ($\mu$m)")

    # Set title
    if config.show_title:
        title = f"3D Trajectories (colored by {color_label})"
        if config.title_fontsize is not None:
            ax.set_title(title, fontsize=config.title_fontsize)
        else:
            ax.set_title(title)

    # Set tick font size
    if config.tick_fontsize is not None:
        ax.tick_params(labelsize=config.tick_fontsize)

    # Set equal aspect ratio based on data range
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]
    max_range = max(x_range, y_range, z_range)
    if max_range > 0:
        ax.set_box_aspect([x_range / max_range, y_range / max_range, z_range / max_range])

    # Limit tick count to avoid text overlap on small-range axes
    ax.xaxis.set_major_locator(MaxNLocator(nbins=config.tick_nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=config.tick_nbins))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=config.tick_nbins))

    # Set view angle
    ax.view_init(elev=config.elevation, azim=config.azimuth)

    # Add colorbar
    if config.show_colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        if config.colorbar_fontsize is not None:
            cbar.set_label(color_label, fontsize=config.colorbar_fontsize)
            cbar.ax.tick_params(labelsize=config.colorbar_fontsize)
        else:
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

    # Set labels and title for D histogram
    label_fontsize = config.label_fontsize
    title_fontsize = config.title_fontsize
    stats_fontsize = config.stats_fontsize if config.stats_fontsize is not None else 9

    if config.show_labels:
        if label_fontsize is not None:
            ax_d.set_xlabel(D_ALPHA_LABEL, fontsize=label_fontsize)
            ax_d.set_ylabel("Count", fontsize=label_fontsize)
        else:
            ax_d.set_xlabel(D_ALPHA_LABEL)
            ax_d.set_ylabel("Count")

    if config.show_title:
        if title_fontsize is not None:
            ax_d.set_title("Generalized Diffusion Coefficient Distribution", fontsize=title_fontsize)
        else:
            ax_d.set_title("Generalized Diffusion Coefficient Distribution")

    if config.tick_fontsize is not None:
        ax_d.tick_params(labelsize=config.tick_fontsize)

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
            fontsize=stats_fontsize,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    # Plot alpha histogram
    ax_alpha.hist(alpha_values, bins=config.n_bins, edgecolor="black", alpha=0.7)

    # Set labels and title for alpha histogram
    if config.show_labels:
        if label_fontsize is not None:
            ax_alpha.set_xlabel("α (diffusion exponent)", fontsize=label_fontsize)
            ax_alpha.set_ylabel("Count", fontsize=label_fontsize)
        else:
            ax_alpha.set_xlabel("α (diffusion exponent)")
            ax_alpha.set_ylabel("Count")

    if config.show_title:
        if title_fontsize is not None:
            ax_alpha.set_title("Diffusion Exponent Distribution", fontsize=title_fontsize)
        else:
            ax_alpha.set_title("Diffusion Exponent Distribution")

    if config.tick_fontsize is not None:
        ax_alpha.tick_params(labelsize=config.tick_fontsize)

    # Add reference line at alpha=1 (normal diffusion)
    if config.show_alpha_reference_line:
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
            fontsize=stats_fontsize,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    fig.tight_layout()
    return fig, (ax_d, ax_alpha)


def plot_diffusion_scatter(
    result: DiffusionAnalysisResult,
    config: DiffusionScatterConfig | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot D-alpha scatter with marginal histograms.

    Creates a joint plot layout:
    - Main scatter axis: D (x) vs alpha (y)
    - Right marginal histogram: alpha distribution

    Parameters
    ----------
    result : DiffusionAnalysisResult
        Analysis result containing particle statistics.
    config : DiffusionScatterConfig | None
        Scatter configuration.

    Returns
    -------
    tuple[Figure, tuple[Axes, Axes]]
        Figure and (ax_scatter, ax_hist_alpha) axes.
    """
    from pt3d.analysis.config import DiffusionScatterConfig

    if config is None:
        config = DiffusionScatterConfig()

    fig = plt.figure(figsize=config.figsize, constrained_layout=True)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(4.0, 1.4),
        wspace=0.05,
    )
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_hist_alpha = fig.add_subplot(grid[0, 1], sharey=ax_scatter)

    label_fontsize = config.label_fontsize
    title_fontsize = config.title_fontsize
    stats_fontsize = config.stats_fontsize if config.stats_fontsize is not None else 9

    if len(result.particle_results) == 0:
        if config.show_labels:
            if label_fontsize is not None:
                ax_scatter.set_xlabel(D_ALPHA_MILLI_LABEL, fontsize=label_fontsize)
                ax_scatter.set_ylabel("α (diffusion exponent)", fontsize=label_fontsize)
            else:
                ax_scatter.set_xlabel(D_ALPHA_MILLI_LABEL)
                ax_scatter.set_ylabel("α (diffusion exponent)")

        ax_scatter.set_title("D vs α Scatter (no data)")
        ax_hist_alpha.set_title("α Histogram (no data)")
        return fig, (ax_scatter, ax_hist_alpha)

    d_values = result.particle_results["D"].to_numpy()
    alpha_values = result.particle_results["alpha"].to_numpy()

    valid_mask = np.isfinite(d_values) & np.isfinite(alpha_values)
    d_values = d_values[valid_mask]
    alpha_values = alpha_values[valid_mask]

    if len(d_values) == 0:
        if config.show_labels:
            if label_fontsize is not None:
                ax_scatter.set_xlabel(D_ALPHA_MILLI_LABEL, fontsize=label_fontsize)
                ax_scatter.set_ylabel("α (diffusion exponent)", fontsize=label_fontsize)
            else:
                ax_scatter.set_xlabel(D_ALPHA_MILLI_LABEL)
                ax_scatter.set_ylabel("α (diffusion exponent)")

        ax_scatter.set_title("D vs α Scatter (no valid data)")
        return fig, (ax_scatter, ax_hist_alpha)

    d_scale = 1e3
    d_values_scaled = d_values * d_scale

    ax_scatter.scatter(
        d_values_scaled,
        alpha_values,
        s=config.scatter_marker_size,
        facecolors=config.scatter_facecolor,
        edgecolors=config.scatter_edgecolor,
        linewidths=config.scatter_linewidth,
        alpha=config.scatter_alpha,
    )
    ax_hist_alpha.hist(
        alpha_values,
        bins=config.n_bins,
        orientation="horizontal",
        color=config.hist_color,
        edgecolor=config.hist_edgecolor,
        alpha=config.hist_alpha,
    )

    # Keep marginal panel compact by hiding redundant y tick labels.
    ax_hist_alpha.tick_params(axis="y", labelleft=False)

    if config.show_alpha_reference_line:
        ax_scatter.axhline(y=1.0, color="red", linestyle="--", linewidth=1.2)
        ax_hist_alpha.axhline(y=1.0, color="red", linestyle="--", linewidth=1.0)

    # Add padding when all values are identical so points/bars are visible.
    d_min = float(d_values_scaled.min())
    d_max = float(d_values_scaled.max())
    alpha_min = float(alpha_values.min())
    alpha_max = float(alpha_values.max())

    d_margin = 0.05 * (d_max - d_min) if d_max > d_min else 0.1
    alpha_margin = 0.05 * (alpha_max - alpha_min) if alpha_max > alpha_min else 0.1
    x_upper = d_max + d_margin
    y_upper = alpha_max + alpha_margin
    ax_scatter.set_xlim(0.0, x_upper if x_upper > 0 else 1.0)
    y_offset = min(0.03, max(0.01, 0.05 * y_upper)) if y_upper > 0 else 0.02
    ax_scatter.set_ylim(-y_offset, y_upper if y_upper > 0 else 1.0)

    if config.show_labels:
        if label_fontsize is not None:
            ax_scatter.set_xlabel(D_ALPHA_MILLI_LABEL, fontsize=label_fontsize)
            ax_scatter.set_ylabel("α (diffusion exponent)", fontsize=label_fontsize)
            ax_hist_alpha.set_xlabel("Count", fontsize=label_fontsize)
        else:
            ax_scatter.set_xlabel(D_ALPHA_MILLI_LABEL)
            ax_scatter.set_ylabel("α (diffusion exponent)")
            ax_hist_alpha.set_xlabel("Count")

    if config.show_title:
        title = "Diffusion Exponent vs Generalized Diffusion Coefficient"
        if title_fontsize is not None:
            ax_scatter.set_title(title, fontsize=title_fontsize)
        else:
            ax_scatter.set_title(title)

    if config.tick_fontsize is not None:
        for axis in (ax_scatter, ax_hist_alpha):
            axis.tick_params(labelsize=config.tick_fontsize)

    if config.show_stats:
        mean_d = np.mean(d_values)
        median_d = np.median(d_values)
        mean_d_scaled = mean_d * d_scale
        median_d_scaled = median_d * d_scale
        mean_alpha = np.mean(alpha_values)
        median_alpha = np.median(alpha_values)
        stats_text = (
            f"n = {len(d_values)}\n"
            f"Mean Dα (10^-3): {mean_d_scaled:.2f}\nMedian Dα (10^-3): {median_d_scaled:.2f}\n"
            f"Mean α: {mean_alpha:.2f}\nMedian α: {median_alpha:.2f}"
        )
        ax_scatter.annotate(
            stats_text,
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=stats_fontsize,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    return fig, (ax_scatter, ax_hist_alpha)


def plot_msd_loglog(
    result: DiffusionAnalysisResult,
    config: MSDPlotConfig | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot MSD vs time delay on log-log scale.

    Shows individual particle MSDs, ensemble average, and an optional
    slope-one reference line 6 * mean_d * t. Despite the legacy "alpha=1 fit"
    legend, this function does not fit a curve or use the estimated alpha
    for that line. For anomalous diffusion it is only a visual reference,
    not the model MSD = 6 * D * t**alpha.

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
    - Slope < 1: Subdiffusion
    - Slope > 1: Superdiffusion
    The slope alone does not identify the physical cause.
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
        ax.set_ylabel(r"MSD ($\mu$m$^2$)")
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
    label_fontsize = config.label_fontsize
    if config.show_labels:
        if label_fontsize is not None:
            ax.set_xlabel("Time lag (s)", fontsize=label_fontsize)
            ax.set_ylabel(r"MSD ($\mu$m$^2$)", fontsize=label_fontsize)
        else:
            ax.set_xlabel("Time lag (s)")
            ax.set_ylabel(r"MSD ($\mu$m$^2$)")

    if config.show_title:
        mean_alpha = result.mean_alpha
        title = f"MSD vs Time (mean α = {mean_alpha:.2f})"
        if config.title_fontsize is not None:
            ax.set_title(title, fontsize=config.title_fontsize)
        else:
            ax.set_title(title)

    if config.tick_fontsize is not None:
        ax.tick_params(labelsize=config.tick_fontsize)

    if config.show_legend:
        if config.legend_fontsize is not None:
            ax.legend(loc="upper left", fontsize=config.legend_fontsize)
        else:
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
    ax_d.set_xlabel(D_ALPHA_LABEL)
    ax_d.set_ylabel("Count")
    ax_d.set_title("Generalized Diffusion Coefficient Distribution")

    # Add statistics
    if len(result.particle_results) > 0:
        summary = result.summary()
        stats_text = (
            f"Mean Dα: {summary['D_mean']:.4f} " + r"$\mu$m$^2$/s$^\alpha$"
            + f"\nMedian Dα: {summary['D_median']:.4f} " + r"$\mu$m$^2$/s$^\alpha$"
        )
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
