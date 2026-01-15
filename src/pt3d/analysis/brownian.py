"""Brownian motion analysis for particle tracking results.

This module provides functions to analyze diffusion properties from
tracked particle trajectories, including MSD computation and
diffusion coefficient extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pt3d.analysis.config import DiffusionAnalysisConfig
    from pt3d.config import VoxelSize
    from pt3d.pipeline import PipelineResult

from pt3d.synth.brownian import compute_msd, fit_diffusion_exponent

logger = logging.getLogger(__name__)


@dataclass
class ParticleDiffusionResult:
    """Diffusion analysis result for a single particle.

    Attributes
    ----------
    particle : int
        Particle ID from tracking.
    D : float
        Diffusion coefficient in um^2/s.
    alpha : float
        Diffusion exponent (1.0 for normal diffusion).
    n_frames : int
        Number of frames in the trajectory.
    msd : NDArray[np.float64]
        MSD values at each time lag.
    time_lags : NDArray[np.float64]
        Time lag values in seconds.
    """

    particle: int
    D: float
    alpha: float
    n_frames: int
    msd: NDArray[np.float64]
    time_lags: NDArray[np.float64]


@dataclass
class DiffusionAnalysisResult:
    """Complete diffusion analysis result.

    Attributes
    ----------
    particle_results : pd.DataFrame
        Summary DataFrame with columns [particle, D, alpha, n_frames].
    msd_per_particle : dict[int, ParticleDiffusionResult]
        Detailed results for each particle.
    ensemble_msd : NDArray[np.float64] | None
        Ensemble-averaged MSD values.
    ensemble_time_lags : NDArray[np.float64] | None
        Time lags for ensemble MSD.
    config : DiffusionAnalysisConfig
        Configuration used for analysis.
    positions_um : dict[int, NDArray[np.float64]]
        Particle positions in micrometers for visualization.
    """

    particle_results: pd.DataFrame
    msd_per_particle: dict[int, ParticleDiffusionResult]
    ensemble_msd: NDArray[np.float64] | None
    ensemble_time_lags: NDArray[np.float64] | None
    config: DiffusionAnalysisConfig
    positions_um: dict[int, NDArray[np.float64]] = field(default_factory=dict)

    @property
    def n_particles(self) -> int:
        """Number of particles analyzed."""
        return len(self.particle_results)

    @property
    def mean_d(self) -> float:
        """Mean diffusion coefficient."""
        if len(self.particle_results) == 0:
            return 0.0
        return float(self.particle_results["D"].mean())

    @property
    def std_d(self) -> float:
        """Standard deviation of diffusion coefficient."""
        if len(self.particle_results) == 0:
            return 0.0
        return float(self.particle_results["D"].std())

    @property
    def mean_alpha(self) -> float:
        """Mean diffusion exponent."""
        if len(self.particle_results) == 0:
            return 0.0
        return float(self.particle_results["alpha"].mean())

    @property
    def std_alpha(self) -> float:
        """Standard deviation of diffusion exponent."""
        if len(self.particle_results) == 0:
            return 0.0
        return float(self.particle_results["alpha"].std())

    def summary(self) -> dict:
        """Return summary statistics."""
        if len(self.particle_results) == 0:
            return {
                "n_particles": 0,
                "D_mean": 0.0,
                "D_std": 0.0,
                "D_median": 0.0,
                "alpha_mean": 0.0,
                "alpha_std": 0.0,
                "alpha_median": 0.0,
            }
        return {
            "n_particles": self.n_particles,
            "D_mean": self.mean_d,
            "D_std": self.std_d,
            "D_median": float(self.particle_results["D"].median()),
            "alpha_mean": self.mean_alpha,
            "alpha_std": self.std_alpha,
            "alpha_median": float(self.particle_results["alpha"].median()),
        }


def tracks_to_positions(
    tracks: pd.DataFrame,
    voxel_size: VoxelSize,
    min_length: int = 10,
) -> dict[int, NDArray[np.float64]]:
    """Convert tracks DataFrame to position arrays in physical units.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks with columns [particle, frame, z, y, x].
    voxel_size : VoxelSize
        Physical voxel dimensions for unit conversion.
    min_length : int
        Minimum track length to include.

    Returns
    -------
    dict[int, NDArray[np.float64]]
        Mapping from particle ID to positions array of shape (n_frames, 3)
        in micrometers. Positions are ordered by frame, with NaN for gaps.
    """
    if len(tracks) == 0:
        return {}

    required_cols = {"particle", "frame", "z", "y", "x"}
    if not required_cols.issubset(tracks.columns):
        missing = required_cols - set(tracks.columns)
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    # Get voxel size as array for vectorized conversion
    voxel_array = np.array(
        [voxel_size.z_um, voxel_size.y_um, voxel_size.x_um],
        dtype=np.float64,
    )

    result = {}
    n_filtered = 0

    for particle_id, group in tracks.groupby("particle"):
        # Filter by minimum length
        if len(group) < min_length:
            n_filtered += 1
            continue

        # Sort by frame
        group = group.sort_values("frame")

        # Determine frame range
        frame_min = int(group["frame"].min())
        frame_max = int(group["frame"].max())
        n_frames = frame_max - frame_min + 1

        # Create position array with NaN for gaps
        positions = np.full((n_frames, 3), np.nan, dtype=np.float64)

        # Fill in known positions (vectorized)
        frames = group["frame"].to_numpy(dtype=np.int64)
        idx = frames - frame_min
        coords = group[["z", "y", "x"]].to_numpy(dtype=np.float64) * voxel_array
        positions[idx] = coords

        result[int(particle_id)] = positions

    logger.info(f"Converted {len(result)} tracks (filtered {n_filtered} tracks < {min_length} frames)")
    return result


def interpolate_gaps(
    positions: NDArray[np.float64],
    method: str = "linear",
) -> NDArray[np.float64]:
    """Interpolate missing frames in a trajectory.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Positions array with NaN for missing frames, shape (n_frames, 3).
    method : str
        Interpolation method: "linear" or "none" (keep gaps).

    Returns
    -------
    NDArray[np.float64]
        Interpolated positions.
    """
    if method == "none":
        return positions.copy()

    positions = positions.copy()
    n_frames = len(positions)

    for dim in range(3):
        col = positions[:, dim]
        valid_mask = ~np.isnan(col)

        if not np.any(valid_mask) or np.all(valid_mask):
            continue

        valid_indices = np.where(valid_mask)[0]
        valid_values = col[valid_mask]

        # Interpolate
        all_indices = np.arange(n_frames)
        positions[:, dim] = np.interp(all_indices, valid_indices, valid_values)

    return positions


def compute_msd_with_gaps(
    positions: NDArray[np.float64],
    max_lag: int | None = None,
) -> NDArray[np.float64]:
    """Compute MSD handling frame gaps (NaN values).

    Parameters
    ----------
    positions : NDArray[np.float64]
        Positions array with possible NaN for missing frames.
    max_lag : int | None
        Maximum time lag. If None, uses n_frames // 2.

    Returns
    -------
    NDArray[np.float64]
        MSD values for each time lag.
    """
    n_frames = len(positions)

    if max_lag is None:
        max_lag = n_frames // 2

    max_lag = min(max_lag, n_frames - 1)

    msd = np.zeros(max_lag + 1)
    msd[0] = 0.0

    for lag in range(1, max_lag + 1):
        displacements = positions[lag:] - positions[:-lag]
        squared_disp = np.sum(displacements**2, axis=1)

        # Count valid (non-NaN) displacements
        valid_mask = ~np.isnan(squared_disp)

        if np.any(valid_mask):
            msd[lag] = np.nanmean(squared_disp[valid_mask])
        else:
            msd[lag] = np.nan

    return msd


def analyze_single_track(
    positions: NDArray[np.float64],
    dt: float,
    max_lag_fraction: float = 0.5,
    fit_range_fraction: tuple[float, float] = (0.1, 0.5),
) -> tuple[float, float, NDArray[np.float64]]:
    """Analyze diffusion for a single particle trajectory.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Positions of shape (n_frames, 3) in micrometers.
    dt : float
        Time step in seconds.
    max_lag_fraction : float
        Maximum time lag as fraction of track length.
    fit_range_fraction : tuple[float, float]
        Range for fitting as (start, end) fraction.

    Returns
    -------
    tuple[float, float, NDArray[np.float64]]
        (diffusion_coefficient, diffusion_exponent, msd_array)
    """
    n_frames = len(positions)

    # Check for NaN values
    has_gaps = np.any(np.isnan(positions))

    if has_gaps:
        # Use gap-aware MSD computation
        max_lag = int(n_frames * max_lag_fraction)
        msd = compute_msd_with_gaps(positions, max_lag)
    else:
        # Use standard MSD computation
        max_lag = int(n_frames * max_lag_fraction)
        # Reshape for compute_msd which expects (n_particles, n_steps, 3)
        msd = compute_msd(positions.reshape(1, -1, 3), max_lag)

    # Determine fit range
    fit_start = max(1, int(len(msd) * fit_range_fraction[0]))
    fit_end = max(fit_start + 2, int(len(msd) * fit_range_fraction[1]))

    # Fit diffusion exponent
    alpha, d_coeff = fit_diffusion_exponent(msd, dt=dt, fit_range=(fit_start, fit_end))

    return d_coeff, alpha, msd


def compute_ensemble_msd(
    positions_dict: dict[int, NDArray[np.float64]],
    dt: float,
    max_lag: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute ensemble-averaged MSD from multiple trajectories.

    Parameters
    ----------
    positions_dict : dict[int, NDArray[np.float64]]
        Mapping from particle ID to position arrays.
    dt : float
        Time step in seconds.
    max_lag : int | None
        Maximum time lag. If None, uses minimum track length // 2.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        (msd_values, time_lags_in_seconds)
    """
    if len(positions_dict) == 0:
        return np.array([0.0]), np.array([0.0])

    # Determine max_lag from minimum track length
    if max_lag is None:
        min_length = min(len(pos) for pos in positions_dict.values())
        max_lag = min_length // 2

    # Collect MSD from all particles
    all_msds = []
    for positions in positions_dict.values():
        has_gaps = np.any(np.isnan(positions))
        if has_gaps:
            msd = compute_msd_with_gaps(positions, max_lag)
        else:
            msd = compute_msd(positions.reshape(1, -1, 3), max_lag)

        # Pad with NaN if shorter
        if len(msd) < max_lag + 1:
            padded = np.full(max_lag + 1, np.nan)
            padded[: len(msd)] = msd
            msd = padded

        all_msds.append(msd[: max_lag + 1])

    # Compute ensemble average (ignoring NaN)
    all_msds_array = np.array(all_msds)
    ensemble_msd = np.nanmean(all_msds_array, axis=0)

    # Create time lags array
    time_lags = np.arange(max_lag + 1) * dt

    return ensemble_msd, time_lags


def analyze_diffusion(
    result: PipelineResult,
    config: DiffusionAnalysisConfig | None = None,
) -> DiffusionAnalysisResult:
    """Analyze diffusion properties from tracking results.

    This is the main entry point for Brownian motion analysis.

    Parameters
    ----------
    result : PipelineResult
        Pipeline result containing tracks DataFrame.
    config : DiffusionAnalysisConfig | None
        Analysis configuration. Uses defaults if None.

    Returns
    -------
    DiffusionAnalysisResult
        Complete analysis results including per-particle D and alpha.

    Examples
    --------
    >>> from pt3d import run_pipeline, PipelineConfig
    >>> from pt3d.analysis import analyze_diffusion, DiffusionAnalysisConfig
    >>>
    >>> # Run tracking pipeline
    >>> result = run_pipeline(data, config)
    >>>
    >>> # Analyze diffusion
    >>> analysis_config = DiffusionAnalysisConfig(dt=0.1, min_track_length=10)
    >>> analysis = analyze_diffusion(result, analysis_config)
    >>>
    >>> print(analysis.summary())
    >>> print(analysis.particle_results)
    """
    # Import here to avoid circular imports
    from pt3d.analysis.config import DiffusionAnalysisConfig as DAConfig

    if config is None:
        config = DAConfig()

    logger.info(f"Starting diffusion analysis with min_track_length={config.min_track_length}")

    # Convert tracks to positions
    voxel_size = result.config.input.voxel_size
    positions_dict = tracks_to_positions(
        result.tracks,
        voxel_size,
        min_length=config.min_track_length,
    )

    if len(positions_dict) == 0:
        logger.warning("No tracks meet minimum length requirement")
        return DiffusionAnalysisResult(
            particle_results=pd.DataFrame(columns=["particle", "D", "alpha", "n_frames"]),
            msd_per_particle={},
            ensemble_msd=None,
            ensemble_time_lags=None,
            config=config,
            positions_um={},
        )

    # Interpolate gaps if configured
    if config.interpolate_gaps:
        positions_dict = {pid: interpolate_gaps(pos, method="linear") for pid, pos in positions_dict.items()}

    # Analyze each particle
    particle_data = []
    msd_per_particle = {}

    for particle_id, positions in positions_dict.items():
        n_frames = len(positions)

        d_coeff, alpha, msd = analyze_single_track(
            positions,
            dt=config.dt,
            max_lag_fraction=config.msd.max_lag_fraction,
            fit_range_fraction=config.msd.fit_range_fraction,
        )

        # Create time lags array
        time_lags = np.arange(len(msd)) * config.dt

        particle_data.append(
            {
                "particle": particle_id,
                "D": d_coeff,
                "alpha": alpha,
                "n_frames": n_frames,
            }
        )

        msd_per_particle[particle_id] = ParticleDiffusionResult(
            particle=particle_id,
            D=d_coeff,
            alpha=alpha,
            n_frames=n_frames,
            msd=msd,
            time_lags=time_lags,
        )

    # Create summary DataFrame
    particle_results = pd.DataFrame(particle_data)

    # Compute ensemble MSD
    ensemble_msd, ensemble_time_lags = compute_ensemble_msd(positions_dict, dt=config.dt)

    logger.info(
        f"Analyzed {len(particle_results)} particles: "
        f"D={particle_results['D'].mean():.4f} +/- {particle_results['D'].std():.4f} um^2/s, "
        f"alpha={particle_results['alpha'].mean():.2f} +/- {particle_results['alpha'].std():.2f}"
    )

    return DiffusionAnalysisResult(
        particle_results=particle_results,
        msd_per_particle=msd_per_particle,
        ensemble_msd=ensemble_msd,
        ensemble_time_lags=ensemble_time_lags,
        config=config,
        positions_um=positions_dict,
    )
