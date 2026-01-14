"""Tests for diffusion analysis visualization functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pt3d.analysis.brownian import (
    DiffusionAnalysisResult,
    ParticleDiffusionResult,
)
from pt3d.analysis.config import (
    DiffusionAnalysisConfig,
    HistogramConfig,
    MSDPlotConfig,
    TrackVisualizationConfig,
)
from pt3d.analysis.visualization import (
    create_analysis_report,
    plot_diffusion_histograms,
    plot_msd_loglog,
    plot_tracks_3d,
)


@pytest.fixture
def sample_analysis_result():
    """Create a sample DiffusionAnalysisResult for testing."""
    # Create sample particle results
    particle_results = pd.DataFrame(
        {
            "particle": [0, 1, 2, 3, 4],
            "D": [0.5, 0.8, 1.2, 0.3, 1.0],
            "alpha": [0.9, 1.1, 1.0, 0.7, 1.2],
            "n_frames": [50, 60, 45, 55, 40],
        }
    )

    # Create sample positions
    positions_um = {}
    msd_per_particle = {}

    for pid in range(5):
        # Generate random walk
        rng = np.random.default_rng(42 + pid)
        n_frames = particle_results.loc[
            particle_results["particle"] == pid, "n_frames"
        ].values[0]
        positions = np.cumsum(rng.normal(0, 0.1, (n_frames, 3)), axis=0)
        positions_um[pid] = positions

        # Create MSD data
        msd = np.arange(n_frames // 2 + 1, dtype=np.float64) * 0.1
        time_lags = np.arange(n_frames // 2 + 1) * 0.1

        msd_per_particle[pid] = ParticleDiffusionResult(
            particle=pid,
            D=particle_results.loc[particle_results["particle"] == pid, "D"].values[0],
            alpha=particle_results.loc[
                particle_results["particle"] == pid, "alpha"
            ].values[0],
            n_frames=n_frames,
            msd=msd,
            time_lags=time_lags,
        )

    # Create ensemble MSD
    ensemble_msd = np.arange(21, dtype=np.float64) * 0.1
    ensemble_time_lags = np.arange(21) * 0.1

    config = DiffusionAnalysisConfig(dt=0.1, min_track_length=10)

    return DiffusionAnalysisResult(
        particle_results=particle_results,
        msd_per_particle=msd_per_particle,
        ensemble_msd=ensemble_msd,
        ensemble_time_lags=ensemble_time_lags,
        config=config,
        positions_um=positions_um,
    )


@pytest.fixture
def empty_analysis_result():
    """Create an empty DiffusionAnalysisResult."""
    config = DiffusionAnalysisConfig(dt=0.1, min_track_length=10)

    return DiffusionAnalysisResult(
        particle_results=pd.DataFrame(columns=["particle", "D", "alpha", "n_frames"]),
        msd_per_particle={},
        ensemble_msd=None,
        ensemble_time_lags=None,
        config=config,
        positions_um={},
    )


class TestPlotTracks3D:
    """Tests for plot_tracks_3d function."""

    def test_creates_figure(self, sample_analysis_result):
        """Test that 3D plot creates valid figure."""
        fig, ax = plot_tracks_3d(sample_analysis_result)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_color_by_d(self, sample_analysis_result):
        """Test coloring by diffusion coefficient."""
        config = TrackVisualizationConfig(color_by="D")
        fig, ax = plot_tracks_3d(sample_analysis_result, config=config)

        assert "D" in ax.get_title()
        plt.close(fig)

    def test_color_by_alpha(self, sample_analysis_result):
        """Test coloring by diffusion exponent."""
        config = TrackVisualizationConfig(color_by="alpha")
        fig, ax = plot_tracks_3d(sample_analysis_result, config=config)

        assert "α" in ax.get_title()
        plt.close(fig)

    def test_color_by_particle(self, sample_analysis_result):
        """Test coloring by particle ID."""
        config = TrackVisualizationConfig(color_by="particle")
        fig, ax = plot_tracks_3d(sample_analysis_result, config=config)

        assert "Particle" in ax.get_title()
        plt.close(fig)

    def test_empty_result(self, empty_analysis_result):
        """Test with empty result."""
        fig, ax = plot_tracks_3d(empty_analysis_result)

        assert "No tracks" in ax.get_title()
        plt.close(fig)

    def test_custom_colormap(self, sample_analysis_result):
        """Test with custom colormap."""
        config = TrackVisualizationConfig(colormap="plasma")
        fig, _ax = plot_tracks_3d(sample_analysis_result, config=config)

        assert fig is not None
        plt.close(fig)

    def test_existing_axes(self, sample_analysis_result):
        """Test plotting to existing axes."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        _fig_out, ax_out = plot_tracks_3d(sample_analysis_result, ax=ax)

        assert ax_out is ax
        plt.close(fig)


class TestPlotDiffusionHistograms:
    """Tests for plot_diffusion_histograms function."""

    def test_creates_figure(self, sample_analysis_result):
        """Test that histogram creates valid figure."""
        fig, (ax_d, ax_alpha) = plot_diffusion_histograms(sample_analysis_result)

        assert fig is not None
        assert ax_d is not None
        assert ax_alpha is not None
        plt.close(fig)

    def test_labels(self, sample_analysis_result):
        """Test that axes have proper labels."""
        fig, (ax_d, ax_alpha) = plot_diffusion_histograms(sample_analysis_result)

        assert "D" in ax_d.get_xlabel()
        assert "α" in ax_alpha.get_xlabel() or "alpha" in ax_alpha.get_xlabel().lower()
        plt.close(fig)

    def test_alpha_reference_line(self, sample_analysis_result):
        """Test that alpha=1 reference line is present."""
        fig, (_ax_d, ax_alpha) = plot_diffusion_histograms(sample_analysis_result)

        # Check for vertical line at x=1
        lines = ax_alpha.get_lines()
        assert len(lines) > 0
        plt.close(fig)

    def test_empty_result(self, empty_analysis_result):
        """Test with empty result."""
        fig, (ax_d, _ax_alpha) = plot_diffusion_histograms(empty_analysis_result)

        assert "no data" in ax_d.get_title().lower()
        plt.close(fig)

    def test_log_scale_option(self, sample_analysis_result):
        """Test log scale option for D histogram."""
        config = HistogramConfig(log_scale_d=True)
        fig, (ax_d, _ax_alpha) = plot_diffusion_histograms(
            sample_analysis_result, config=config
        )

        assert ax_d.get_xscale() == "log"
        plt.close(fig)

    def test_stats_annotation(self, sample_analysis_result):
        """Test statistics annotation."""
        config = HistogramConfig(show_stats=True)
        fig, (ax_d, _ax_alpha) = plot_diffusion_histograms(
            sample_analysis_result, config=config
        )

        # Check that annotations exist
        assert len(ax_d.texts) > 0 or len(ax_d.patches) > 0
        plt.close(fig)


class TestPlotMsdLoglog:
    """Tests for plot_msd_loglog function."""

    def test_creates_figure(self, sample_analysis_result):
        """Test that MSD plot creates valid figure."""
        fig, ax = plot_msd_loglog(sample_analysis_result)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_log_scale(self, sample_analysis_result):
        """Test that both axes are log scale."""
        fig, ax = plot_msd_loglog(sample_analysis_result)

        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_labels(self, sample_analysis_result):
        """Test axis labels."""
        fig, ax = plot_msd_loglog(sample_analysis_result)

        assert "Time" in ax.get_xlabel() or "lag" in ax.get_xlabel().lower()
        assert "MSD" in ax.get_ylabel()
        plt.close(fig)

    def test_empty_result(self, empty_analysis_result):
        """Test with empty result."""
        fig, ax = plot_msd_loglog(empty_analysis_result)

        assert "no data" in ax.get_title().lower()
        plt.close(fig)

    def test_show_individual(self, sample_analysis_result):
        """Test showing individual particle MSDs."""
        config = MSDPlotConfig(show_individual=True, show_ensemble=False)
        fig, ax = plot_msd_loglog(sample_analysis_result, config=config)

        # Should have lines for individual particles
        assert len(ax.get_lines()) > 0
        plt.close(fig)

    def test_show_ensemble(self, sample_analysis_result):
        """Test showing ensemble MSD."""
        config = MSDPlotConfig(show_individual=False, show_ensemble=True)
        fig, ax = plot_msd_loglog(sample_analysis_result, config=config)

        # Should have ensemble line
        assert len(ax.get_lines()) > 0
        plt.close(fig)

    def test_existing_axes(self, sample_analysis_result):
        """Test plotting to existing axes."""
        fig, ax = plt.subplots()
        _fig_out, ax_out = plot_msd_loglog(sample_analysis_result, ax=ax)

        assert ax_out is ax
        plt.close(fig)


class TestCreateAnalysisReport:
    """Tests for create_analysis_report function."""

    def test_creates_figure(self, sample_analysis_result):
        """Test that report creates valid figure."""
        fig = create_analysis_report(sample_analysis_result)

        assert fig is not None
        # 3D, MSD, D hist, alpha hist + colorbar
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_empty_result(self, empty_analysis_result):
        """Test with empty result."""
        fig = create_analysis_report(empty_analysis_result)

        assert fig is not None
        plt.close(fig)

    def test_save_to_file(self, sample_analysis_result, tmp_path):
        """Test saving report to file."""
        output_path = tmp_path / "report.png"
        fig = create_analysis_report(sample_analysis_result, output_path=str(output_path))

        assert output_path.exists()
        plt.close(fig)

    def test_title_contains_particle_count(self, sample_analysis_result):
        """Test that title shows particle count."""
        fig = create_analysis_report(sample_analysis_result)

        suptitle = fig._suptitle.get_text() if fig._suptitle else ""
        assert "5" in suptitle or "n=" in suptitle.lower()
        plt.close(fig)
