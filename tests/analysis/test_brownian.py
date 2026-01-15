"""Tests for diffusion analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pt3d.analysis.brownian import (
    DiffusionAnalysisResult,
    analyze_diffusion,
    analyze_single_track,
    compute_ensemble_msd,
    compute_msd_with_gaps,
    interpolate_gaps,
    tracks_to_positions,
)
from pt3d.analysis.config import DiffusionAnalysisConfig
from pt3d.config import VoxelSize


@pytest.fixture
def voxel_size():
    """Default voxel size for tests."""
    return VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1)


@pytest.fixture
def sample_tracks():
    """Sample tracks DataFrame for testing."""
    # Create 3 particles with 20 frames each
    data = []
    for p in range(3):
        for f in range(20):
            data.append(
                {
                    "particle": p,
                    "frame": f,
                    "z": 10.0 + 0.1 * f + 0.5 * p,
                    "y": 20.0 + 0.2 * f + 1.0 * p,
                    "x": 30.0 + 0.3 * f + 1.5 * p,
                }
            )
    return pd.DataFrame(data)


@pytest.fixture
def sample_tracks_with_gaps():
    """Sample tracks with frame gaps."""
    data = []
    # Particle 0: frames 0, 2, 4, 6, 8, 10, 12, 14, 16, 18 (gaps)
    for f in range(0, 20, 2):
        data.append(
            {
                "particle": 0,
                "frame": f,
                "z": 10.0 + 0.2 * f,
                "y": 20.0 + 0.4 * f,
                "x": 30.0 + 0.6 * f,
            }
        )
    # Particle 1: all 20 frames (no gaps)
    for f in range(20):
        data.append(
            {
                "particle": 1,
                "frame": f,
                "z": 15.0 + 0.1 * f,
                "y": 25.0 + 0.2 * f,
                "x": 35.0 + 0.3 * f,
            }
        )
    return pd.DataFrame(data)


class TestTracksToPositions:
    """Tests for tracks_to_positions function."""

    def test_basic_conversion(self, sample_tracks, voxel_size):
        """Test basic conversion with complete tracks."""
        positions = tracks_to_positions(sample_tracks, voxel_size, min_length=10)

        assert len(positions) == 3
        for _pid, pos in positions.items():
            assert pos.shape == (20, 3)
            assert not np.any(np.isnan(pos))

    def test_unit_conversion(self, sample_tracks, voxel_size):
        """Test that coordinates are converted to physical units."""
        positions = tracks_to_positions(sample_tracks, voxel_size, min_length=10)

        # First particle, first frame
        pos = positions[0][0]
        # z: 10.0 * 1.0 = 10.0 um
        # y: 20.0 * 0.1 = 2.0 um
        # x: 30.0 * 0.1 = 3.0 um
        assert np.isclose(pos[0], 10.0)
        assert np.isclose(pos[1], 2.0)
        assert np.isclose(pos[2], 3.0)

    def test_min_length_filter(self, voxel_size):
        """Test that short tracks are filtered."""
        tracks = pd.DataFrame(
            {
                "particle": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "frame": [0, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "z": [10.0] * 12,
                "y": [20.0] * 12,
                "x": [30.0] * 12,
            }
        )

        # With min_length=5, particle 0 should be filtered out
        positions = tracks_to_positions(tracks, voxel_size, min_length=5)
        assert 0 not in positions
        assert 1 in positions

    def test_gap_handling(self, sample_tracks_with_gaps, voxel_size):
        """Test handling of frame gaps."""
        positions = tracks_to_positions(sample_tracks_with_gaps, voxel_size, min_length=5)

        # Particle 0 has gaps (frames 0, 2, 4, ...)
        pos_0 = positions[0]
        # Frame range is 0-18, so 19 frames total
        assert pos_0.shape[0] == 19

        # Check that gaps are NaN
        assert np.isnan(pos_0[1, 0])  # frame 1 should be NaN
        assert np.isnan(pos_0[3, 0])  # frame 3 should be NaN

        # Particle 1 has no gaps
        pos_1 = positions[1]
        assert not np.any(np.isnan(pos_1))

    def test_empty_tracks(self, voxel_size):
        """Test with empty DataFrame."""
        tracks = pd.DataFrame(columns=["particle", "frame", "z", "y", "x"])
        positions = tracks_to_positions(tracks, voxel_size, min_length=5)
        assert len(positions) == 0

    def test_missing_columns(self, voxel_size):
        """Test error on missing columns."""
        tracks = pd.DataFrame({"particle": [0], "frame": [0], "z": [10.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            tracks_to_positions(tracks, voxel_size, min_length=1)


class TestInterpolateGaps:
    """Tests for interpolate_gaps function."""

    def test_no_gaps(self):
        """Test with complete data (no interpolation needed)."""
        positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        result = interpolate_gaps(positions, method="linear")
        np.testing.assert_array_equal(result, positions)

    def test_linear_interpolation(self):
        """Test linear interpolation of gaps."""
        positions = np.array([[0, 0, 0], [np.nan, np.nan, np.nan], [2, 2, 2]], dtype=np.float64)
        result = interpolate_gaps(positions, method="linear")

        expected = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_none_method(self):
        """Test that 'none' method keeps gaps."""
        positions = np.array([[0, 0, 0], [np.nan, np.nan, np.nan], [2, 2, 2]], dtype=np.float64)
        result = interpolate_gaps(positions, method="none")
        assert np.isnan(result[1, 0])


class TestComputeMsdWithGaps:
    """Tests for compute_msd_with_gaps function."""

    def test_no_gaps(self):
        """Test MSD computation without gaps."""
        # Linear trajectory
        positions = np.array([[i, i, i] for i in range(10)], dtype=np.float64)
        msd = compute_msd_with_gaps(positions, max_lag=5)

        assert len(msd) == 6
        assert msd[0] == 0.0
        # For linear motion: displacement at lag k is [k, k, k]
        # squared displacement = 3 * k^2
        for lag in range(1, 6):
            expected = 3 * lag**2
            assert np.isclose(msd[lag], expected)

    def test_with_gaps(self):
        """Test MSD computation with gaps (NaN values)."""
        positions = np.array(
            [
                [0, 0, 0],
                [np.nan, np.nan, np.nan],
                [2, 2, 2],
                [3, 3, 3],
                [np.nan, np.nan, np.nan],
                [5, 5, 5],
            ],
            dtype=np.float64,
        )
        msd = compute_msd_with_gaps(positions, max_lag=2)

        assert len(msd) == 3
        assert msd[0] == 0.0
        # Should compute valid displacements only
        assert not np.isnan(msd[1])


class TestAnalyzeSingleTrack:
    """Tests for analyze_single_track function."""

    def test_brownian_motion(self):
        """Test analysis of simulated Brownian motion."""
        # Generate Brownian motion
        rng = np.random.default_rng(42)
        n_steps = 100
        d_coeff = 1.0
        dt = 0.1

        # Standard Brownian motion: increments ~ N(0, 2*D*dt)
        increments = rng.normal(0, np.sqrt(2 * d_coeff * dt), (n_steps - 1, 3))
        positions = np.zeros((n_steps, 3))
        positions[1:] = np.cumsum(increments, axis=0)

        d_fit, alpha, _msd = analyze_single_track(positions, dt=dt)

        # Alpha should be close to 1.0 for normal diffusion
        assert 0.7 < alpha < 1.3
        # D should be in reasonable range (statistical variation expected)
        assert d_fit > 0

    def test_linear_motion(self):
        """Test analysis of linear (directed) motion."""
        # Linear trajectory: constant velocity
        n_steps = 50
        dt = 0.1
        v = 1.0  # velocity in um/s

        positions = np.zeros((n_steps, 3))
        for i in range(n_steps):
            t = i * dt
            positions[i] = [v * t, v * t, v * t]

        _d_fit, alpha, _msd = analyze_single_track(positions, dt=dt)

        # Alpha should be close to 2.0 for ballistic motion
        assert alpha > 1.5


class TestComputeEnsembleMsd:
    """Tests for compute_ensemble_msd function."""

    def test_single_particle(self):
        """Test ensemble MSD with single particle."""
        positions = {0: np.array([[i, i, i] for i in range(20)], dtype=np.float64)}
        msd, time_lags = compute_ensemble_msd(positions, dt=0.1)

        assert len(msd) == len(time_lags)
        assert msd[0] == 0.0
        assert time_lags[0] == 0.0

    def test_multiple_particles(self):
        """Test ensemble MSD with multiple particles."""
        rng = np.random.default_rng(42)
        positions = {}
        for i in range(5):
            increments = rng.normal(0, 1.0, (19, 3))
            traj = np.zeros((20, 3))
            traj[1:] = np.cumsum(increments, axis=0)
            positions[i] = traj

        msd, _time_lags = compute_ensemble_msd(positions, dt=0.1)

        assert len(msd) == 11  # max_lag = 20 // 2 = 10, so 11 values
        assert msd[0] == 0.0

    def test_empty_dict(self):
        """Test with empty positions dictionary."""
        msd, _time_lags = compute_ensemble_msd({}, dt=0.1)
        assert len(msd) == 1
        assert msd[0] == 0.0


class TestAnalyzeDiffusion:
    """Integration tests for analyze_diffusion function."""

    @pytest.fixture
    def mock_pipeline_result(self, sample_tracks, voxel_size):
        """Create a mock PipelineResult."""
        from dataclasses import dataclass, field
        from datetime import datetime, timezone

        from pt3d.config import (
            DetectionConfig,
            InputConfig,
            PipelineConfig,
            PostprocessConfig,
            TrackingConfig,
        )

        config = PipelineConfig(
            input=InputConfig(voxel_size=voxel_size),
            detection=DetectionConfig(diameter=(5, 9, 9)),
            tracking=TrackingConfig(search_range_um=2.0),
            postprocess=PostprocessConfig(min_track_length=2),
        )

        @dataclass
        class MockPipelineResult:
            detections: pd.DataFrame = field(default_factory=pd.DataFrame)
            tracks: pd.DataFrame = field(default_factory=pd.DataFrame)
            track_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
            config: PipelineConfig = None
            input_info: dict = field(default_factory=dict)
            start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
            end_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

        return MockPipelineResult(tracks=sample_tracks, config=config)

    def test_full_analysis(self, mock_pipeline_result):
        """Test complete analysis pipeline."""
        config = DiffusionAnalysisConfig(dt=0.1, min_track_length=10)
        result = analyze_diffusion(mock_pipeline_result, config)

        assert isinstance(result, DiffusionAnalysisResult)
        assert result.n_particles == 3
        assert "D" in result.particle_results.columns
        assert "alpha" in result.particle_results.columns
        assert "n_frames" in result.particle_results.columns

    def test_summary(self, mock_pipeline_result):
        """Test summary statistics."""
        config = DiffusionAnalysisConfig(dt=0.1, min_track_length=10)
        result = analyze_diffusion(mock_pipeline_result, config)
        summary = result.summary()

        assert "n_particles" in summary
        assert "D_mean" in summary
        assert "D_std" in summary
        assert "alpha_mean" in summary

    def test_no_tracks_meet_criteria(self, mock_pipeline_result):
        """Test when no tracks meet minimum length."""
        # Set very high min_track_length
        config = DiffusionAnalysisConfig(dt=0.1, min_track_length=100)
        result = analyze_diffusion(mock_pipeline_result, config)

        assert result.n_particles == 0
        assert len(result.particle_results) == 0
