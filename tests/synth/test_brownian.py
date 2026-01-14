"""Tests for Brownian motion simulation module."""

import numpy as np
import pytest

from pt3d.config import VoxelSize
from pt3d.synth.brownian import (
    apply_boundary_conditions,
    compute_msd,
    fbm_covariance_matrix,
    fit_diffusion_exponent,
    generate_brownian_particles,
    generate_fbm_trajectory,
    generate_fbm_trajectory_3d,
)


class TestFBMCovarianceMatrix:
    """Tests for FBM covariance matrix generation."""

    def test_covariance_matrix_shape(self) -> None:
        """Test covariance matrix has correct shape."""
        n = 100
        cov = fbm_covariance_matrix(n, hurst_exponent=0.5)
        assert cov.shape == (n, n)

    def test_covariance_matrix_symmetric(self) -> None:
        """Test covariance matrix is symmetric."""
        cov = fbm_covariance_matrix(50, hurst_exponent=0.5)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_covariance_matrix_positive_semidefinite(self) -> None:
        """Test covariance matrix is positive semi-definite."""
        for H in [0.3, 0.5, 0.7]:
            cov = fbm_covariance_matrix(50, hurst_exponent=H)
            eigenvalues = np.linalg.eigvalsh(cov)
            # Allow small numerical errors
            assert np.all(eigenvalues >= -1e-8)

    def test_invalid_hurst_exponent(self) -> None:
        """Test that invalid Hurst exponent raises error."""
        with pytest.raises(ValueError, match="hurst_exponent must be in range"):
            fbm_covariance_matrix(10, hurst_exponent=0.0)
        with pytest.raises(ValueError, match="hurst_exponent must be in range"):
            fbm_covariance_matrix(10, hurst_exponent=1.0)

    def test_invalid_n(self) -> None:
        """Test that invalid n raises error."""
        with pytest.raises(ValueError, match="n must be at least 1"):
            fbm_covariance_matrix(0, hurst_exponent=0.5)


class TestFBMTrajectory:
    """Tests for 1D FBM trajectory generation."""

    def test_trajectory_shape(self) -> None:
        """Test trajectory has correct shape."""
        trajectory = generate_fbm_trajectory(n_steps=100, hurst_exponent=0.5, seed=42)
        assert trajectory.shape == (100,)

    def test_trajectory_starts_at_zero(self) -> None:
        """Test trajectory starts at origin."""
        trajectory = generate_fbm_trajectory(n_steps=100, hurst_exponent=0.5, seed=42)
        assert trajectory[0] == 0.0

    def test_reproducibility(self) -> None:
        """Test that same seed produces same trajectory."""
        t1 = generate_fbm_trajectory(n_steps=100, hurst_exponent=0.5, seed=42)
        t2 = generate_fbm_trajectory(n_steps=100, hurst_exponent=0.5, seed=42)
        np.testing.assert_array_equal(t1, t2)

    def test_different_seeds_differ(self) -> None:
        """Test that different seeds produce different trajectories."""
        t1 = generate_fbm_trajectory(n_steps=100, hurst_exponent=0.5, seed=42)
        t2 = generate_fbm_trajectory(n_steps=100, hurst_exponent=0.5, seed=43)
        assert not np.allclose(t1, t2)

    def test_scale_factor(self) -> None:
        """Test that scale factor affects variance."""
        t1 = generate_fbm_trajectory(n_steps=100, hurst_exponent=0.5, scale=1.0, seed=42)
        t2 = generate_fbm_trajectory(n_steps=100, hurst_exponent=0.5, scale=2.0, seed=42)
        # Variance should scale with scale^2
        np.testing.assert_array_almost_equal(t2, t1 * 2.0)


class TestFBMTrajectory3D:
    """Tests for 3D FBM trajectory generation."""

    def test_trajectory_3d_shape(self) -> None:
        """Test 3D trajectory has correct shape."""
        trajectory = generate_fbm_trajectory_3d(
            n_steps=100,
            hurst_exponent=0.5,
            diffusion_coefficients=(1.0, 1.0, 1.0),
            dt=1.0,
            seed=42,
        )
        assert trajectory.shape == (100, 3)

    def test_trajectory_3d_starts_at_zero(self) -> None:
        """Test 3D trajectory starts at origin."""
        trajectory = generate_fbm_trajectory_3d(
            n_steps=100,
            hurst_exponent=0.5,
            diffusion_coefficients=(1.0, 1.0, 1.0),
            dt=1.0,
            seed=42,
        )
        np.testing.assert_array_equal(trajectory[0], [0.0, 0.0, 0.0])

    def test_isotropic_diffusion(self) -> None:
        """Test isotropic diffusion coefficient."""
        trajectory = generate_fbm_trajectory_3d(
            n_steps=100,
            hurst_exponent=0.5,
            diffusion_coefficients=1.0,  # Isotropic
            dt=1.0,
            seed=42,
        )
        assert trajectory.shape == (100, 3)


class TestBoundaryConditions:
    """Tests for boundary condition handling."""

    def test_reflective_boundary(self) -> None:
        """Test reflective boundary conditions."""
        # Create positions that go out of bounds
        positions = np.array([[[5.0, 10.0, 10.0], [-5.0, 10.0, 10.0]]])  # 1 particle, 2 steps
        bounds = ((0.0, 20.0), (0.0, 20.0), (0.0, 20.0))

        result = apply_boundary_conditions(positions, bounds, mode="reflective")

        # Position should be reflected back
        assert result[0, 1, 0] == 5.0  # -5 reflected to 5

    def test_periodic_boundary(self) -> None:
        """Test periodic boundary conditions."""
        positions = np.array([[[10.0, 10.0, 10.0], [25.0, 10.0, 10.0]]])  # Goes past upper bound
        bounds = ((0.0, 20.0), (0.0, 20.0), (0.0, 20.0))

        result = apply_boundary_conditions(positions, bounds, mode="periodic")

        # Position should wrap around
        assert result[0, 1, 0] == 5.0  # 25 wraps to 5

    def test_absorbing_boundary(self) -> None:
        """Test absorbing boundary conditions."""
        positions = np.array([[[10.0, 10.0, 10.0], [25.0, 10.0, 10.0]]])
        bounds = ((0.0, 20.0), (0.0, 20.0), (0.0, 20.0))

        result = apply_boundary_conditions(positions, bounds, mode="absorbing")

        # Position should be clamped to boundary
        assert result[0, 1, 0] == 20.0


class TestMSDComputation:
    """Tests for MSD computation."""

    def test_msd_shape(self) -> None:
        """Test MSD output shape."""
        positions = np.random.randn(10, 100, 3).cumsum(axis=1)
        msd = compute_msd(positions)
        assert msd.shape == (51,)  # max_lag = 100 // 2 = 50, plus 1 for lag 0

    def test_msd_starts_at_zero(self) -> None:
        """Test MSD at lag 0 is zero."""
        positions = np.random.randn(10, 100, 3).cumsum(axis=1)
        msd = compute_msd(positions)
        assert msd[0] == 0.0

    def test_msd_single_particle(self) -> None:
        """Test MSD computation for single particle."""
        positions = np.random.randn(100, 3).cumsum(axis=0)
        msd = compute_msd(positions)
        assert msd[0] == 0.0

    def test_msd_linear_for_brownian(self) -> None:
        """Test that MSD is approximately linear for Brownian motion."""
        # Generate many Brownian trajectories
        rng = np.random.default_rng(42)
        n_particles = 100
        n_steps = 200
        D = 1.0  # Diffusion coefficient
        dt = 1.0

        # Generate 3D random walks
        increments = rng.normal(0, np.sqrt(2 * D * dt), (n_particles, n_steps - 1, 3))
        positions = np.zeros((n_particles, n_steps, 3))
        positions[:, 1:, :] = np.cumsum(increments, axis=1)

        msd = compute_msd(positions, max_lag=50)

        # Fit to get alpha
        alpha, _ = fit_diffusion_exponent(msd, dt=dt)

        # Should be close to 1.0 for normal diffusion
        assert 0.9 < alpha < 1.1


class TestDiffusionExponentFit:
    """Tests for diffusion exponent fitting."""

    def test_linear_msd(self) -> None:
        """Test fitting linear MSD."""
        # Create perfect linear MSD: MSD = 6*D*t
        D = 1.0
        dt = 1.0
        lags = np.arange(51)
        msd = 6 * D * lags * dt  # 3D diffusion

        alpha, D_fit = fit_diffusion_exponent(msd, dt=dt)

        assert abs(alpha - 1.0) < 0.01
        assert abs(D_fit - D) < 0.1

    def test_subdiffusive_msd(self) -> None:
        """Test fitting subdiffusive MSD."""
        # Create MSD = A * t^0.6
        alpha_true = 0.6
        A = 6.0
        dt = 1.0
        lags = np.arange(51)
        msd = np.zeros_like(lags, dtype=np.float64)
        msd[1:] = A * (lags[1:] * dt) ** alpha_true

        alpha, _ = fit_diffusion_exponent(msd, dt=dt)

        assert abs(alpha - alpha_true) < 0.05


class TestGenerateBrownianParticles:
    """Tests for the main generate_brownian_particles function."""

    def test_output_shapes(self) -> None:
        """Test that outputs have correct shapes."""
        shape = (20, 16, 32, 32)
        n_particles = 5

        volumes, positions = generate_brownian_particles(
            shape=shape,
            n_particles=n_particles,
            diffusion_coefficient=0.5,
            hurst_exponent=0.5,
            seed=42,
        )

        assert volumes.shape == shape
        assert positions.shape == (n_particles, shape[0], 3)

    def test_with_voxel_size(self) -> None:
        """Test with physical units via VoxelSize."""
        voxel_size = VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1)

        volumes, positions = generate_brownian_particles(
            shape=(10, 16, 32, 32),
            n_particles=3,
            diffusion_coefficient=0.5,
            dt=0.1,
            voxel_size=voxel_size,
            seed=42,
        )

        assert volumes.shape == (10, 16, 32, 32)
        assert positions.shape == (3, 10, 3)

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""
        kwargs = {
            "shape": (10, 16, 32, 32),
            "n_particles": 5,
            "diffusion_coefficient": 0.5,
            "hurst_exponent": 0.5,
            "seed": 42,
        }

        v1, p1 = generate_brownian_particles(**kwargs)
        v2, p2 = generate_brownian_particles(**kwargs)

        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(p1, p2)

    def test_boundary_reflection(self) -> None:
        """Test that particles stay within boundaries."""
        shape = (50, 16, 32, 32)
        particle_sigma = (2.0, 3.0, 3.0)
        margin_z = particle_sigma[0] * 3
        margin_y = particle_sigma[1] * 3
        margin_x = particle_sigma[2] * 3

        _, positions = generate_brownian_particles(
            shape=shape,
            n_particles=10,
            diffusion_coefficient=2.0,  # Large D to test boundaries
            hurst_exponent=0.5,
            boundary_mode="reflective",
            particle_sigma=particle_sigma,
            seed=42,
        )

        # Check all positions are within bounds
        assert np.all(positions[:, :, 0] >= margin_z)
        assert np.all(positions[:, :, 0] <= shape[1] - margin_z)
        assert np.all(positions[:, :, 1] >= margin_y)
        assert np.all(positions[:, :, 1] <= shape[2] - margin_y)
        assert np.all(positions[:, :, 2] >= margin_x)
        assert np.all(positions[:, :, 2] <= shape[3] - margin_x)

    def test_noise_level(self) -> None:
        """Test that noise is added correctly."""
        v_no_noise, _ = generate_brownian_particles(
            shape=(5, 16, 32, 32),
            n_particles=3,
            diffusion_coefficient=0.5,
            noise_level=0.0,
            seed=42,
        )

        v_with_noise, _ = generate_brownian_particles(
            shape=(5, 16, 32, 32),
            n_particles=3,
            diffusion_coefficient=0.5,
            noise_level=0.1,
            seed=42,
        )

        # Volumes should differ due to noise
        assert not np.allclose(v_no_noise, v_with_noise)


@pytest.mark.slow
class TestMSDBehavior:
    """Tests for MSD behavior with different Hurst exponents.

    These tests generate many trajectories and check statistical properties.
    """

    @pytest.mark.parametrize(
        "hurst,expected_alpha",
        [
            (0.5, 1.0),  # Normal diffusion
            (0.3, 0.6),  # Subdiffusion
            (0.7, 1.4),  # Superdiffusion
        ],
    )
    def test_msd_scaling(self, hurst: float, expected_alpha: float) -> None:
        """Test that MSD scales correctly with time.

        For FBM: MSD ~ t^(2H) = t^alpha where alpha = 2H
        """
        n_particles = 50
        n_steps = 300
        D = 1.0

        # Generate many trajectories for ensemble averaging
        rng = np.random.default_rng(42)
        positions = np.zeros((n_particles, n_steps, 3))

        for i in range(n_particles):
            trajectory = generate_fbm_trajectory_3d(
                n_steps=n_steps,
                hurst_exponent=hurst,
                diffusion_coefficients=(D, D, D),
                dt=1.0,
                rng=rng,
            )
            positions[i] = trajectory

        msd = compute_msd(positions, max_lag=n_steps // 4)
        alpha, _ = fit_diffusion_exponent(msd, dt=1.0)

        # Allow 20% tolerance due to statistical fluctuations
        assert abs(alpha - expected_alpha) < 0.2, f"Expected alpha~{expected_alpha}, got {alpha}"
