"""Tests for Brownian motion configuration module."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pt3d.config import VoxelSize
from pt3d.synth.brownian import generate_brownian_particles_from_config
from pt3d.synth.brownian_config import (
    BrownianSimulationConfig,
    DiffusionConfig,
    ParticlePopulationConfig,
    ParticleTypeConfig,
)


class TestParticleTypeConfig:
    """Tests for ParticleTypeConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ParticleTypeConfig()
        assert config.shape == "gaussian"
        assert config.sigma == (2.0, 3.0, 3.0)
        assert config.intensity == 1.0
        assert config.radius is None

    def test_sphere_config(self) -> None:
        """Test sphere particle configuration."""
        config = ParticleTypeConfig(shape="sphere", radius=5.0)
        assert config.shape == "sphere"
        assert config.radius == 5.0

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields raise validation error."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ParticleTypeConfig(shape="gaussian", extra_field="invalid")  # type: ignore[call-arg]


class TestDiffusionConfig:
    """Tests for DiffusionConfig."""

    def test_required_diffusion_coefficient(self) -> None:
        """Test that diffusion_coefficient is required."""
        with pytest.raises(ValidationError):
            DiffusionConfig()  # type: ignore[call-arg]

    def test_isotropic_diffusion(self) -> None:
        """Test isotropic diffusion coefficient."""
        config = DiffusionConfig(diffusion_coefficient=0.5)
        assert config.diffusion_coefficient == 0.5
        assert config.hurst_exponent == 0.5
        assert config.dt == 1.0

    def test_anisotropic_diffusion(self) -> None:
        """Test anisotropic diffusion coefficient."""
        config = DiffusionConfig(diffusion_coefficient=(0.1, 0.5, 1.0))
        assert config.diffusion_coefficient == (0.1, 0.5, 1.0)

    def test_hurst_exponent_validation(self) -> None:
        """Test Hurst exponent must be in (0, 1)."""
        with pytest.raises(ValidationError):
            DiffusionConfig(diffusion_coefficient=0.5, hurst_exponent=0.0)
        with pytest.raises(ValidationError):
            DiffusionConfig(diffusion_coefficient=0.5, hurst_exponent=1.0)

    def test_dt_validation(self) -> None:
        """Test dt must be positive."""
        with pytest.raises(ValidationError):
            DiffusionConfig(diffusion_coefficient=0.5, dt=0.0)
        with pytest.raises(ValidationError):
            DiffusionConfig(diffusion_coefficient=0.5, dt=-1.0)

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields raise validation error."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DiffusionConfig(diffusion_coefficient=0.5, extra_field="invalid")  # type: ignore[call-arg]


class TestParticlePopulationConfig:
    """Tests for ParticlePopulationConfig."""

    def test_minimal_config(self) -> None:
        """Test minimal population configuration."""
        config = ParticlePopulationConfig(
            n_particles=10,
            diffusion=DiffusionConfig(diffusion_coefficient=0.5),
        )
        assert config.n_particles == 10
        assert config.particle_type.shape == "gaussian"  # Default
        assert config.initial_positions is None

    def test_n_particles_validation(self) -> None:
        """Test n_particles must be positive."""
        with pytest.raises(ValidationError):
            ParticlePopulationConfig(
                n_particles=0,
                diffusion=DiffusionConfig(diffusion_coefficient=0.5),
            )

    def test_initial_positions(self) -> None:
        """Test explicit initial positions."""
        positions = [(10.0, 20.0, 30.0), (15.0, 25.0, 35.0)]
        config = ParticlePopulationConfig(
            n_particles=2,
            diffusion=DiffusionConfig(diffusion_coefficient=0.5),
            initial_positions=positions,
        )
        assert config.initial_positions == positions

    def test_initial_region(self) -> None:
        """Test initial region configuration."""
        region = ((5.0, 25.0), (10.0, 50.0), (10.0, 50.0))
        config = ParticlePopulationConfig(
            n_particles=5,
            diffusion=DiffusionConfig(diffusion_coefficient=0.5),
            initial_region=region,
        )
        assert config.initial_region == region


class TestBrownianSimulationConfig:
    """Tests for BrownianSimulationConfig."""

    def test_minimal_config(self) -> None:
        """Test minimal simulation configuration."""
        config = BrownianSimulationConfig(
            shape=(50, 32, 64, 64),
            voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
            populations=[
                ParticlePopulationConfig(
                    n_particles=5,
                    diffusion=DiffusionConfig(diffusion_coefficient=0.5),
                ),
            ],
        )
        assert config.shape == (50, 32, 64, 64)
        assert config.boundary_mode == "reflective"
        assert config.noise_level == 0.1
        assert config.seed is None

    def test_multiple_populations(self) -> None:
        """Test configuration with multiple populations."""
        config = BrownianSimulationConfig(
            shape=(50, 32, 64, 64),
            voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
            populations=[
                ParticlePopulationConfig(
                    n_particles=5,
                    diffusion=DiffusionConfig(diffusion_coefficient=0.5, hurst_exponent=0.5),
                ),
                ParticlePopulationConfig(
                    n_particles=3,
                    diffusion=DiffusionConfig(diffusion_coefficient=0.3, hurst_exponent=0.3),
                ),
            ],
        )
        assert len(config.populations) == 2

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields raise validation error."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BrownianSimulationConfig(
                shape=(50, 32, 64, 64),
                voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
                populations=[
                    ParticlePopulationConfig(
                        n_particles=5,
                        diffusion=DiffusionConfig(diffusion_coefficient=0.5),
                    ),
                ],
                invalid_field="extra",  # type: ignore[call-arg]
            )


class TestYAMLSerialization:
    """Tests for YAML serialization and deserialization."""

    def test_to_yaml_and_from_yaml(self, tmp_path: Path) -> None:
        """Test round-trip YAML serialization."""
        config = BrownianSimulationConfig(
            shape=(50, 32, 64, 64),
            voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
            populations=[
                ParticlePopulationConfig(
                    n_particles=5,
                    particle_type=ParticleTypeConfig(
                        shape="gaussian",
                        sigma=(2.0, 3.0, 3.0),
                        intensity=1.0,
                    ),
                    diffusion=DiffusionConfig(
                        diffusion_coefficient=0.5,
                        hurst_exponent=0.5,
                        dt=0.1,
                    ),
                ),
            ],
            boundary_mode="reflective",
            noise_level=0.05,
            seed=42,
        )

        # Save to YAML
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        # Load from YAML
        loaded_config = BrownianSimulationConfig.from_yaml(yaml_path)

        # Compare
        assert loaded_config.shape == config.shape
        assert loaded_config.voxel_size == config.voxel_size
        assert len(loaded_config.populations) == len(config.populations)
        assert loaded_config.boundary_mode == config.boundary_mode
        assert loaded_config.noise_level == config.noise_level
        assert loaded_config.seed == config.seed

    def test_from_yaml_file_not_found(self) -> None:
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            BrownianSimulationConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_invalid_content(self, tmp_path: Path) -> None:
        """Test loading invalid YAML content raises validation error."""
        yaml_path = tmp_path / "invalid_config.yaml"
        yaml_path.write_text("shape: invalid\n")

        with pytest.raises(ValidationError):
            BrownianSimulationConfig.from_yaml(yaml_path)


class TestGenerateFromConfig:
    """Tests for generate_brownian_particles_from_config function."""

    def test_single_population(self) -> None:
        """Test generating data with single population."""
        config = BrownianSimulationConfig(
            shape=(20, 16, 32, 32),
            voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
            populations=[
                ParticlePopulationConfig(
                    n_particles=5,
                    diffusion=DiffusionConfig(diffusion_coefficient=0.5),
                ),
            ],
            seed=42,
        )

        volumes, positions, labels = generate_brownian_particles_from_config(config)

        assert volumes.shape == (20, 16, 32, 32)
        assert positions.shape == (5, 20, 3)
        assert labels.shape == (5,)
        assert np.all(labels == 0)  # All from population 0

    def test_multiple_populations(self) -> None:
        """Test generating data with multiple populations."""
        config = BrownianSimulationConfig(
            shape=(20, 16, 32, 32),
            voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
            populations=[
                ParticlePopulationConfig(
                    n_particles=3,
                    diffusion=DiffusionConfig(diffusion_coefficient=0.5, hurst_exponent=0.5),
                ),
                ParticlePopulationConfig(
                    n_particles=2,
                    diffusion=DiffusionConfig(diffusion_coefficient=0.3, hurst_exponent=0.3),
                ),
            ],
            seed=42,
        )

        volumes, positions, labels = generate_brownian_particles_from_config(config)

        assert volumes.shape == (20, 16, 32, 32)
        assert positions.shape == (5, 20, 3)  # 3 + 2 = 5 particles
        assert labels.shape == (5,)
        assert np.sum(labels == 0) == 3  # 3 from population 0
        assert np.sum(labels == 1) == 2  # 2 from population 1

    def test_explicit_initial_positions(self) -> None:
        """Test with explicit initial positions."""
        initial_pos = [(8.0, 16.0, 16.0), (8.0, 16.0, 16.0)]
        config = BrownianSimulationConfig(
            shape=(10, 16, 32, 32),
            voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
            populations=[
                ParticlePopulationConfig(
                    n_particles=2,
                    diffusion=DiffusionConfig(diffusion_coefficient=0.0),  # No diffusion
                    initial_positions=initial_pos,
                ),
            ],
            seed=42,
        )

        _, positions, _ = generate_brownian_particles_from_config(config)

        # With D=0, particles should stay at initial positions
        np.testing.assert_array_almost_equal(positions[0, 0, :], initial_pos[0])
        np.testing.assert_array_almost_equal(positions[1, 0, :], initial_pos[1])

    def test_initial_positions_count_mismatch(self) -> None:
        """Test that mismatched initial positions count raises error."""
        config = BrownianSimulationConfig(
            shape=(10, 16, 32, 32),
            voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
            populations=[
                ParticlePopulationConfig(
                    n_particles=3,  # 3 particles
                    diffusion=DiffusionConfig(diffusion_coefficient=0.5),
                    initial_positions=[(10.0, 16.0, 16.0)],  # Only 1 position
                ),
            ],
            seed=42,
        )

        with pytest.raises(ValueError, match="Number of initial_positions"):
            generate_brownian_particles_from_config(config)

    def test_reproducibility(self) -> None:
        """Test that same seed produces same results."""
        config = BrownianSimulationConfig(
            shape=(10, 16, 32, 32),
            voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
            populations=[
                ParticlePopulationConfig(
                    n_particles=5,
                    diffusion=DiffusionConfig(diffusion_coefficient=0.5),
                ),
            ],
            seed=42,
        )

        v1, p1, l1 = generate_brownian_particles_from_config(config)
        v2, p2, l2 = generate_brownian_particles_from_config(config)

        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(l1, l2)

    def test_sphere_particles(self) -> None:
        """Test generating sphere particles."""
        config = BrownianSimulationConfig(
            shape=(5, 16, 32, 32),
            voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
            populations=[
                ParticlePopulationConfig(
                    n_particles=2,
                    particle_type=ParticleTypeConfig(shape="sphere", radius=3.0),
                    diffusion=DiffusionConfig(diffusion_coefficient=0.5),
                ),
            ],
            seed=42,
        )

        volumes, positions, _ = generate_brownian_particles_from_config(config)

        assert volumes.shape == (5, 16, 32, 32)
        assert positions.shape == (2, 5, 3)
