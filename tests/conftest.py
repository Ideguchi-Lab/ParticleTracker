"""pytest fixtures for pt3d tests."""

import numpy as np
import pandas as pd
import pytest

from pt3d.synth import generate_gaussian_particle, generate_moving_particles


@pytest.fixture
def synthetic_volume() -> np.ndarray:
    """Generate a single 3D volume with known particles."""
    shape = (32, 64, 64)  # z, y, x
    volume = np.zeros(shape, dtype=np.float64)

    # Add particles at known positions
    positions = [
        (10, 20, 30),
        (15, 40, 50),
        (20, 30, 20),
    ]

    for pos in positions:
        particle = generate_gaussian_particle(
            shape, pos, sigma=(2.0, 3.0, 3.0), intensity=1.0
        )
        volume += particle

    # Add small amount of noise
    rng = np.random.default_rng(42)
    volume += rng.normal(0, 0.02, shape)
    return np.clip(volume, 0, None)


@pytest.fixture
def synthetic_4d_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 4D particle data for testing."""
    volumes, ground_truth = generate_moving_particles(
        shape=(10, 32, 64, 64),  # t, z, y, x
        n_particles=5,
        particle_sigma=(2.0, 3.0, 3.0),
        velocity_range=(0.5, 2.0),
        intensity=1.0,
        noise_level=0.02,
        seed=42,
    )
    return volumes, ground_truth


@pytest.fixture
def sample_detections() -> pd.DataFrame:
    """Generate sample detection DataFrame."""
    return pd.DataFrame(
        {
            "frame": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "z": [10.0, 15.0, 20.0, 10.5, 15.5, 20.5, 11.0, 16.0, 21.0],
            "y": [20.0, 40.0, 30.0, 20.5, 40.5, 30.5, 21.0, 41.0, 31.0],
            "x": [30.0, 50.0, 20.0, 30.5, 50.5, 20.5, 31.0, 51.0, 21.0],
            "mass": [100.0] * 9,
            "size": [3.0] * 9,
        }
    )


@pytest.fixture
def sample_tracks(sample_detections: pd.DataFrame) -> pd.DataFrame:
    """Generate sample tracks DataFrame."""
    df = sample_detections.copy()
    # Assign particle IDs
    df["particle"] = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    return df
