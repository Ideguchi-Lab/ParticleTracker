"""Synthetic data generators for testing particle tracking.

This module provides functions to generate synthetic 3D and 4D
particle data for testing detection and tracking algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def generate_sphere_volume(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    intensity: float = 1.0,
) -> NDArray[np.float64]:
    """Generate a 3D volume with a single solid sphere.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape (z, y, x)
    center : tuple[float, float, float]
        Sphere center (z, y, x) in pixels
    radius : float
        Sphere radius in pixels
    intensity : float
        Intensity value inside the sphere

    Returns
    -------
    NDArray[np.float64]
        3D array with sphere

    Examples
    --------
    >>> volume = generate_sphere_volume((32, 64, 64), (16, 32, 32), 5.0)
    >>> volume.shape
    (32, 64, 64)
    """
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(
        (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    )
    volume = np.zeros(shape, dtype=np.float64)
    volume[dist <= radius] = intensity
    return volume


def generate_circle_slice(
    shape: tuple[int, int],
    center: tuple[float, float],
    radius: float,
    intensity: float = 1.0,
) -> NDArray[np.float64]:
    """Generate a 2D slice with a solid circle.

    Parameters
    ----------
    shape : tuple[int, int]
        Slice shape (y, x)
    center : tuple[float, float]
        Circle center (y, x) in pixels
    radius : float
        Circle radius in pixels
    intensity : float
        Intensity value inside the circle

    Returns
    -------
    NDArray[np.float64]
        2D array with circle

    Examples
    --------
    >>> slice_2d = generate_circle_slice((64, 64), (32, 32), 10.0)
    >>> slice_2d.shape
    (64, 64)
    """
    y, x = np.ogrid[: shape[0], : shape[1]]
    dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    slice_2d = np.zeros(shape, dtype=np.float64)
    slice_2d[dist <= radius] = intensity
    return slice_2d


def generate_gaussian_particle(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    sigma: tuple[float, float, float],
    intensity: float = 1.0,
) -> NDArray[np.float64]:
    """Generate a 3D Gaussian particle.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape (z, y, x)
    center : tuple[float, float, float]
        Particle center (z, y, x) in pixels
    sigma : tuple[float, float, float]
        Gaussian sigma (sz, sy, sx) in pixels
    intensity : float
        Peak intensity

    Returns
    -------
    NDArray[np.float64]
        3D array with Gaussian particle

    Examples
    --------
    >>> particle = generate_gaussian_particle(
    ...     (32, 64, 64), (16, 32, 32), (2.0, 3.0, 3.0)
    ... )
    >>> particle.max()  # Should be close to 1.0
    1.0
    """
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    gaussian = np.exp(
        -(
            (z - center[0]) ** 2 / (2 * sigma[0] ** 2)
            + (y - center[1]) ** 2 / (2 * sigma[1] ** 2)
            + (x - center[2]) ** 2 / (2 * sigma[2] ** 2)
        )
    )
    return gaussian * intensity


def generate_multiple_particles(
    shape: tuple[int, int, int],
    centers: list[tuple[float, float, float]],
    sigma: tuple[float, float, float],
    intensities: list[float] | None = None,
) -> NDArray[np.float64]:
    """Generate a 3D volume with multiple Gaussian particles.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape (z, y, x)
    centers : list[tuple[float, float, float]]
        List of particle centers (z, y, x)
    sigma : tuple[float, float, float]
        Gaussian sigma for all particles
    intensities : list[float] | None
        Intensities for each particle. If None, all are 1.0

    Returns
    -------
    NDArray[np.float64]
        3D array with multiple particles
    """
    if intensities is None:
        intensities = [1.0] * len(centers)

    volume = np.zeros(shape, dtype=np.float64)
    for center, intensity in zip(centers, intensities, strict=True):
        particle = generate_gaussian_particle(shape, center, sigma, intensity)
        volume += particle

    return volume


def generate_moving_particles(
    shape: tuple[int, int, int, int],
    n_particles: int,
    particle_sigma: tuple[float, float, float],
    velocity_range: tuple[float, float],
    intensity: float = 1.0,
    noise_level: float = 0.1,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate 4D time series with moving particles.

    Creates a synthetic dataset with particles moving through
    a 3D volume over time. Particles reflect at boundaries.

    Parameters
    ----------
    shape : tuple[int, int, int, int]
        Volume shape (t, z, y, x)
    n_particles : int
        Number of particles
    particle_sigma : tuple[float, float, float]
        Gaussian sigma (sz, sy, sx) for particles
    velocity_range : tuple[float, float]
        (min, max) velocity magnitude per frame in pixels
    intensity : float
        Particle intensity
    noise_level : float
        Gaussian noise standard deviation
    seed : int | None
        Random seed for reproducibility

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        (volumes, ground_truth_positions)
        volumes: shape (t, z, y, x)
        ground_truth_positions: shape (n_particles, t, 3) for [z, y, x]

    Examples
    --------
    >>> volumes, positions = generate_moving_particles(
    ...     shape=(10, 32, 64, 64),
    ...     n_particles=5,
    ...     particle_sigma=(2.0, 3.0, 3.0),
    ...     velocity_range=(0.5, 2.0),
    ...     seed=42,
    ... )
    >>> volumes.shape
    (10, 32, 64, 64)
    >>> positions.shape
    (5, 10, 3)
    """
    rng = np.random.default_rng(seed)
    n_frames, nz, ny, nx = shape

    # Margin from boundaries based on particle size
    margin_z = particle_sigma[0] * 3
    margin_y = particle_sigma[1] * 3
    margin_x = particle_sigma[2] * 3

    # Initialize particle positions
    positions = np.zeros((n_particles, n_frames, 3))
    positions[:, 0, 0] = rng.uniform(margin_z, nz - margin_z, n_particles)
    positions[:, 0, 1] = rng.uniform(margin_y, ny - margin_y, n_particles)
    positions[:, 0, 2] = rng.uniform(margin_x, nx - margin_x, n_particles)

    # Generate random velocities (direction and magnitude)
    vel_min, vel_max = velocity_range
    velocities = np.zeros((n_particles, 3))
    for i in range(n_particles):
        # Random direction
        direction = rng.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        # Random magnitude
        magnitude = rng.uniform(vel_min, vel_max)
        velocities[i] = direction * magnitude

    # Propagate positions with boundary reflection
    for t in range(1, n_frames):
        new_positions = positions[:, t - 1, :] + velocities

        # Reflect at boundaries
        for dim, (margin, limit) in enumerate(
            [(margin_z, nz), (margin_y, ny), (margin_x, nx)]
        ):
            # Lower boundary
            mask_low = new_positions[:, dim] < margin
            velocities[mask_low, dim] = np.abs(velocities[mask_low, dim])
            new_positions[mask_low, dim] = 2 * margin - new_positions[mask_low, dim]

            # Upper boundary
            mask_high = new_positions[:, dim] > limit - margin
            velocities[mask_high, dim] = -np.abs(velocities[mask_high, dim])
            new_positions[mask_high, dim] = (
                2 * (limit - margin) - new_positions[mask_high, dim]
            )

        positions[:, t, :] = new_positions

    # Generate volumes
    volumes = np.zeros(shape, dtype=np.float64)
    for t in range(n_frames):
        for p in range(n_particles):
            center = tuple(positions[p, t, :])
            particle = generate_gaussian_particle(
                (nz, ny, nx),
                center,
                particle_sigma,
                intensity,
            )
            volumes[t] += particle

    # Add noise
    if noise_level > 0:
        volumes += rng.normal(0, noise_level, volumes.shape)
        volumes = np.clip(volumes, 0, None)

    return volumes, positions


def save_synthetic_data(
    output_dir: str,
    volumes: NDArray[np.float64],
    positions: NDArray[np.float64] | None = None,
    prefix: str = "synthetic",
) -> dict[str, str]:
    """Save synthetic data to numpy files.

    Parameters
    ----------
    output_dir : str
        Output directory
    volumes : NDArray[np.float64]
        Volume data
    positions : NDArray[np.float64] | None
        Ground truth positions
    prefix : str
        Filename prefix

    Returns
    -------
    dict[str, str]
        Paths to saved files
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save volumes
    volume_path = output_dir / f"{prefix}_volumes.npy"
    np.save(volume_path, volumes)
    paths["volumes"] = str(volume_path)

    # Save positions if provided
    if positions is not None:
        positions_path = output_dir / f"{prefix}_positions.npy"
        np.save(positions_path, positions)
        paths["positions"] = str(positions_path)

    return paths
