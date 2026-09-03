"""Brownian motion simulation for particle tracking.

This module provides functions to generate synthetic particle trajectories
following various diffusion models including normal Brownian motion,
subdiffusion, and superdiffusion using Fractional Brownian Motion (FBM).

The implementation uses Cholesky decomposition of the FBM covariance matrix
for exact trajectory generation. For normal Brownian motion (H=0.5), this
reduces to standard Gaussian random walks.

References
----------
.. [1] Mandelbrot, B. B., & Van Ness, J. W. (1968). Fractional Brownian
       motions, fractional noises and applications. SIAM review, 10(4), 422-437.
.. [2] Dieker, T. (2004). Simulation of fractional Brownian motion.
       Thesis, University of Twente.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pt3d.config import VoxelSize
    from pt3d.synth.brownian_config import BrownianSimulationConfig

from pt3d.synth.generators import generate_gaussian_particle, generate_sphere_volume


def fbm_covariance_matrix(n: int, hurst_exponent: float) -> NDArray[np.float64]:
    """Generate the covariance matrix for Fractional Brownian Motion.

    The covariance matrix for FBM is defined as:
    C(s,t) = 0.5 * (|s|^(2H) + |t|^(2H) - |t-s|^(2H))

    Parameters
    ----------
    n : int
        Size of the covariance matrix (number of time points).
    hurst_exponent : float
        Hurst exponent H in range (0, 1).
        - H = 0.5: Standard Brownian motion
        - H < 0.5: Antipersistent (subdiffusion)
        - H > 0.5: Persistent (superdiffusion)

    Returns
    -------
    NDArray[np.float64]
        Covariance matrix of shape (n, n).

    Examples
    --------
    >>> cov = fbm_covariance_matrix(100, 0.5)
    >>> cov.shape
    (100, 100)
    """
    if n < 1:
        msg = "n must be at least 1"
        raise ValueError(msg)
    if not 0 < hurst_exponent < 1:
        msg = "hurst_exponent must be in range (0, 1)"
        raise ValueError(msg)

    # Time indices (1, 2, ..., n)
    i = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
    j = np.arange(1, n + 1, dtype=np.float64).reshape(1, -1)

    h2 = 2 * hurst_exponent

    # FBM covariance: C(s,t) = 0.5 * (|s|^(2H) + |t|^(2H) - |s-t|^(2H))
    cov = 0.5 * (np.abs(i) ** h2 + np.abs(j) ** h2 - np.abs(i - j) ** h2)

    # Add small diagonal for numerical stability
    cov += np.eye(n) * 1e-10

    return cov


def generate_fbm_trajectory(
    n_steps: int,
    hurst_exponent: float,
    scale: float = 1.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate a 1D Fractional Brownian Motion trajectory.

    Uses Cholesky decomposition of the covariance matrix for exact FBM generation.

    Parameters
    ----------
    n_steps : int
        Number of time steps to generate.
    hurst_exponent : float
        Hurst exponent H in range (0, 1).
        - H = 0.5: Standard Brownian motion
        - H < 0.5: Antipersistent (subdiffusion)
        - H > 0.5: Persistent (superdiffusion)
    scale : float
        Scale factor for the trajectory. For standard BM with diffusion
        coefficient D: scale = sqrt(2*D*dt).
    seed : int | None
        Random seed. Ignored if rng is provided.
    rng : np.random.Generator | None
        Random number generator. If provided, seed is ignored.

    Returns
    -------
    NDArray[np.float64]
        FBM trajectory of shape (n_steps,). First value is 0 (starting point).

    Examples
    --------
    >>> trajectory = generate_fbm_trajectory(100, 0.5, seed=42)
    >>> trajectory.shape
    (100,)
    >>> trajectory[0]
    0.0
    """
    if n_steps < 1:
        msg = "n_steps must be at least 1"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng(seed)

    if n_steps == 1:
        return np.array([0.0])

    # Generate covariance matrix for n_steps - 1 increments
    cov = fbm_covariance_matrix(n_steps - 1, hurst_exponent)

    # Cholesky decomposition
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Fallback: add more regularization if Cholesky fails
        cov += np.eye(n_steps - 1) * 1e-8
        chol = np.linalg.cholesky(cov)

    # Generate standard normal samples
    z = rng.standard_normal(n_steps - 1)

    # Transform to FBM (cumulative)
    fbm_values = chol @ z

    # Prepend 0 as starting point and scale
    trajectory = np.zeros(n_steps)
    trajectory[1:] = fbm_values * scale

    return trajectory


def generate_fbm_trajectory_3d(
    n_steps: int,
    hurst_exponent: float,
    diffusion_coefficients: tuple[float, float, float] | float,
    dt: float = 1.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate a 3D anisotropic FBM trajectory.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    hurst_exponent : float
        Hurst exponent H (same for all axes).
    diffusion_coefficients : tuple[float, float, float] | float
        Diffusion coefficients (Dz, Dy, Dx) in um^2/s.
        If float, uses isotropic diffusion.
    dt : float
        Time step in seconds.
    seed : int | None
        Random seed. Ignored if rng is provided.
    rng : np.random.Generator | None
        Random number generator.

    Returns
    -------
    NDArray[np.float64]
        3D trajectory of shape (n_steps, 3) for [z, y, x] in micrometers.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Handle isotropic diffusion
    if isinstance(diffusion_coefficients, int | float):
        diffusion_coefficients = (
            float(diffusion_coefficients),
            float(diffusion_coefficients),
            float(diffusion_coefficients),
        )

    trajectory = np.zeros((n_steps, 3))

    for dim, d_coeff in enumerate(diffusion_coefficients):
        if d_coeff > 0:
            # Scale factor: sqrt(2*D) * dt^H for FBM
            # This gives proper MSD scaling: MSD ~ 2*D * t^(2H)
            scale = np.sqrt(2 * d_coeff) * (dt**hurst_exponent)
            fbm_1d = generate_fbm_trajectory(
                n_steps=n_steps,
                hurst_exponent=hurst_exponent,
                scale=scale,
                rng=rng,
            )
            trajectory[:, dim] = fbm_1d

    return trajectory


def apply_boundary_conditions(
    positions: NDArray[np.float64],
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    mode: Literal["reflective", "periodic", "absorbing"] = "reflective",
) -> NDArray[np.float64]:
    """Apply boundary conditions to particle positions.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Particle positions of shape (n_particles, n_steps, 3) or (n_steps, 3).
    bounds : tuple[tuple[float, float], ...]
        Boundary limits ((z_min, z_max), (y_min, y_max), (x_min, x_max)).
    mode : Literal["reflective", "periodic", "absorbing"]
        Boundary condition type:
        - "reflective": Particles bounce back at boundaries
        - "periodic": Particles wrap around
        - "absorbing": Particles stick at boundaries

    Returns
    -------
    NDArray[np.float64]
        Positions with boundary conditions applied.
    """
    positions = positions.copy()
    original_shape = positions.shape

    # Reshape to (n_particles, n_steps, 3) if needed
    if positions.ndim == 2:
        positions = positions.reshape(1, *positions.shape)

    for dim in range(3):
        lower, upper = bounds[dim]
        range_width = upper - lower

        if mode == "reflective":
            # Apply reflective boundary conditions
            pos_dim = positions[:, :, dim]

            # Handle multiple reflections iteratively
            max_iterations = 100
            for _ in range(max_iterations):
                below = pos_dim < lower
                above = pos_dim > upper

                if not np.any(below) and not np.any(above):
                    break

                # Reflect at lower boundary
                pos_dim[below] = 2 * lower - pos_dim[below]

                # Reflect at upper boundary
                pos_dim[above] = 2 * upper - pos_dim[above]

            positions[:, :, dim] = pos_dim

        elif mode == "periodic":
            # Wrap around
            positions[:, :, dim] = ((positions[:, :, dim] - lower) % range_width) + lower

        elif mode == "absorbing":
            # Clamp to boundaries
            positions[:, :, dim] = np.clip(positions[:, :, dim], lower, upper)

    # Restore original shape
    if len(original_shape) == 2:
        positions = positions.reshape(original_shape)

    return positions


def compute_msd(
    positions: NDArray[np.float64],
    max_lag: int | None = None,
) -> NDArray[np.float64]:
    """Compute Mean Squared Displacement from trajectories.

    Parameters
    ----------
    positions : NDArray[np.float64]
        Particle positions. Can be:
        - Shape (n_particles, n_steps, 3): Multiple particles
        - Shape (n_steps, 3): Single particle
        - Shape (n_particles, n_steps, 1) or (n_steps, 1): 1D trajectory
    max_lag : int | None
        Maximum time lag to compute. If None, uses n_steps // 2.

    Returns
    -------
    NDArray[np.float64]
        MSD values for each time lag, shape (max_lag + 1,).
        msd[0] is always 0 (lag 0).

    Examples
    --------
    >>> positions = np.random.randn(10, 100, 3).cumsum(axis=1)
    >>> msd = compute_msd(positions)
    >>> msd[0]
    0.0
    """
    # Handle different input shapes
    if positions.ndim == 2:
        positions = positions.reshape(1, *positions.shape)

    _, n_steps, _ = positions.shape

    if max_lag is None:
        max_lag = n_steps // 2

    max_lag = min(max_lag, n_steps - 1)

    msd = np.zeros(max_lag + 1)
    msd[0] = 0.0  # MSD at lag 0 is always 0

    for lag in range(1, max_lag + 1):
        # Compute squared displacements for all particles and time points
        displacements = positions[:, lag:, :] - positions[:, :-lag, :]
        squared_displacements = np.sum(displacements**2, axis=-1)
        msd[lag] = np.mean(squared_displacements)

    return msd


def fit_diffusion_exponent(
    msd: NDArray[np.float64],
    dt: float = 1.0,
    fit_range: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Fit MSD data to extract diffusion exponent alpha.

    Fits MSD = A * t^alpha using log-log linear regression.
    For normal diffusion (alpha=1), A = 2*n_dim*D where n_dim is
    the number of spatial dimensions.

    Parameters
    ----------
    msd : NDArray[np.float64]
        Mean squared displacement values.
    dt : float
        Time step between frames.
    fit_range : tuple[int, int] | None
        Range of time lags to fit (start, end). If None, uses (1, len(msd)//2).

    Returns
    -------
    tuple[float, float]
        (alpha, D_apparent) where alpha is the diffusion exponent
        and D_apparent is the generalized diffusion coefficient in the
        model MSD = 6*D*t^alpha (units: um^2/s^alpha).

    Examples
    --------
    >>> msd = np.array([0, 1, 2, 3, 4, 5])  # Linear MSD
    >>> alpha, D = fit_diffusion_exponent(msd, dt=1.0)
    >>> abs(alpha - 1.0) < 0.1  # Should be close to 1 for normal diffusion
    True
    """
    if fit_range is None:
        fit_range = (1, max(2, len(msd) // 2))

    start, end = fit_range
    start = max(1, start)  # Exclude lag 0
    end = min(end, len(msd))

    if end <= start:
        msg = f"Invalid fit_range: start={start}, end={end}"
        raise ValueError(msg)

    # Extract data for fitting
    lags = np.arange(start, end)
    msd_values = msd[start:end]

    # Filter out zero or negative values for log
    valid = msd_values > 0
    if not np.any(valid):
        return 1.0, 0.0

    lags = lags[valid]
    msd_values = msd_values[valid]

    # Log-log linear fit: log(MSD) = log(A) + alpha * log(t)
    log_t = np.log(lags * dt)
    log_msd = np.log(msd_values)

    # Linear regression
    coeffs = np.polyfit(log_t, log_msd, 1)
    alpha = coeffs[0]
    log_amplitude = coeffs[1]
    amplitude = np.exp(log_amplitude)

    # For 3D diffusion: MSD = 6*D*t^alpha
    # So D_apparent = A / 6 (units: um^2/s^alpha)
    d_apparent = amplitude / 6

    return float(alpha), float(d_apparent)


def generate_brownian_particles(
    shape: tuple[int, int, int, int],
    n_particles: int,
    diffusion_coefficient: tuple[float, float, float] | float,
    dt: float = 1.0,
    hurst_exponent: float = 0.5,
    voxel_size: VoxelSize | None = None,
    intensity: float = 1.0,
    particle_sigma: tuple[float, float, float] = (2.0, 3.0, 3.0),
    noise_level: float = 0.1,
    boundary_mode: Literal["reflective", "periodic", "absorbing"] = "reflective",
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate 4D time series with particles following Brownian/anomalous diffusion.

    Parameters
    ----------
    shape : tuple[int, int, int, int]
        Volume shape (t, z, y, x) in pixels.
    n_particles : int
        Number of particles to simulate.
    diffusion_coefficient : tuple[float, float, float] | float
        Diffusion coefficient D in um^2/s for each axis (Dz, Dy, Dx).
        If float, uses isotropic diffusion.
    dt : float
        Time step between frames in seconds. Default is 1.0.
    hurst_exponent : float
        Hurst exponent H controlling diffusion type:
        - H = 0.5: Normal Brownian motion (MSD ~ t)
        - H < 0.5: Subdiffusion (MSD ~ t^alpha, alpha < 1)
        - H > 0.5: Superdiffusion (MSD ~ t^alpha, alpha > 1)
        Note: alpha = 2H
    voxel_size : VoxelSize | None
        Physical voxel dimensions for coordinate conversion.
        If None, assumes 1 um/pixel isotropic (coordinates in um = pixels).
    intensity : float
        Peak intensity of particles. Default is 1.0.
    particle_sigma : tuple[float, float, float]
        Gaussian sigma (sz, sy, sx) for particle PSF in pixels.
    noise_level : float
        Gaussian noise standard deviation. Default is 0.1.
    boundary_mode : Literal["reflective", "periodic", "absorbing"]
        Boundary condition type. Default is "reflective".
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        (volumes, ground_truth_positions)
        volumes: shape (t, z, y, x)
        ground_truth_positions: shape (n_particles, t, 3) for [z, y, x] in pixels

    Examples
    --------
    >>> from pt3d.config import VoxelSize
    >>> voxel_size = VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1)
    >>> volumes, positions = generate_brownian_particles(
    ...     shape=(100, 32, 64, 64),
    ...     n_particles=10,
    ...     diffusion_coefficient=0.5,
    ...     dt=0.1,
    ...     hurst_exponent=0.5,
    ...     voxel_size=voxel_size,
    ...     seed=42,
    ... )
    >>> volumes.shape
    (100, 32, 64, 64)
    >>> positions.shape
    (10, 100, 3)
    """
    rng = np.random.default_rng(seed)
    n_frames, nz, ny, nx = shape

    # Margin from boundaries based on particle size
    margin_z = particle_sigma[0] * 3
    margin_y = particle_sigma[1] * 3
    margin_x = particle_sigma[2] * 3

    # Convert voxel size to array for coordinate conversion
    if voxel_size is not None:
        voxel_array = np.array([voxel_size.z_um, voxel_size.y_um, voxel_size.x_um], dtype=np.float64)
    else:
        voxel_array = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # Handle isotropic diffusion coefficient
    if isinstance(diffusion_coefficient, int | float):
        d_tuple = (
            float(diffusion_coefficient),
            float(diffusion_coefficient),
            float(diffusion_coefficient),
        )
    else:
        d_tuple = diffusion_coefficient

    # Initialize positions array
    positions = np.zeros((n_particles, n_frames, 3))

    # Generate initial positions
    initial_z = rng.uniform(margin_z, nz - margin_z, n_particles)
    initial_y = rng.uniform(margin_y, ny - margin_y, n_particles)
    initial_x = rng.uniform(margin_x, nx - margin_x, n_particles)

    # Generate trajectories for each particle
    for i in range(n_particles):
        # Generate 3D FBM trajectory in physical units (micrometers)
        trajectory_um = generate_fbm_trajectory_3d(
            n_steps=n_frames,
            hurst_exponent=hurst_exponent,
            diffusion_coefficients=d_tuple,
            dt=dt,
            rng=rng,
        )

        # Convert to pixel coordinates
        trajectory_px = trajectory_um / voxel_array

        # Add initial position
        positions[i, :, 0] = initial_z[i] + trajectory_px[:, 0]
        positions[i, :, 1] = initial_y[i] + trajectory_px[:, 1]
        positions[i, :, 2] = initial_x[i] + trajectory_px[:, 2]

    # Apply boundary conditions
    bounds = (
        (margin_z, nz - margin_z),
        (margin_y, ny - margin_y),
        (margin_x, nx - margin_x),
    )
    positions = apply_boundary_conditions(positions, bounds, mode=boundary_mode)

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


def generate_brownian_particles_from_config(
    config: BrownianSimulationConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Generate Brownian motion data from a configuration object.

    This function supports multiple particle populations with different
    diffusion properties, allowing simulation of heterogeneous systems.

    Parameters
    ----------
    config : BrownianSimulationConfig
        Configuration object specifying simulation parameters.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]
        (volumes, positions, population_labels)
        volumes: shape (t, z, y, x) - synthetic image volumes
        positions: shape (total_particles, t, 3) - ground truth positions [z, y, x]
        population_labels: shape (total_particles,) - population index for each particle

    Examples
    --------
    >>> from pt3d.config import VoxelSize
    >>> from pt3d.synth.brownian_config import (
    ...     BrownianSimulationConfig,
    ...     DiffusionConfig,
    ...     ParticlePopulationConfig,
    ... )
    >>> config = BrownianSimulationConfig(
    ...     shape=(50, 32, 64, 64),
    ...     voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
    ...     populations=[
    ...         ParticlePopulationConfig(
    ...             n_particles=5,
    ...             diffusion=DiffusionConfig(
    ...                 diffusion_coefficient=0.5,
    ...                 hurst_exponent=0.5,
    ...             ),
    ...         ),
    ...     ],
    ...     seed=42,
    ... )
    >>> volumes, positions, labels = generate_brownian_particles_from_config(config)
    """
    rng = np.random.default_rng(config.seed)
    n_frames, nz, ny, nx = config.shape

    # Collect all positions and labels
    all_positions = []
    all_labels = []

    # Initialize volumes
    volumes = np.zeros(config.shape, dtype=np.float64)

    for pop_idx, pop in enumerate(config.populations):
        # Get particle appearance settings
        particle_type = pop.particle_type

        # Get diffusion settings
        diff = pop.diffusion
        d_coeff = diff.diffusion_coefficient
        hurst = diff.hurst_exponent
        dt = diff.dt

        # Convert voxel size to array
        voxel_array = np.array(
            [config.voxel_size.z_um, config.voxel_size.y_um, config.voxel_size.x_um],
            dtype=np.float64,
        )

        # Handle isotropic diffusion
        d_tuple = (float(d_coeff), float(d_coeff), float(d_coeff)) if isinstance(d_coeff, int | float) else d_coeff

        # Calculate margins based on particle size
        if particle_type.shape == "gaussian":
            margin_z = particle_type.sigma[0] * 3
            margin_y = particle_type.sigma[1] * 3
            margin_x = particle_type.sigma[2] * 3
        else:
            radius = particle_type.radius or 5.0
            margin_z = margin_y = margin_x = radius + 2

        # Generate initial positions
        if pop.initial_positions is not None:
            # Use explicit initial positions
            initial_positions = np.array(pop.initial_positions)
            if len(initial_positions) != pop.n_particles:
                msg = (
                    f"Number of initial_positions ({len(initial_positions)}) "
                    f"does not match n_particles ({pop.n_particles})"
                )
                raise ValueError(msg)
        elif pop.initial_region is not None:
            # Use specified region
            region = pop.initial_region
            initial_z = rng.uniform(region[0][0], region[0][1], pop.n_particles)
            initial_y = rng.uniform(region[1][0], region[1][1], pop.n_particles)
            initial_x = rng.uniform(region[2][0], region[2][1], pop.n_particles)
            initial_positions = np.stack([initial_z, initial_y, initial_x], axis=1)
        else:
            # Use default: full volume with margin
            initial_z = rng.uniform(margin_z, nz - margin_z, pop.n_particles)
            initial_y = rng.uniform(margin_y, ny - margin_y, pop.n_particles)
            initial_x = rng.uniform(margin_x, nx - margin_x, pop.n_particles)
            initial_positions = np.stack([initial_z, initial_y, initial_x], axis=1)

        # Generate trajectories
        positions = np.zeros((pop.n_particles, n_frames, 3))

        for i in range(pop.n_particles):
            # Generate 3D FBM trajectory in physical units
            trajectory_um = generate_fbm_trajectory_3d(
                n_steps=n_frames,
                hurst_exponent=hurst,
                diffusion_coefficients=d_tuple,
                dt=dt,
                rng=rng,
            )

            # Convert to pixel coordinates
            trajectory_px = trajectory_um / voxel_array

            # Add initial position
            positions[i, :, :] = initial_positions[i] + trajectory_px

        # Apply boundary conditions
        bounds = (
            (margin_z, nz - margin_z),
            (margin_y, ny - margin_y),
            (margin_x, nx - margin_x),
        )
        positions = apply_boundary_conditions(positions, bounds, mode=config.boundary_mode)

        # Generate particle images
        for t in range(n_frames):
            for p in range(pop.n_particles):
                center = tuple(positions[p, t, :])

                if particle_type.shape == "gaussian":
                    particle = generate_gaussian_particle(
                        (nz, ny, nx),
                        center,
                        particle_type.sigma,
                        particle_type.intensity,
                    )
                else:  # sphere
                    radius = particle_type.radius or 5.0
                    particle = generate_sphere_volume(
                        (nz, ny, nx),
                        center,
                        radius,
                        particle_type.intensity,
                    )

                volumes[t] += particle

        # Store positions and labels
        all_positions.append(positions)
        all_labels.extend([pop_idx] * pop.n_particles)

    # Concatenate all positions
    all_positions_array = np.concatenate(all_positions, axis=0)
    all_labels_array = np.array(all_labels, dtype=np.int64)

    # Add noise
    if config.noise_level > 0:
        volumes += rng.normal(0, config.noise_level, volumes.shape)
        volumes = np.clip(volumes, 0, None)

    return volumes, all_positions_array, all_labels_array
