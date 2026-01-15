"""Configuration models for Brownian motion simulation.

This module provides Pydantic models for configuring synthetic particle
generation with various diffusion behaviors including normal Brownian motion,
subdiffusion, and superdiffusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass

from pt3d.config import VoxelSize


class ParticleTypeConfig(BaseModel):
    """Configuration for particle appearance.

    Attributes
    ----------
    shape : Literal["gaussian", "sphere"]
        Particle shape type. "gaussian" for PSF-like particles,
        "sphere" for solid spherical particles.
    sigma : tuple[float, float, float]
        Gaussian sigma (sz, sy, sx) in pixels. Used when shape="gaussian".
    radius : float | None
        Sphere radius in pixels. Used when shape="sphere".
    intensity : float
        Peak intensity of the particle.
    """

    model_config = ConfigDict(extra="forbid")

    shape: Literal["gaussian", "sphere"] = "gaussian"
    sigma: tuple[float, float, float] = (2.0, 3.0, 3.0)
    radius: float | None = None
    intensity: float = 1.0


class DiffusionConfig(BaseModel):
    """Configuration for diffusion properties.

    Attributes
    ----------
    diffusion_coefficient : float | tuple[float, float, float]
        Diffusion coefficient D in um^2/s. Can be isotropic (float) or
        anisotropic (Dz, Dy, Dx).
    hurst_exponent : float
        Hurst exponent H controlling diffusion type:
        - H = 0.5: Normal Brownian motion (MSD ~ t)
        - H < 0.5: Subdiffusion (MSD ~ t^(2H), antipersistent)
        - H > 0.5: Superdiffusion (MSD ~ t^(2H), persistent)
    dt : float
        Time step between frames in seconds.
    """

    model_config = ConfigDict(extra="forbid")

    diffusion_coefficient: float | tuple[float, float, float]
    hurst_exponent: float = Field(default=0.5, gt=0.0, lt=1.0)
    dt: float = Field(default=1.0, gt=0.0)


class ParticlePopulationConfig(BaseModel):
    """Configuration for a population of particles.

    Attributes
    ----------
    n_particles : int
        Number of particles in this population.
    particle_type : ParticleTypeConfig
        Configuration for particle appearance.
    diffusion : DiffusionConfig
        Configuration for diffusion properties.
    initial_positions : list[tuple[float, float, float]] | None
        Explicit initial positions (z, y, x) in pixels. If None,
        positions are randomly generated within the volume.
    initial_region : tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
        Region for random initial position generation as
        ((z_min, z_max), (y_min, y_max), (x_min, x_max)) in pixels.
        If None, uses full volume with margin.
    """

    model_config = ConfigDict(extra="forbid")

    n_particles: int = Field(gt=0)
    particle_type: ParticleTypeConfig = Field(default_factory=ParticleTypeConfig)
    diffusion: DiffusionConfig
    initial_positions: list[tuple[float, float, float]] | None = None
    initial_region: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None


class BrownianSimulationConfig(BaseModel):
    """Configuration for Brownian motion simulation.

    This is the main configuration class that combines all settings for
    generating synthetic particle data with various diffusion behaviors.

    Attributes
    ----------
    shape : tuple[int, int, int, int]
        Volume shape (t, z, y, x) in pixels/frames.
    voxel_size : VoxelSize
        Physical voxel dimensions for coordinate conversion.
    populations : list[ParticlePopulationConfig]
        List of particle population configurations. Multiple populations
        allow mixing different diffusion types in the same simulation.
    boundary_mode : Literal["reflective", "periodic", "absorbing"]
        Boundary condition type.
    noise_level : float
        Gaussian noise standard deviation to add to volumes.
    seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> from pt3d.config import VoxelSize
    >>> config = BrownianSimulationConfig(
    ...     shape=(100, 32, 64, 64),
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
    """

    model_config = ConfigDict(extra="forbid")

    shape: tuple[int, int, int, int]
    voxel_size: VoxelSize
    populations: list[ParticlePopulationConfig]
    boundary_mode: Literal["reflective", "periodic", "absorbing"] = "reflective"
    noise_level: float = Field(default=0.1, ge=0.0)
    seed: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> BrownianSimulationConfig:
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        BrownianSimulationConfig
            Loaded configuration instance.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValidationError
            If the YAML content does not match the expected schema.
        """
        import yaml

        with Path(path).open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file.

        Parameters
        ----------
        path : str | Path
            Path to save the YAML configuration file.
        """
        import yaml

        # Convert to dict and ensure tuples become lists for YAML compatibility
        data = self.model_dump(mode="json")

        with Path(path).open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
