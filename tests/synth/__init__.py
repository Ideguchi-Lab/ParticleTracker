"""Synthetic data generation for testing.

This module re-exports from pt3d.synth for convenience.
"""

from pt3d.synth import (
    generate_gaussian_particle,
    generate_moving_particles,
    generate_sphere_volume,
)

__all__ = [
    "generate_sphere_volume",
    "generate_gaussian_particle",
    "generate_moving_particles",
]
