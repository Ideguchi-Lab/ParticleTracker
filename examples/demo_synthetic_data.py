"""
Synthetic Data Generation Demo
==============================

This script generates synthetic 3D and 4D particle data for testing
and demonstration purposes. The generated data includes:

1. A 3D volume with multiple spherical particles
2. A 3D volume with Gaussian particles
3. A 4D time series with moving particles

All data is saved as .npy files for easy loading.

Usage:
    python demo_synthetic_data.py

Output files are saved to ./output/synthetic/
"""

from pathlib import Path

import numpy as np

from pt3d.synth import (
    generate_circle_slice,
    generate_gaussian_particle,
    generate_moving_particles,
    generate_multiple_particles,
    generate_sphere_volume,
)


def main() -> None:
    output_dir = Path("./output/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic particle data...")
    print(f"Output directory: {output_dir.absolute()}")

    # 1. Generate 2D circle (for visualization)
    print("\n1. Generating 2D circle slice...")
    circle = generate_circle_slice(
        shape=(64, 64),
        center=(32.0, 32.0),
        radius=10.0,
        intensity=1.0,
    )
    np.save(output_dir / "circle_2d.npy", circle)
    print(f"   Shape: {circle.shape}")
    print(f"   Saved to: circle_2d.npy")

    # 2. Generate 3D sphere
    print("\n2. Generating 3D sphere volume...")
    sphere = generate_sphere_volume(
        shape=(32, 64, 64),
        center=(16.0, 32.0, 32.0),
        radius=8.0,
        intensity=1.0,
    )
    np.save(output_dir / "sphere_3d.npy", sphere)
    print(f"   Shape: {sphere.shape}")
    print(f"   Saved to: sphere_3d.npy")

    # 3. Generate 3D Gaussian particle
    print("\n3. Generating 3D Gaussian particle...")
    gaussian = generate_gaussian_particle(
        shape=(32, 64, 64),
        center=(16.0, 32.0, 32.0),
        sigma=(3.0, 5.0, 5.0),  # Anisotropic (z is smaller)
        intensity=1.0,
    )
    np.save(output_dir / "gaussian_3d.npy", gaussian)
    print(f"   Shape: {gaussian.shape}")
    print(f"   Saved to: gaussian_3d.npy")

    # 4. Generate 3D volume with multiple particles
    print("\n4. Generating 3D volume with multiple particles...")
    centers = [
        (8.0, 16.0, 16.0),
        (16.0, 32.0, 32.0),
        (24.0, 48.0, 48.0),
        (10.0, 48.0, 16.0),
        (22.0, 16.0, 48.0),
    ]
    multi_particles = generate_multiple_particles(
        shape=(32, 64, 64),
        centers=centers,
        sigma=(2.0, 3.0, 3.0),
        intensities=[1.0, 0.8, 1.2, 0.9, 1.1],
    )
    # Add some noise
    rng = np.random.default_rng(42)
    multi_particles += rng.normal(0, 0.05, multi_particles.shape)
    multi_particles = np.clip(multi_particles, 0, None)
    np.save(output_dir / "multi_particles_3d.npy", multi_particles)
    print(f"   Shape: {multi_particles.shape}")
    print(f"   Number of particles: {len(centers)}")
    print(f"   Saved to: multi_particles_3d.npy")

    # Save ground truth positions
    centers_array = np.array(centers)
    np.save(output_dir / "multi_particles_3d_positions.npy", centers_array)
    print(f"   Ground truth positions saved to: multi_particles_3d_positions.npy")

    # 5. Generate 4D time series with moving particles
    print("\n5. Generating 4D time series with moving particles...")
    volumes_4d, positions_4d = generate_moving_particles(
        shape=(20, 32, 64, 64),  # 20 frames, 32x64x64 volume
        n_particles=10,
        particle_sigma=(2.0, 3.0, 3.0),
        velocity_range=(0.5, 2.0),
        intensity=1.0,
        noise_level=0.05,
        seed=42,
    )
    np.save(output_dir / "moving_particles_4d.npy", volumes_4d)
    np.save(output_dir / "moving_particles_4d_positions.npy", positions_4d)
    print(f"   Volume shape: {volumes_4d.shape} (t, z, y, x)")
    print(f"   Number of particles: {positions_4d.shape[0]}")
    print(f"   Number of frames: {positions_4d.shape[1]}")
    print(f"   Saved to: moving_particles_4d.npy")
    print(f"   Ground truth positions saved to: moving_particles_4d_positions.npy")

    # Summary
    print("\n" + "=" * 60)
    print("Summary of generated files:")
    print("=" * 60)
    for npy_file in sorted(output_dir.glob("*.npy")):
        data = np.load(npy_file)
        print(f"  {npy_file.name}: shape={data.shape}, dtype={data.dtype}")

    print(f"\nAll files saved to: {output_dir.absolute()}")
    print("\nTo load these files:")
    print("  import numpy as np")
    print("  data = np.load('output/synthetic/moving_particles_4d.npy')")


if __name__ == "__main__":
    main()
