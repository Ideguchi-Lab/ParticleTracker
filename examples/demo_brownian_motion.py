"""
Brownian Motion Simulation Demo
===============================

This script demonstrates how to generate synthetic particle data
with various diffusion behaviors:
- Normal Brownian motion (H=0.5, α=1)
- Subdiffusion (H<0.5, α<1) - antipersistent FBM
- Superdiffusion (H>0.5, α>1) - persistent FBM

The generalized diffusion coefficient has units um^2/s^(2H).
Boundary handling can change the observed MSD from the unbounded FBM power law.

Output files are saved to ./output/brownian/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pt3d.config import VoxelSize
from pt3d.synth import (
    BrownianSimulationConfig,
    DiffusionConfig,
    ParticlePopulationConfig,
    ParticleTypeConfig,
    compute_msd,
    fit_diffusion_exponent,
    generate_brownian_particles_from_config,
)


def main() -> None:
    output_dir = Path("./output/brownian")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Brownian motion simulation data...")
    print(f"Output directory: {output_dir.absolute()}")

    # ========================================
    # Configuration (edit these parameters)
    # ========================================
    config = BrownianSimulationConfig(
        shape=(100, 32, 64, 64),  # (t, z, y, x)
        voxel_size=VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1),
        populations=[
            # Population 1: Normal diffusion
            ParticlePopulationConfig(
                n_particles=5,
                particle_type=ParticleTypeConfig(
                    shape="gaussian",
                    sigma=(2.0, 3.0, 3.0),
                    intensity=1.0,
                ),
                diffusion=DiffusionConfig(
                    diffusion_coefficient=0.5,  # um^2/s
                    hurst_exponent=0.5,  # Normal diffusion (alpha=1.0)
                    dt=0.1,  # 100ms per frame
                ),
            ),
            # Population 2: Subdiffusion
            ParticlePopulationConfig(
                n_particles=5,
                particle_type=ParticleTypeConfig(
                    shape="gaussian",
                    sigma=(2.0, 3.0, 3.0),
                    intensity=0.8,
                ),
                diffusion=DiffusionConfig(
                    diffusion_coefficient=0.3,
                    hurst_exponent=0.3,  # Subdiffusion (alpha=0.6)
                    dt=0.1,
                ),
            ),
            # Population 3: Superdiffusion
            ParticlePopulationConfig(
                n_particles=3,
                particle_type=ParticleTypeConfig(
                    shape="gaussian",
                    sigma=(2.0, 3.0, 3.0),
                    intensity=1.2,
                ),
                diffusion=DiffusionConfig(
                    diffusion_coefficient=0.4,
                    hurst_exponent=0.7,  # Superdiffusion (alpha=1.4)
                    dt=0.1,
                ),
            ),
        ],
        boundary_mode="reflective",
        noise_level=0.05,
        seed=42,
    )

    # Generate data
    print("\nGenerating particles with different diffusion types...")
    volumes, positions, population_labels = generate_brownian_particles_from_config(config)

    print(f"  Volume shape: {volumes.shape} (t, z, y, x)")
    print(f"  Total particles: {positions.shape[0]}")
    print(f"  Frames: {positions.shape[1]}")

    # Save data
    np.save(output_dir / "brownian_volumes.npy", volumes)
    np.save(output_dir / "brownian_positions.npy", positions)
    np.save(output_dir / "brownian_labels.npy", population_labels)
    config.to_yaml(str(output_dir / "config.yaml"))

    print(f"\nData saved to {output_dir}")

    # Compute and plot MSD for each population
    print("\nComputing MSD for each population...")
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    population_names = [
        "Normal (H=0.5)",
        "Subdiffusion (H=0.3)",
        "Superdiffusion (H=0.7)",
    ]
    colors = ["blue", "green", "red"]

    # MSD plot (log-log scale)
    ax1 = axes[0]
    for pop_idx, pop_config in enumerate(config.populations):
        mask = population_labels == pop_idx
        pop_positions = positions[mask]

        if len(pop_positions) > 0:
            # Convert positions to physical units for MSD calculation
            voxel_array = np.array(
                [
                    config.voxel_size.z_um,
                    config.voxel_size.y_um,
                    config.voxel_size.x_um,
                ]
            )
            pop_positions_um = pop_positions * voxel_array

            msd = compute_msd(pop_positions_um)
            alpha, _ = fit_diffusion_exponent(msd, dt=pop_config.diffusion.dt)

            time_lags = np.arange(len(msd)) * pop_config.diffusion.dt
            ax1.loglog(
                time_lags[1:],
                msd[1:],
                "o-",
                color=colors[pop_idx],
                label=f"{population_names[pop_idx]}: α={alpha:.2f}",
                markersize=3,
            )

    ax1.set_xlabel("Time lag (s)")
    ax1.set_ylabel("MSD (μm²)")
    ax1.legend()
    ax1.set_title("Mean Squared Displacement (log-log)")
    ax1.grid(True, alpha=0.3)

    # Sample trajectories plot
    ax2 = axes[1]
    for pop_idx in range(len(config.populations)):
        mask = population_labels == pop_idx
        pop_positions = positions[mask]

        if len(pop_positions) > 0:
            # Plot first particle from each population
            traj = pop_positions[0]
            # Convert to physical units
            traj_um = traj * np.array(
                [
                    config.voxel_size.z_um,
                    config.voxel_size.y_um,
                    config.voxel_size.x_um,
                ]
            )
            ax2.plot(
                traj_um[:, 2],
                traj_um[:, 1],
                "-",
                color=colors[pop_idx],
                label=population_names[pop_idx],
                alpha=0.7,
            )
            # Mark start and end points
            ax2.scatter(traj_um[0, 2], traj_um[0, 1], color=colors[pop_idx], s=50, marker="o")
            ax2.scatter(traj_um[-1, 2], traj_um[-1, 1], color=colors[pop_idx], s=50, marker="s")

    ax2.set_xlabel("X (μm)")
    ax2.set_ylabel("Y (μm)")
    ax2.legend()
    ax2.set_title("Sample Trajectories (XY projection)")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "msd_comparison.png", dpi=150)
    plt.close()

    print(f"MSD plot saved to {output_dir / 'msd_comparison.png'}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for pop_idx, pop_config in enumerate(config.populations):
        mask = population_labels == pop_idx
        n = np.sum(mask)
        hurst = pop_config.diffusion.hurst_exponent
        d_coeff = pop_config.diffusion.diffusion_coefficient
        print(f"  {population_names[pop_idx]}:")
        print(f"    Particles: {n}")
        print(f"    Hurst exponent (H): {hurst}")
        print(f"    Expected α = 2H: {2 * hurst:.2f}")
        print(f"    Generalized diffusion coefficient (D): {d_coeff} um^2/s^(2H)")

    print("\nDone!")


if __name__ == "__main__":
    main()
