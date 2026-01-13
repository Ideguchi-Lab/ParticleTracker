#!/usr/bin/env python3
"""Brownian Motion Analysis Demo.

This script demonstrates how to analyze diffusion properties
from particle tracking results.

Usage:
    python demo_brownian_analysis.py

This will:
1. Generate synthetic particle data with known diffusion
2. Run tracking pipeline
3. Analyze diffusion (MSD, D, alpha)
4. Create visualization plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from pt3d import run_pipeline
from pt3d.analysis import (
    DiffusionAnalysisConfig,
    analyze_diffusion,
    create_analysis_report,
    plot_diffusion_histograms,
    plot_msd_loglog,
    plot_tracks_3d,
)
from pt3d.config import (
    DetectionConfig,
    InputConfig,
    PipelineConfig,
    PostprocessConfig,
    TrackingConfig,
    VoxelSize,
)
from pt3d.synth import generate_brownian_particles


def main():
    """Run the Brownian motion analysis demo."""
    output_dir = Path("./output/brownian_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define physical parameters
    voxel_size = VoxelSize(z_um=1.0, y_um=0.1, x_um=0.1)
    dt = 0.1  # 100ms per frame
    d_true = 0.5  # um^2/s

    print("=" * 60)
    print("Brownian Motion Analysis Demo")
    print("=" * 60)
    print("\nPhysical parameters:")
    print(f"  Voxel size: z={voxel_size.z_um}, y={voxel_size.y_um}, x={voxel_size.x_um} um")
    print(f"  Time step: {dt} s")
    print(f"  True diffusion coefficient: {d_true} um^2/s")

    # Generate synthetic data
    print("\n1. Generating synthetic Brownian motion data...")
    volumes, ground_truth = generate_brownian_particles(
        shape=(100, 32, 64, 64),  # 100 frames
        n_particles=15,
        diffusion_coefficient=d_true,
        dt=dt,
        hurst_exponent=0.5,  # Normal diffusion
        voxel_size=voxel_size,
        intensity=1.0,
        particle_sigma=(2.0, 3.0, 3.0),
        noise_level=0.1,
        boundary_mode="reflective",
        seed=42,
    )
    print(f"   Generated volume shape: {volumes.shape}")
    print(f"   Ground truth positions shape: {ground_truth.shape}")

    # Run tracking pipeline
    print("\n2. Running tracking pipeline...")
    pipeline_config = PipelineConfig(
        input=InputConfig(voxel_size=voxel_size),
        detection=DetectionConfig(
            diameter=(5, 9, 9),
            minmass=0.1,
            threshold=None,
        ),
        tracking=TrackingConfig(
            search_range_um=2.0,
            memory=2,
        ),
        postprocess=PostprocessConfig(
            min_track_length=5,
        ),
    )

    result = run_pipeline(volumes, pipeline_config)
    print(f"   Detected particles: {result.n_detections}")
    print(f"   Linked tracks: {result.n_tracks}")

    # Analyze diffusion
    print("\n3. Analyzing diffusion properties...")
    analysis_config = DiffusionAnalysisConfig(
        dt=dt,
        min_track_length=10,
        interpolate_gaps=False,
    )
    analysis = analyze_diffusion(result, analysis_config)

    # Print summary
    print("\n" + "=" * 60)
    print("Diffusion Analysis Summary")
    print("=" * 60)
    summary = analysis.summary()
    print(f"  Particles analyzed: {summary['n_particles']}")
    print(f"  Mean D: {summary['D_mean']:.4f} +/- {summary['D_std']:.4f} um^2/s")
    print(f"  Median D: {summary['D_median']:.4f} um^2/s")
    print(f"  Mean alpha: {summary['alpha_mean']:.2f} +/- {summary['alpha_std']:.2f}")
    print(f"  Median alpha: {summary['alpha_median']:.2f}")

    # Compare with ground truth
    print("\n  Comparison with ground truth:")
    print(f"    True D = {d_true} um^2/s")
    print(f"    Estimated D = {summary['D_mean']:.4f} um^2/s")
    error_pct = abs(summary["D_mean"] - d_true) / d_true * 100
    print(f"    Error: {error_pct:.1f}%")

    # Save per-particle results
    results_path = output_dir / "diffusion_results.csv"
    analysis.particle_results.to_csv(results_path, index=False)
    print(f"\n   Per-particle results saved to: {results_path}")

    # Create visualizations
    print("\n4. Generating visualizations...")

    # 1. 3D tracks colored by D
    print("   - 3D trajectory plot...")
    fig, _ax = plot_tracks_3d(analysis)
    fig.savefig(output_dir / "tracks_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Histograms
    print("   - Diffusion coefficient and exponent histograms...")
    fig, _ = plot_diffusion_histograms(analysis)
    fig.savefig(output_dir / "diffusion_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. MSD plot
    print("   - MSD log-log plot...")
    fig, _ax = plot_msd_loglog(analysis)
    fig.savefig(output_dir / "msd_loglog.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Full report
    print("   - Full analysis report...")
    fig = create_analysis_report(analysis, str(output_dir / "analysis_report.png"))
    plt.close(fig)

    print(f"\n5. All results saved to: {output_dir.absolute()}")
    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"   - {f.name}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
