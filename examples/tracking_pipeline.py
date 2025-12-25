"""
Complete Tracking Pipeline Demo
===============================

This example demonstrates the full pt3d pipeline from loading data
to exporting tracking results.

Prerequisites:
    1. Run demo_synthetic_data.py first to generate test data
    2. Or provide your own .npy file

Usage:
    python tracking_pipeline.py
"""

from pathlib import Path

import numpy as np

from pt3d.config import (
    DetectionConfig,
    ExportConfig,
    InputConfig,
    PipelineConfig,
    PostprocessConfig,
    TrackingConfig,
    VoxelSize,
)
from pt3d.detect import detect_batch
from pt3d.pipeline import run_pipeline
from pt3d.postprocess import compute_track_stats
from pt3d.track import link_detections


def demo_step_by_step() -> None:
    """Demonstrate step-by-step tracking."""
    print("=" * 60)
    print("Step-by-Step Tracking Demo")
    print("=" * 60)

    # Load or generate data
    data_path = Path("./output/synthetic/moving_particles_4d.npy")
    if data_path.exists():
        frames = np.load(data_path)
        print(f"Loaded data from: {data_path}")
    else:
        print("Generating synthetic 4D data...")
        from pt3d.synth import generate_moving_particles

        frames, _ = generate_moving_particles(
            shape=(20, 32, 64, 64),
            n_particles=10,
            particle_sigma=(2.0, 3.0, 3.0),
            velocity_range=(0.5, 2.0),
            seed=42,
        )

    print(f"Data shape: {frames.shape}")

    # Step 1: Detection
    print("\n--- Step 1: Detection ---")
    detect_config = DetectionConfig(
        diameter=(5, 9, 9),
        minmass=0.1,
    )
    detections = detect_batch(frames, detect_config)
    print(f"Detected {len(detections)} particles across {detections['frame'].nunique()} frames")

    # Step 2: Tracking
    print("\n--- Step 2: Tracking ---")
    voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
    track_config = TrackingConfig(
        search_range_um=2.0,  # Maximum displacement per frame in µm
        memory=2,  # Allow gaps of 2 frames
    )
    tracks = link_detections(detections, track_config, voxel_size)
    print(f"Linked into {tracks['particle'].nunique()} tracks")

    # Step 3: Compute statistics
    print("\n--- Step 3: Track Statistics ---")
    stats = compute_track_stats(tracks, voxel_size)
    print(f"Track statistics:")
    print(stats.to_string())

    # Summary
    print(f"\n--- Summary ---")
    print(f"Total detections: {len(detections)}")
    print(f"Total tracks: {len(stats)}")
    print(f"Mean track length: {stats['length'].mean():.1f} frames")
    print(f"Mean velocity: {stats['mean_velocity_um'].mean():.2f} µm/frame")


def demo_pipeline() -> None:
    """Demonstrate the complete pipeline."""
    print("\n" + "=" * 60)
    print("Complete Pipeline Demo")
    print("=" * 60)

    # Load or generate data
    data_path = Path("./output/synthetic/moving_particles_4d.npy")
    if data_path.exists():
        frames = np.load(data_path)
        print(f"Loaded data from: {data_path}")
    else:
        print("Generating synthetic 4D data...")
        from pt3d.synth import generate_moving_particles

        frames, _ = generate_moving_particles(
            shape=(20, 32, 64, 64),
            n_particles=10,
            particle_sigma=(2.0, 3.0, 3.0),
            velocity_range=(0.5, 2.0),
            seed=42,
        )

    # Configure pipeline
    config = PipelineConfig(
        input=InputConfig(
            axis_order="tzyx",
            voxel_size=VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2),
        ),
        detection=DetectionConfig(
            diameter=(5, 9, 9),
            minmass=0.1,
        ),
        tracking=TrackingConfig(
            search_range_um=2.0,
            memory=2,
        ),
        postprocess=PostprocessConfig(
            min_track_length=5,  # Remove tracks shorter than 5 frames
        ),
        export=ExportConfig(
            output_dir=Path("./output/tracking_results"),
            format="parquet",
            overwrite=True,
        ),
    )

    print(f"\nPipeline configuration:")
    print(f"  Voxel size: {config.input.voxel_size.as_tuple()} µm")
    print(f"  Detection diameter: {config.detection.diameter}")
    print(f"  Search range: {config.tracking.search_range_um} µm")
    print(f"  Min track length: {config.postprocess.min_track_length}")

    # Run pipeline
    print("\nRunning pipeline...")
    result = run_pipeline(frames, config)

    # Show results
    print(f"\n--- Pipeline Results ---")
    print(f"Execution time: {result.duration_seconds:.2f} seconds")
    print(f"Total detections: {result.n_detections}")
    print(f"Total tracks: {result.n_tracks}")

    if len(result.track_stats) > 0:
        print(f"\nTrack statistics:")
        print(f"  Mean length: {result.track_stats['length'].mean():.1f} frames")
        print(f"  Max length: {result.track_stats['length'].max()} frames")
        print(f"  Mean velocity: {result.track_stats['mean_velocity_um'].mean():.2f} µm/frame")

    print(f"\nOutput files saved to: {config.export.output_dir}")


def main() -> None:
    demo_step_by_step()
    demo_pipeline()


if __name__ == "__main__":
    main()
