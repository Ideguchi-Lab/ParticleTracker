"""
Basic 3D Particle Detection Demo
================================

This example demonstrates how to use pt3d for basic 3D particle detection
using trackpy as the detection backend.

Prerequisites:
    1. Run demo_synthetic_data.py first to generate test data
    2. Or provide your own .npy file

Usage:
    python basic_detection.py
"""

from pathlib import Path

import numpy as np

from pt3d.config import DetectionConfig
from pt3d.detect import detect_batch, detect_frame


def demo_single_frame_detection() -> None:
    """Demonstrate detection in a single 3D volume."""
    print("=" * 60)
    print("Single Frame Detection Demo")
    print("=" * 60)

    # Load or generate data
    data_path = Path("./output/synthetic/multi_particles_3d.npy")
    if data_path.exists():
        volume = np.load(data_path)
        print(f"Loaded data from: {data_path}")
    else:
        print("Generating synthetic data...")
        from pt3d.synth import generate_multiple_particles

        centers = [
            (8.0, 16.0, 16.0),
            (16.0, 32.0, 32.0),
            (24.0, 48.0, 48.0),
        ]
        volume = generate_multiple_particles(
            shape=(32, 64, 64),
            centers=centers,
            sigma=(2.0, 3.0, 3.0),
        )
        rng = np.random.default_rng(42)
        volume += rng.normal(0, 0.05, volume.shape)
        volume = np.clip(volume, 0, None)

    print(f"Volume shape: {volume.shape}")

    # Configure detection
    config = DetectionConfig(
        diameter=(5, 9, 9),  # (dz, dy, dx) - must be odd
        minmass=0.1,
        preprocess=True,
    )
    print(f"\nDetection config:")
    print(f"  diameter: {config.diameter}")
    print(f"  minmass: {config.minmass}")

    # Run detection
    detections = detect_frame(volume, config)

    print(f"\nDetected {len(detections)} particles")
    print("\nDetection results:")
    print(detections[["z", "y", "x", "mass"]].to_string())


def demo_batch_detection() -> None:
    """Demonstrate detection across multiple frames."""
    print("\n" + "=" * 60)
    print("Batch Detection Demo (4D Time Series)")
    print("=" * 60)

    # Load or generate 4D data
    data_path = Path("./output/synthetic/moving_particles_4d.npy")
    if data_path.exists():
        frames = np.load(data_path)
        print(f"Loaded data from: {data_path}")
    else:
        print("Generating synthetic 4D data...")
        from pt3d.synth import generate_moving_particles

        frames, _ = generate_moving_particles(
            shape=(10, 32, 64, 64),
            n_particles=5,
            particle_sigma=(2.0, 3.0, 3.0),
            velocity_range=(0.5, 2.0),
            seed=42,
        )

    print(f"Data shape: {frames.shape} (t, z, y, x)")

    # Configure detection
    config = DetectionConfig(
        diameter=(5, 9, 9),
        minmass=0.1,
    )

    # Detect in all frames
    print("\nRunning detection on all frames...")
    detections = detect_batch(frames, config)

    print(f"\nTotal detections: {len(detections)}")
    print(f"Frames with detections: {detections['frame'].nunique()}")
    print(f"Average detections per frame: {len(detections) / detections['frame'].nunique():.1f}")

    # Show summary per frame
    print("\nDetections per frame:")
    frame_counts = detections.groupby("frame").size()
    for frame, count in frame_counts.items():
        print(f"  Frame {frame}: {count} particles")

    # Detect in subset
    print("\nRunning detection on frames 2-5 only...")
    detections_subset = detect_batch(frames, config, frame_range=(2, 5))
    print(f"Detections in subset: {len(detections_subset)}")


def main() -> None:
    demo_single_frame_detection()
    demo_batch_detection()


if __name__ == "__main__":
    main()
