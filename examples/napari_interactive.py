"""
Interactive napari Demo
=======================

This example demonstrates how to use pt3d with napari for
interactive parameter tuning and visualization.

Prerequisites:
    1. Install napari: pip install "napari[all]"
    2. Run demo_synthetic_data.py first to generate test data

Usage:
    python napari_interactive.py
"""

from pathlib import Path

import numpy as np

def main() -> None:
    # Import napari (will fail if not installed)
    try:
        import napari
    except ImportError:
        print("napari is not installed. Install with:")
        print("  pip install 'napari[all]'")
        return

    from pt3d.config import DetectionConfig, TrackingConfig, VoxelSize
    from pt3d.detect import detect_batch
    from pt3d.napari.layers import get_napari_scale, to_napari_points, to_napari_tracks
    from pt3d.postprocess import filter_stubs
    from pt3d.track import link_detections

    # Load or generate data
    data_path = Path("./output/synthetic/moving_particles_4d.npy")
    if data_path.exists():
        volumes = np.load(data_path)
        print(f"Loaded data from: {data_path}")
    else:
        print("Generating synthetic 4D data...")
        from pt3d.synth import generate_moving_particles

        volumes, _ = generate_moving_particles(
            shape=(20, 32, 64, 64),
            n_particles=10,
            particle_sigma=(2.0, 3.0, 3.0),
            velocity_range=(0.5, 2.0),
            seed=42,
        )

    print(f"Data shape: {volumes.shape} (t, z, y, x)")

    # Define voxel size (anisotropic)
    voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)

    # Run detection
    print("Running detection...")
    detect_config = DetectionConfig(
        diameter=(5, 9, 9),
        minmass=0.1,
    )
    detections = detect_batch(volumes, detect_config)
    print(f"Detected {len(detections)} particles")

    # Run tracking
    print("Running tracking...")
    track_config = TrackingConfig(
        search_range_um=2.0,
        memory=2,
    )
    tracks = link_detections(detections, track_config, voxel_size)
    tracks = filter_stubs(tracks, min_length=5)
    print(f"Found {tracks['particle'].nunique()} tracks")

    # Launch napari
    print("\nLaunching napari viewer...")
    viewer = napari.Viewer()

    # Add image layer with correct scale
    scale = get_napari_scale(voxel_size, include_time=True)
    viewer.add_image(
        volumes,
        name="Particles",
        scale=scale,
        colormap="gray",
    )

    # Add detection points
    if len(detections) > 0:
        points = to_napari_points(detections)
        viewer.add_points(
            points,
            name="Detections",
            size=5,
            face_color="yellow",
            scale=scale,
        )

    # Add tracks
    if len(tracks) > 0:
        tracks_data = to_napari_tracks(tracks)
        viewer.add_tracks(
            tracks_data,
            name="Tracks",
            scale=scale,
        )

    print("\nnapari viewer is open.")
    print("Use the slider to navigate through time frames.")
    print("Toggle layers on/off to see detections and tracks.")
    print("\nTo use the pt3d plugin widget:")
    print("  Plugins > 3D Particle Tracker")

    napari.run()


if __name__ == "__main__":
    main()
