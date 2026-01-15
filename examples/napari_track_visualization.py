"""
Track Visualization Demo with napari
=====================================

This example demonstrates the track visualization features of the pt3d
napari plugin, including color-coded tracks and XY max projection views.

Prerequisites:
    1. Install napari: pip install "napari[all]"
    2. Run demo_synthetic_data.py first to generate test data
       (or synthetic data will be generated automatically)

Usage:
    python napari_track_visualization.py
"""

from pathlib import Path

import numpy as np


def main() -> None:
    try:
        import napari
    except ImportError:
        print("napari is not installed. Install with:")
        print("  pip install 'napari[all]'")
        return

    from pt3d.config import DetectionConfig, TrackingConfig, VoxelSize
    from pt3d.detect import detect_batch
    from pt3d.napari.layers import (
        compute_xy_max_projection,
        get_napari_scale,
        to_napari_tracks,
        tracks_visualization_properties,
    )
    from pt3d.postprocess import compute_track_stats, filter_stubs
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

    print(f"Data shape: {volumes.shape} (T, Z, Y, X)")

    # Define physical dimensions
    voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
    scale = get_napari_scale(voxel_size, include_time=True)

    # Detection
    print("\nRunning detection...")
    detect_config = DetectionConfig(diameter=(5, 9, 9), minmass=0.1)
    detections = detect_batch(volumes, detect_config)
    print(f"Detected {len(detections)} particles across {detections['frame'].nunique()} frames")

    # Tracking
    print("\nRunning tracking...")
    track_config = TrackingConfig(search_range_um=2.0, memory=2)
    tracks = link_detections(detections, track_config, voxel_size)
    tracks = filter_stubs(tracks, min_length=5)
    print(f"Linked into {tracks['particle'].nunique()} tracks")

    # Compute statistics
    track_stats = compute_track_stats(tracks, voxel_size)
    print("\nTrack Statistics:")
    print(f"  Total tracks: {len(track_stats)}")
    print(f"  Mean length: {track_stats['length'].mean():.1f} frames")
    print(f"  Max length: {track_stats['length'].max()} frames")
    print(f"  Mean velocity: {track_stats['mean_velocity_um'].mean():.2f} um/frame")

    # Launch napari viewer
    print("\nLaunching napari viewer...")
    viewer = napari.Viewer()

    # Add original 4D image
    viewer.add_image(volumes, name="Particles", scale=scale, colormap="gray")

    # Add tracks with color coding by track ID
    if len(tracks) > 0:
        tracks_data = to_napari_tracks(tracks)
        properties, color_by = tracks_visualization_properties(tracks, track_stats, color_by="track_id")

        viewer.add_tracks(
            tracks_data,
            name="Tracks",
            scale=scale,
            properties=properties,
            color_by=color_by,
            colormap="turbo",
        )

    # Create XY Max Projection view
    print("\nCreating XY max projection...")
    proj_volumes, proj_tracks = compute_xy_max_projection(volumes, tracks)
    proj_scale = (1.0, voxel_size.y_um, voxel_size.x_um)  # T, Y, X

    # Add max projection image (initially hidden)
    viewer.add_image(
        proj_volumes,
        name="XY Max Projection",
        scale=proj_scale,
        colormap="gray",
        visible=False,
    )

    # Add projected tracks (initially hidden)
    if proj_tracks is not None and len(proj_tracks) > 0:
        proj_tracks_data = to_napari_tracks(proj_tracks)
        proj_properties, proj_color_by = tracks_visualization_properties(proj_tracks, track_stats, color_by="track_id")

        viewer.add_tracks(
            proj_tracks_data,
            name="Tracks (Max Proj)",
            scale=proj_scale,
            properties=proj_properties,
            color_by=proj_color_by,
            colormap="turbo",
            visible=False,
        )

    print("\n" + "=" * 60)
    print("napari viewer is open.")
    print("=" * 60)
    print("\nVisualization Tips:")
    print("  - 'Particles' layer: Original 4D data with Z slicing")
    print("  - 'Tracks' layer: Color-coded particle trajectories")
    print("  - Use the time slider at the bottom to navigate frames")
    print("\nTo switch to Max Projection view:")
    print("  1. Hide 'Particles' and 'Tracks' layers")
    print("  2. Show 'XY Max Projection' and 'Tracks (Max Proj)' layers")
    print("\nLayer Controls:")
    print("  - Click layer name to select")
    print("  - Use 'colormap' dropdown to change track colors")
    print("  - Use 'color by' dropdown to change coloring property")
    print("=" * 60)

    napari.run()


if __name__ == "__main__":
    main()
