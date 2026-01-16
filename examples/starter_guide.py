# %%
import warnings
from pathlib import Path

# Suppress napari internal warnings during 3D thumbnail generation
warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

from pt3d.synth import generate_moving_particles
from pt3d.config import (
    VoxelSize,
    DetectionConfig,
    TrackingConfig,
    PostprocessConfig,
)
from pt3d.detect import detect_batch

import napari
from pt3d.napari.layers import get_napari_scale, to_napari_points, to_napari_tracks
from pt3d.track import link_detections
from pt3d.postprocess import compute_track_stats, filter_stubs

# %%

frames, ground_truth = generate_moving_particles(
    shape=(20, 32, 64, 64),  # 20 frames, 32x64x64 volume
    n_particles=10,
    particle_sigma=(2.0, 3.0, 3.0),  # Particle size in pixels
    velocity_range=(0.5, 2.0),  # Pixels per frame
    seed=42,
)
print(f"Generated data shape: {frames.shape}")

# %%
# Physical voxel dimensions (CHECK YOUR MICROSCOPE METADATA!)
voxel_size = VoxelSize(
    z_um=0.4,  # Z spacing in micrometers
    y_um=0.2,  # Y pixel size in micrometers
    x_um=0.2,  # X pixel size in micrometers
)
print(f"Anisotropy ratio (z/x): {voxel_size.anisotropy_ratio}")

# Detection configuration
detect_config = DetectionConfig(
    diameter=(5, 9, 9),  # Particle size (dz, dy, dx) - MUST BE ODD
    minmass=1.2,  # Minimum brightness (start low, increase later)
)

# Tracking configuration
track_config = TrackingConfig(
    search_range_um=2.0,  # Maximum displacement per frame in µm
    memory=2,  # Allow particles to disappear for 2 frames
)

# Postprocessing configuration
postprocess_config = PostprocessConfig(
    min_track_length=5,  # Remove tracks shorter than 5 frames
)

# %%
# Run detection
detections = detect_batch(frames, detect_config)

# View results
print(f"Total detections: {len(detections)}")
print(f"Frames with detections: {detections['frame'].nunique()}")
print(f"Detections per frame: {len(detections) / detections['frame'].nunique():.1f}")

# Preview detection DataFrame
print("\nDetection columns:", detections.columns.tolist())
print(detections.head())

# %%
viewer = napari.Viewer()

# Get scale for 4D data (t, z, y, x)
scale = get_napari_scale(voxel_size, include_time=True)

# Add image data
viewer.add_image(frames, name="Volume", scale=scale)

# Add detections as points
# to_napari_points returns (frame, z, y, x) coordinates for 4D
points = to_napari_points(detections, include_frame=True)
viewer.add_points(
    points,
    name="Detections",
    size=5,
    face_color="red",
    scale=scale,
)

napari.run()

# %%
# Link detections
tracks = link_detections(detections, track_config, voxel_size)

# View results
print(f"Total tracks: {tracks['particle'].nunique()}")
print(f"\nTrack columns:", tracks.columns.tolist())
print(tracks.head())

# %%

# Filter short tracks
filtered_tracks = filter_stubs(tracks, postprocess_config.min_track_length)
print(f"Tracks after filtering: {filtered_tracks['particle'].nunique()}")

# Compute statistics
stats = compute_track_stats(filtered_tracks, voxel_size)
print("\nTrack statistics:")
print(stats)

# Summary
print(f"\n--- Summary ---")
print(f"Mean track length: {stats['length'].mean():.1f} frames")
print(f"Mean velocity: {stats['mean_velocity_um'].mean():.2f} µm/frame")

# %%

from pt3d.napari import Track3DVisualizationWidget

viewer = napari.Viewer()

# Get scale for 4D data (t, z, y, x)
scale = get_napari_scale(voxel_size, include_time=True)

# Add image
viewer.add_image(frames, name="Volume", scale=scale)

# Add detections as points
points = to_napari_points(detections, include_frame=True)
viewer.add_points(
    points,
    name="Detections",
    size=5,
    face_color="yellow",
    scale=scale,
)

# Add tracks
# to_napari_tracks returns (track_id, frame, z, y, x) format
track_data = to_napari_tracks(filtered_tracks)
viewer.add_tracks(track_data, name="Tracks", scale=scale)

# Add 3D visualization widget
widget_3d = Track3DVisualizationWidget(viewer)
widget_3d.set_data(
    image=frames,
    tracks=filtered_tracks,
    voxel_size=voxel_size,
)
viewer.window.add_dock_widget(widget_3d, name="3D Visualization")

# Enable 3D mode
widget_3d.enter_3d_mode()

napari.run()

# %%
output_dir = Path("./output/tracking_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Save as Parquet (recommended for large datasets)
filtered_tracks.to_parquet(output_dir / "tracks.parquet")
stats.to_parquet(output_dir / "track_stats.parquet")

# Or save as CSV
filtered_tracks.to_csv(output_dir / "tracks.csv", index=False)
stats.to_csv(output_dir / "track_stats.csv", index=False)

print(f"Results saved to: {output_dir}")
