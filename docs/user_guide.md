# User Guide

This guide walks you through using pt3d for 3D particle tracking in microscopy data.
By the end, you'll be able to detect particles, track them across frames, and analyze
the results.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Tutorial: Your First Tracking Analysis](#tutorial-your-first-tracking-analysis)
- [Working with Real Data](#working-with-real-data)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Installation

```bash
# Using uv (recommended)
uv pip install -e .

# With napari support for visualization
uv pip install -e ".[napari]"

# With all optional dependencies
uv pip install -e ".[all]"
```

### Data Requirements

pt3d expects 4D volumetric time-series data:

- **Shape**: `(t, z, y, x)` - time, z-slices, height, width
- **Formats**: NumPy arrays (`.npy`), Zarr, TIFF stacks
- **Content**: Bright particles on dark background (or use `invert=True`)

### Required Information

Before starting, you need:

1. **Voxel size** - Physical pixel dimensions from your microscope (in µm)
2. **Approximate particle size** - How many pixels wide are your particles?
3. **Approximate particle velocity** - How far do particles move per frame?

---

## Tutorial: Your First Tracking Analysis

This tutorial uses synthetic data to demonstrate the full workflow.
The same steps apply to real data.

### Step 1: Generate or Load Data

First, let's create synthetic data to work with:

```python
import numpy as np
from pathlib import Path

# Option A: Generate synthetic data
from pt3d.synth import generate_moving_particles

frames, ground_truth = generate_moving_particles(
    shape=(20, 32, 64, 64),  # 20 frames, 32x64x64 volume
    n_particles=10,
    particle_sigma=(2.0, 3.0, 3.0),  # Particle size in pixels
    velocity_range=(0.5, 2.0),  # Pixels per frame
    seed=42,
)
print(f"Generated data shape: {frames.shape}")

# Option B: Load your own data
# frames = np.load("your_data.npy")
# frames = zarr.open("your_data.zarr")[:]
```

### Step 2: Set Up Configuration

Define the physical parameters and detection settings:

```python
from pt3d.config import (
    VoxelSize,
    DetectionConfig,
    TrackingConfig,
    PostprocessConfig,
)

# Physical voxel dimensions (CHECK YOUR MICROSCOPE METADATA!)
voxel_size = VoxelSize(
    z_um=0.8,  # Z spacing in micrometers
    y_um=0.2,  # Y pixel size in micrometers
    x_um=0.2,  # X pixel size in micrometers
)
print(f"Anisotropy ratio (z/x): {voxel_size.anisotropy_ratio}")

# Detection configuration
detect_config = DetectionConfig(
    diameter=(5, 9, 9),  # Particle size (dz, dy, dx) - MUST BE ODD
    minmass=0.1,  # Minimum brightness (start low, increase later)
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
```

### Step 3: Run Detection

Detect particles in all frames:

```python
from pt3d.detect import detect_batch

# Run detection
detections = detect_batch(frames, detect_config)

# View results
print(f"Total detections: {len(detections)}")
print(f"Frames with detections: {detections['frame'].nunique()}")
print(f"Detections per frame: {len(detections) / detections['frame'].nunique():.1f}")

# Preview detection DataFrame
print("\nDetection columns:", detections.columns.tolist())
print(detections.head())
```

Output columns:
- `frame`: Time frame index
- `z`, `y`, `x`: Particle position in pixels
- `mass`: Integrated brightness
- Additional trackpy columns

### Step 4: Visualize Detections (Optional but Recommended)

Before tracking, verify that detection is working correctly:

```python
import napari
from pt3d.napari.layers import get_napari_scale, to_napari_points

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
```

**What to check:**
- Do detected points overlay on actual particles?
- Are there false positives (detections on noise)?
- Are there false negatives (missed particles)?

If detection isn't working well, see [Troubleshooting](#troubleshooting).

### Step 5: Link Detections into Tracks

Connect detections across frames:

```python
from pt3d.track import link_detections

# Link detections
tracks = link_detections(detections, track_config, voxel_size)

# View results
print(f"Total tracks: {tracks['particle'].nunique()}")
print(f"\nTrack columns:", tracks.columns.tolist())
print(tracks.head())
```

The `tracks` DataFrame includes:
- `particle`: Unique track ID
- `frame`, `z`, `y`, `x`: Position at each time point
- All original detection columns

### Step 6: Compute Track Statistics

Analyze track properties:

```python
from pt3d.postprocess import compute_track_stats, filter_stubs

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
```

### Step 7: Visualize Tracks

View the tracking results:

```python
import napari
from pt3d.napari.layers import get_napari_scale, to_napari_tracks

viewer = napari.Viewer()

# Get scale for 4D data (t, z, y, x)
scale = get_napari_scale(voxel_size, include_time=True)

# Add image
viewer.add_image(frames, name="Volume", scale=scale)

# Add tracks
# to_napari_tracks returns (track_id, frame, z, y, x) format
track_data = to_napari_tracks(filtered_tracks)
viewer.add_tracks(track_data, name="Tracks", scale=scale)

napari.run()
```

### Step 8: Export Results

Save your results:

```python
from pathlib import Path

output_dir = Path("./output/tracking_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Save as Parquet (recommended for large datasets)
filtered_tracks.to_parquet(output_dir / "tracks.parquet")
stats.to_parquet(output_dir / "track_stats.parquet")

# Or save as CSV
filtered_tracks.to_csv(output_dir / "tracks.csv", index=False)
stats.to_csv(output_dir / "track_stats.csv", index=False)

print(f"Results saved to: {output_dir}")
```

### Complete Example

Here's the full pipeline in one script:

```python
"""Complete tracking pipeline example."""
import numpy as np
from pt3d.config import (
    VoxelSize,
    DetectionConfig,
    TrackingConfig,
    PostprocessConfig,
)
from pt3d.detect import detect_batch
from pt3d.track import link_detections
from pt3d.postprocess import compute_track_stats, filter_stubs

# 1. Load data
from pt3d.synth import generate_moving_particles
frames, _ = generate_moving_particles(
    shape=(20, 32, 64, 64),
    n_particles=10,
    particle_sigma=(2.0, 3.0, 3.0),
    velocity_range=(0.5, 2.0),
    seed=42,
)

# 2. Configure
voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
detect_config = DetectionConfig(diameter=(5, 9, 9), minmass=0.1)
track_config = TrackingConfig(search_range_um=2.0, memory=2)

# 3. Detect
detections = detect_batch(frames, detect_config)
print(f"Detected {len(detections)} particles")

# 4. Track
tracks = link_detections(detections, track_config, voxel_size)
print(f"Linked into {tracks['particle'].nunique()} tracks")

# 5. Postprocess
filtered = filter_stubs(tracks, min_length=5)
stats = compute_track_stats(filtered, voxel_size)
print(f"Final: {len(stats)} tracks, mean length {stats['length'].mean():.1f}")
```

---

## Working with Real Data

### Loading Different File Formats

```python
import numpy as np
import zarr
from tifffile import imread

# NumPy array
data = np.load("data.npy")

# Zarr
data = zarr.open("data.zarr")[:]

# TIFF stack
data = imread("data.tif")

# Ensure shape is (t, z, y, x)
print(f"Data shape: {data.shape}")
```

### Determining Voxel Size

Your microscope metadata should contain pixel/voxel spacing:

```python
# Example: Extract from OME-TIFF
from tifffile import TiffFile

with TiffFile("data.ome.tif") as tif:
    # Check metadata
    metadata = tif.ome_metadata
    # Parse for pixel sizes...
```

Common voxel sizes:
- Confocal: Z = 0.5-1.0 µm, XY = 0.1-0.2 µm
- Light-sheet: Z = 0.5-2.0 µm, XY = 0.1-0.5 µm
- Widefield: Z = 0.3-0.5 µm, XY = 0.05-0.1 µm

### Handling Large Datasets

For datasets that don't fit in memory:

```python
import zarr

# Open lazily
data = zarr.open("large_data.zarr")

# Process in chunks
chunk_size = 100  # frames
for start in range(0, data.shape[0], chunk_size):
    end = min(start + chunk_size, data.shape[0])
    chunk = data[start:end]

    # Process chunk...
    detections_chunk = detect_batch(chunk, detect_config)
    detections_chunk['frame'] += start  # Adjust frame indices
```

### Preprocessing Tips

Before detection, consider:

```python
import numpy as np
from scipy import ndimage

# Background subtraction
background = ndimage.uniform_filter(data, size=(1, 1, 50, 50))
data_corrected = data - background

# Noise reduction
from scipy.ndimage import gaussian_filter
data_smooth = gaussian_filter(data, sigma=(0, 0.5, 1, 1))

# Note: pt3d's preprocess=True applies bandpass filtering automatically
```

---

## Troubleshooting

### Detection Issues

#### Too Many Detections (False Positives)

**Symptoms:** Detecting noise, multiple detections per particle

**Solutions:**
1. Increase `minmass`:
   ```python
   detect_config = DetectionConfig(diameter=(5, 9, 9), minmass=500.0)
   ```
2. Add `threshold` to set noise floor:
   ```python
   detect_config = DetectionConfig(diameter=(5, 9, 9), minmass=100.0, threshold=10.0)
   ```
3. Increase `diameter` if detecting sub-particle features

#### Too Few Detections (False Negatives)

**Symptoms:** Missing real particles

**Solutions:**
1. Decrease `minmass`:
   ```python
   detect_config = DetectionConfig(diameter=(5, 9, 9), minmass=10.0)
   ```
2. Decrease `diameter` if particles are smaller than expected
3. Check if `invert=True` is needed (dark particles on light background)
4. Ensure `preprocess=True` for proper bandpass filtering

#### Poor Localization

**Symptoms:** Detected positions don't center on particles

**Solutions:**
1. Adjust `diameter` to match actual particle size
2. Ensure diameter values are odd integers
3. Try disabling preprocessing if already preprocessed:
   ```python
   detect_config = DetectionConfig(diameter=(5, 9, 9), preprocess=False)
   ```

### Tracking Issues

#### Fragmented Tracks (Track Breaking)

**Symptoms:** Single particles have multiple short tracks

**Solutions:**
1. Increase `search_range_um`:
   ```python
   track_config = TrackingConfig(search_range_um=5.0)  # was 2.0
   ```
2. Increase `memory` for detection gaps:
   ```python
   track_config = TrackingConfig(search_range_um=2.0, memory=3)  # was 0
   ```
3. Improve detection consistency (see above)

#### Track Swapping (Wrong Links)

**Symptoms:** Tracks jump between different particles

**Solutions:**
1. Decrease `search_range_um`:
   ```python
   track_config = TrackingConfig(search_range_um=1.0)  # was 2.0
   ```
2. Increase `separation` in detection to prevent overlapping detections
3. Use adaptive search for dense fields:
   ```python
   track_config = TrackingConfig(
       search_range_um=1.5,
       adaptive_stop=10.0,
       adaptive_step=0.9,
   )
   ```

#### Slow Tracking

**Symptoms:** Tracking takes very long or crashes

**Solutions:**
1. Enable adaptive search:
   ```python
   track_config = TrackingConfig(
       search_range_um=2.0,
       adaptive_stop=5.0,
       adaptive_step=0.95,
   )
   ```
2. Reduce `search_range_um`
3. Process fewer particles (increase `minmass`)
4. Process data in smaller time chunks

### Common Error Messages

#### "Diameter must be odd"

```python
# Wrong
detect_config = DetectionConfig(diameter=(4, 8, 8))  # Even numbers

# Correct
detect_config = DetectionConfig(diameter=(5, 9, 9))  # Odd numbers
```

#### "adaptive_stop and adaptive_step must be used together"

```python
# Wrong - only one specified
track_config = TrackingConfig(search_range_um=2.0, adaptive_stop=10.0)

# Correct - both specified
track_config = TrackingConfig(
    search_range_um=2.0,
    adaptive_stop=10.0,
    adaptive_step=0.9,
)

# Also correct - neither specified
track_config = TrackingConfig(search_range_um=2.0)
```

#### "No particles detected"

Possible causes:
- `minmass` too high
- `diameter` doesn't match particle size
- Need `invert=True` for dark particles
- Data is empty or all zeros

### Getting Help

If you encounter issues not covered here:

1. Check the [Configuration Reference](configuration.md) for parameter details
2. Review the [examples/](../examples/) directory for working code
3. Open an issue on GitHub with:
   - Your configuration
   - Sample data (if possible)
   - Error message or unexpected behavior

---

## Next Steps

- **Advanced analysis**: See [examples/demo_brownian_analysis.py](../examples/demo_brownian_analysis.py)
  for diffusion coefficient estimation
- **Parameter tuning**: See [Configuration Reference](configuration.md) for detailed
  parameter descriptions
- **napari plugin**: Use `Plugins > 3D Particle Tracker` for interactive analysis
- **Batch processing**: See [examples/tracking_pipeline.py](../examples/tracking_pipeline.py)
  for complete pipeline example
