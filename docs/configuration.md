# Configuration Reference

This document provides a comprehensive guide to all configuration parameters in pt3d,
with practical advice on tuning them for your data.

## Table of Contents

- [Overview](#overview)
- [Physical Parameters](#physical-parameters)
- [Input Parameters](#input-parameters)
- [Detection Parameters](#detection-parameters)
- [Tracking Parameters](#tracking-parameters)
- [Postprocessing Parameters](#postprocessing-parameters)
- [Export Parameters](#export-parameters)
- [Napari Parameters](#napari-parameters)
- [3D Visualization Parameters](#3d-visualization-parameters)
- [Diffusion Analysis Parameters](#diffusion-analysis-parameters)
- [Parameter Tuning Workflow](#parameter-tuning-workflow)
- [Common Scenarios](#common-scenarios)

---

## Overview

pt3d uses [Pydantic](https://docs.pydantic.dev/) models for configuration management.
You can configure the pipeline either through Python code or YAML configuration files.

### Using Python

```python
from pt3d.config import (
    DetectionConfig,
    TrackingConfig,
    VoxelSize,
    PipelineConfig,
)

voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
detect_config = DetectionConfig(diameter=(5, 9, 9), minmass=100.0)
```

### Using YAML

See [examples/sample_config.yaml](../examples/sample_config.yaml) for a complete example.

```yaml
input:
  voxel_size:
    z_um: 0.8
    y_um: 0.2
    x_um: 0.2

detection:
  diameter: [5, 9, 9]
  minmass: 100.0
```

---

## Physical Parameters

### VoxelSize

Physical voxel dimensions in micrometers. **This is critical for accurate tracking.**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `z_um` | float | Yes | Z spacing in micrometers |
| `y_um` | float | Yes | Y spacing in micrometers |
| `x_um` | float | Yes | X spacing in micrometers |

#### Why VoxelSize Matters

In microscopy, Z resolution is typically worse than XY resolution due to optical
sectioning. This creates **anisotropic voxels**. For example:

- Confocal: Z = 0.8 µm, XY = 0.2 µm (4:1 ratio)
- Light-sheet: Z = 1.0 µm, XY = 0.1 µm (10:1 ratio)

pt3d uses voxel size to:
1. Convert pixel coordinates to physical (µm) coordinates
2. Scale coordinates for isotropic distance calculations during tracking
3. Report velocities and displacements in physical units

#### How to Determine Voxel Size

Check your microscope metadata:
- **ImageJ/Fiji**: Image > Properties
- **Microscope software**: Look for "voxel size" or "pixel spacing"
- **OME-TIFF**: Embedded in metadata

#### Anisotropy Ratio

The `anisotropy_ratio` property returns `z_um / x_um`:

```python
voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
print(voxel_size.anisotropy_ratio)  # 4.0
```

A ratio > 1 indicates Z is coarser than XY (typical in most microscopy).

---

## Input Parameters

### InputConfig

Configuration for input data loading and interpretation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | Path \| None | None | Path to input file (zarr, TIFF, or npy) |
| `axis_order` | str | "tzyx" | Axis order for file inputs; ignored for ndarray pipeline inputs |
| `dtype` | str \| None | None | Stored metadata; no automatic conversion |
| `voxel_size` | VoxelSize | **Required** | Physical voxel dimensions |

### axis_order

For file inputs to `run_pipeline`, specifies how data dimensions are organized.
Default is `"tzyx"` (time, z, y, x). For a 3D file, specify `"zyx"`.
An ndarray passed directly to `run_pipeline` must already be in `tzyx` or `zyx`
order; `axis_order` is ignored for this input type. Normalize other array layouts
explicitly with `pt3d.io.load_array(data, axis_order="zyxt")` first.

**Supported axes:**
- `t` - Time dimension
- `z` - Z (depth) dimension
- `y` - Y (height) dimension
- `x` - X (width) dimension

**Examples:**
```python
# Standard 4D time series
input_config = InputConfig(axis_order="tzyx", voxel_size=voxel_size)

# Single 3D volume (no time)
input_config = InputConfig(axis_order="zyx", voxel_size=voxel_size)

# Data with different axis order
input_config = InputConfig(axis_order="zyxt", voxel_size=voxel_size)
```

### dtype

This field is accepted and saved with the configuration, but neither pipeline
applies it. Convert data explicitly when a particular dtype is needed:

```python
from pt3d.io import load_data

frames = load_data("data.zarr", axis_order="tzyx").astype("float32", copy=False)
# Pass frames to run_pipeline(frames, pipeline_config).
```

---

## Detection Parameters

### DetectionConfig

Parameters for particle detection using [trackpy](https://soft-matter.github.io/trackpy/).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diameter` | tuple[int, int, int] \| None | None | Pixel diameter (dz, dy, dx); specify this or diameter_um |
| `diameter_um` | float \| None | None | Isotropic physical diameter in µm; alternative to diameter |
| `minmass` | float | 0.0 | Minimum integrated brightness |
| `threshold` | float \| None | None | Noise floor threshold |
| `separation` | tuple[int, int, int] \| None | None | Minimum separation between features |
| `invert` | bool | False | Invert for dark particles on light background |
| `preprocess` | bool | True | Use trackpy's bandpass preprocessing |

### diameter (CRITICAL)

Exactly one of `diameter` and `diameter_um` must be supplied. With `diameter_um`,
pass `voxel_size` to the detection function; the pipeline does this automatically.
Conversion divides by each voxel dimension, rounds to an integer (minimum 1),
then increments even integers to the next odd integer. This is not always the
nearest odd integer to the original value.

```python
detect_config = DetectionConfig(diameter_um=1.8)
detections = detect_batch(frames, detect_config, voxel_size=voxel_size)
```

The most important detection parameter. Specifies the expected particle size
in pixels as `(dz, dy, dx)`.

**Requirements:**
- All values **must be positive odd integers** (1, 3, 5, 7, 9, ...)
- Should approximate the full width at half maximum (FWHM) of particles

**How to estimate diameter:**

1. Open your data in napari or ImageJ
2. Measure the apparent size of a typical particle in each dimension
3. Round to the nearest odd integer

**Example for anisotropic data:**

If particles appear 5 pixels wide in Z and 9 pixels wide in XY:
```python
diameter = (5, 9, 9)
```

**Effects of incorrect diameter:**

| Setting | Result |
|---------|--------|
| Too small | Detects noise, multiple detections per particle |
| Too large | Misses small particles, poor localization |
| Even number | Validation error |

### minmass

Minimum integrated brightness to accept a particle. This filters out dim
detections that are likely noise.

**How to tune:**

1. Start with `minmass=0` to see all detections
2. Visualize detections in napari
3. Increase `minmass` until noise detections disappear
4. Typical values: 10-1000 depending on signal intensity

```python
# Start permissive
detect_config = DetectionConfig(diameter=(5, 9, 9), minmass=0)

# After viewing results, increase to filter noise
detect_config = DetectionConfig(diameter=(5, 9, 9), minmass=100.0)
```

### threshold

Clips the bandpass filter result below this value (noise floor).
Set to `None` for automatic threshold selection.

**When to use:**
- Noisy data with variable background
- When `minmass` alone isn't sufficient

### separation

Minimum distance between detected particles in pixels `(dz, dy, dx)`.
Default is `diameter + 1` if not specified.

**When to adjust:**
- Decrease for densely packed particles
- Increase to prevent false double-detections

### invert

Set to `True` when tracking **dark particles on a light background**.
Default is `False` (bright particles on dark background).

### preprocess

Whether to apply trackpy's bandpass filter before detection.
Default is `True` (recommended).

**When to disable:**
- Your data is already preprocessed
- You need exact intensity values from raw data

---

## Tracking Parameters

### TrackingConfig

Parameters for linking detections across frames.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `search_range_um` | float | **Required** | Maximum displacement per frame in µm |
| `memory` | int | 0 | Frames to remember lost particles |
| `adaptive_stop` | float \| None | None | Stop distance in scaled coordinate units |
| `adaptive_step` | float \| None | None | Adaptive search step factor |

### search_range_um (CRITICAL)

The maximum distance a particle can move between consecutive frames, in
**micrometers**.

**This is the most important tracking parameter.**

**How to estimate:**

1. Estimate typical particle velocity in µm/frame
2. Set `search_range_um` to 1.5-3× the typical displacement
3. Should be larger than typical movement but smaller than nearest-neighbor distance

**Example:**
```python
# Particles typically move ~1 µm/frame
# Set search range slightly larger
track_config = TrackingConfig(search_range_um=2.0)
```

**Effects of incorrect search_range_um:**

| Setting | Result |
|---------|--------|
| Too small | Tracks break (fragmented trajectories) |
| Too large | Wrong particles linked (track swapping) |

**Tips:**
- For Brownian motion: `search_range_um ≈ 3 × sqrt(2 × D × dt)`
  where D is diffusion coefficient and dt is frame interval
- For directed motion: `search_range_um ≈ 1.5 × max_velocity × dt`

### memory

Number of frames a particle can "disappear" and still be linked to the same track.
Useful for handling temporary detection failures.

**When to use:**
- Particles occasionally go out of focus
- Detection misses some frames due to noise
- Particles blink (e.g., fluorophore blinking)

```python
# Allow particles to disappear for up to 2 frames
track_config = TrackingConfig(search_range_um=2.0, memory=2)
```

**Trade-off:** Higher memory increases computation time and may link unrelated particles.

### adaptive_stop and adaptive_step

Advanced parameters for dense particle fields. Both must be specified together
or both set to `None`.

- `adaptive_stop`: Give up on an oversized subnet when the reduced search range
  is at or below this distance. This is a distance threshold, not a particle count.
  Unlike `search_range_um`, it is passed unchanged to trackpy in scaled units;
  one scaled unit equals `min(voxel_size.as_tuple())` micrometers.
- `adaptive_step`: Reduce search_range by this factor (e.g., 0.9 = 90%)

**When to use:**
- Very dense particle fields (>100 particles/frame)
- Tracking is slow due to combinatorial explosion

```python
# For dense scenarios
stop_um = 0.1  # Desired lower search-distance threshold in micrometers
track_config = TrackingConfig(
    search_range_um=2.0,
    memory=2,
    adaptive_stop=stop_um / min(voxel_size.as_tuple()),
    adaptive_step=0.9,
)
```

---

## Postprocessing Parameters

### PostprocessConfig

Parameters for filtering and refining tracks after linking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_track_length` | int | 2 | Minimum frames for a valid track |
| `max_velocity_um` | float \| None | None | Maximum velocity for outlier removal |

### min_track_length

Minimum number of observed points per track. Missing frames do not count:
a track detected only at frames 0 and 10 has length 2, not 11.
This removes short "stub" tracks that are likely noise.

**Typical values:**
- 5-10 for general analysis
- 10-20 for diffusion coefficient estimation
- 2 (default) to remove single-point tracks
- 1 to keep all tracks at this filtering step

```python
postprocess = PostprocessConfig(min_track_length=5)
```

### max_velocity_um

Maximum allowed velocity in µm/frame. Track points exceeding this velocity
are removed. Velocity is displacement divided by the frame difference, including
gaps. The first point of each track is retained; short-track filtering follows.

**When to use:**
- Remove tracking errors (sudden jumps)
- Filter physically unrealistic movements

```python
# Remove points moving faster than 10 µm/frame
postprocess = PostprocessConfig(max_velocity_um=10.0)
```

---

## Export Parameters

### ExportConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | Path | **Required** | Output directory |
| `format` | "parquet" \| "csv" | "parquet" | Output format |
| `overwrite` | bool | False | Allow overwriting files |

**Parquet vs CSV:**
- **Parquet**: Binary format, smaller files, faster I/O, preserves data types
- **CSV**: Human-readable, larger files, compatible with Excel

---

## Napari Parameters

### NapariConfig

Stored preferences for napari visualization. The current pipeline and widgets
do not read these fields to configure layers. Set names and slice data explicitly
in the code that creates the layers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_name` | str | "Volume" | Stored image name; not applied |
| `points_name` | str | "Detections" | Stored points name; not applied |
| `tracks_name` | str | "Tracks" | Stored tracks name; not applied |
| `frame_range` | tuple[int, int] \| None | None | Stored range; not applied |

### frame_range

Setting this field does not select display frames. For an image-only preview,
slice the array when creating the layer:

```python
viewer.add_image(frames[0:100], name="Preview")
```

For matching points and tracks, filter their frame rows as well; subtract the
slice start from frame coordinates if the preview begins after frame zero.

### Visualization Helpers

pt3d provides helper functions for napari visualization in `pt3d.napari.layers`:

```python
from pt3d.napari.layers import (
    get_napari_scale,      # Get scale tuple for layers
    to_napari_points,      # Convert detections to points format
    to_napari_tracks,      # Convert tracks to tracks format
)

# Get scale for 4D data (includes time dimension)
scale = get_napari_scale(voxel_size, include_time=True)

# Convert data for napari
points = to_napari_points(detections, include_frame=True)
tracks = to_napari_tracks(tracks_df)
```

---

## 3D Visualization Parameters

Configuration classes for the `Track3DVisualizationWidget`.

### Volume3DConfig

Configuration for 3D volume rendering.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rendering_mode` | Literal | "mip" | Volume rendering mode |
| `contrast_percentile_low` | float | 1.0 | Lower percentile for contrast (0-100) |
| `contrast_percentile_high` | float | 99.0 | Upper percentile for contrast (0-100) |
| `gamma` | float | 1.0 | Gamma correction value |
| `opacity` | float | 0.5 | Layer opacity (0-1) |
| `iso_threshold` | float | 0.5 | ISO threshold relative to contrast limits (0-1) |
| `colormap` | str | "gray" | Colormap for volume rendering |

#### rendering_mode

Available volume rendering modes:

| Mode | Description | Best For |
|------|-------------|----------|
| `"mip"` | Maximum Intensity Projection | Bright particles, quick overview |
| `"attenuated_mip"` | MIP with depth attenuation | Better depth perception |
| `"translucent"` | Semi-transparent rendering | Viewing internal structures |
| `"iso"` | Isosurface rendering | Cell boundaries, surfaces |

```python
from pt3d.config import Volume3DConfig

# For viewing bright particles with depth
vol_config = Volume3DConfig(
    rendering_mode="attenuated_mip",
    contrast_percentile_low=1.0,
    contrast_percentile_high=99.5,
    gamma=0.9,
    opacity=0.6,
)
```

### Track3DConfig

Configuration for 3D track visualization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `colormap` | str | "turbo" | Track colormap |
| `color_by` | Literal | "track_id" | Property for coloring |
| `tail_length` | int | 10 | Trail length in frames |
| `show_current_position` | bool | True | Reserved; no rendering effect |

#### color_by

Available coloring options:

| Value | Description |
|-------|-------------|
| `"track_id"` | Each track gets a unique color |
| `"time"` | Color changes along trajectory |
| `"length"` | Longer tracks are brighter |
| `"velocity"` | Faster particles are brighter |

```python
from pt3d.config import Track3DConfig

# Color by velocity with long trails
track_config = Track3DConfig(
    colormap="viridis",
    color_by="velocity",
    tail_length=20,
    show_current_position=True,  # Reserved; currently has no effect
)
```

### Points3DConfig

Configuration for 3D detection points visualization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | float | 5.0 | Point size in display units |
| `face_color` | str | "yellow" | Point face color |
| `opacity` | float | 0.8 | Point opacity (0-1) |
| `show_current_frame_only` | bool | True | Reserved; no rendering effect |

```python
from pt3d.config import Points3DConfig

# Large red points
points_config = Points3DConfig(
    size=10.0,
    face_color="red",
    opacity=0.9,
    show_current_frame_only=True,  # Reserved; currently has no effect
)
```

### Camera Presets

Predefined camera angles accessible via `get_camera_preset()`:

| Preset | Angles | Description |
|--------|--------|-------------|
| `"xy"` | (0, 0, 90) | Top-down view (looking along Z) |
| `"xz"` | (0, -90, 90) | Front view (looking along Y) |
| `"yz"` | (90, 0, 0) | Side view (looking along X) |
| `"isometric"` | (30, 45, 0) | 3D isometric perspective |

```python
from pt3d.napari.layers import get_camera_preset, CAMERA_PRESETS

# Get preset info
preset = get_camera_preset("isometric")
print(preset)  # {'angles': (30, 45, 0), 'name': 'Isometric', ...}

# Apply to viewer
viewer.camera.angles = preset["angles"]
```

---

## Diffusion Analysis Parameters

For analyzing Brownian motion and estimating diffusion coefficients.

### DiffusionAnalysisConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_track_length` | int | 10 | Minimum observed points for analysis (gaps excluded) |
| `dt` | float | 1.0 | Time step between frames (seconds) |
| `msd` | MSDConfig | (default) | MSD computation settings |
| `interpolate_gaps` | bool | False | Interpolate missing frames |

### MSDConfig

Mean Squared Displacement computation settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_lag_fraction` | float | 0.5 | Per-particle maximum lag as fraction of frame span, including gaps |
| `fit_range_fraction` | tuple | (0.1, 0.5) | Fit-index fractions of len(msd), including lag zero |

**Note:** `fit_range_fraction` must satisfy `0 <= start < end <= 1`.

For a trajectory spanning `n_frames` (including gaps), the per-particle maximum
lag is `min(int(n_frames * max_lag_fraction), n_frames - 1)`. The MSD array includes
lag zero. Fit indices are calculated as:

```python
start = max(1, int(len(msd) * start_fraction))
end = min(len(msd), max(start + 2, int(len(msd) * end_fraction)))
# Fit msd[start:end]; end is exclusive. Exclude nonpositive and NaN MSD values.
```

For 70 frames, `max_lag_fraction=0.3` and `fit_range_fraction=(0.1, 0.8)`
give 22 MSD values and fit indices `[2:17]` (lags 2 through 16).
The ensemble MSD separately uses half the shortest frame span and does not use
`max_lag_fraction`. Each available per-particle MSD has equal weight at a lag.

The fitted model is `MSD = 6 * D * t**alpha`, with time in seconds and positions
in micrometers. `D` is a generalized coefficient in `µm²/s^alpha`.

### MSDPlotConfig.show_fit

Despite its name, `show_fit=True` draws a slope-one reference line
`6 * mean_d * t`, with the legacy legend `α=1 fit`. It does not fit the ensemble
MSD or use estimated exponents. For anomalous diffusion, this line is only a
visual reference; `show_fit=False` hides it. To draw a particle's fitted model,
use that particle's `6 * D * time_lags**alpha`.

---

## Parameter Tuning Workflow

Follow this systematic approach for tuning parameters on new data:

### Step 1: Get Voxel Size Right

1. Check your microscope metadata for pixel/voxel spacing
2. Create a `VoxelSize` object with correct values
3. Verify the anisotropy ratio makes sense for your microscope

### Step 2: Tune Detection (diameter first)

1. Start with estimated `diameter` based on visual inspection
2. Set `minmass=0` initially
3. Run detection on a subset of frames
4. Visualize in napari:
   ```python
   import napari
   from pt3d.napari.layers import get_napari_scale, to_napari_points

   viewer = napari.Viewer()
   scale = get_napari_scale(voxel_size, include_time=True)  # For 4D data
   viewer.add_image(data, scale=scale)
   points = to_napari_points(detections, include_frame=True)
   viewer.add_points(points, scale=scale, size=3)
   napari.run()
   ```
5. Adjust `diameter`:
   - Too many detections per particle → increase diameter
   - Missing particles → decrease diameter
6. Increase `minmass` to filter noise:
   - Increase until false positives disappear
   - Don't go too high (losing real particles)

### Step 3: Validate Detection Quality

Before tracking, ensure detection is working well:
- Count detections per frame (should be stable)
- Check for systematic over/under-detection
- Verify detected positions match actual particles

### Step 4: Tune Tracking (search_range_um first)

1. Estimate typical particle displacement per frame
2. Set `search_range_um` to 1.5-2× this value
3. Run tracking on a subset
4. Visualize tracks in napari
5. Adjust:
   - Broken tracks → increase `search_range_um`
   - Track swapping → decrease `search_range_um`
6. Add `memory` if particles occasionally disappear

### Step 5: Tune Postprocessing

1. Set `min_track_length` based on analysis needs
2. Add `max_velocity_um` if seeing unrealistic jumps

### Step 6: Full Pipeline Run

1. Run on full dataset
2. Validate results
3. Export and analyze

---

## Common Scenarios

### Scenario 1: Bright Particles, Low Noise

Typical settings:
```yaml
detection:
  diameter: [5, 9, 9]
  minmass: 100.0
  preprocess: true

tracking:
  search_range_um: 2.0
  memory: 1

postprocess:
  min_track_length: 5
```

### Scenario 2: Dim Particles, High Noise

Start permissive, then filter:
```yaml
detection:
  diameter: [5, 9, 9]
  minmass: 10.0  # Low threshold
  threshold: 1.0  # Add noise floor
  preprocess: true

tracking:
  search_range_um: 2.0
  memory: 2  # Handle detection gaps

postprocess:
  min_track_length: 10  # Longer tracks more reliable
  max_velocity_um: 5.0  # Remove outliers
```

### Scenario 3: Dense Particle Field

Use adaptive search:
```yaml
detection:
  diameter: [5, 9, 9]
  minmass: 100.0
  separation: [3, 5, 5]  # Reduce separation

tracking:
  search_range_um: 1.0  # Smaller range
  memory: 1
  adaptive_stop: 0.5  # Scaled units; equals 0.1 um if the smallest voxel is 0.2 um
  adaptive_step: 0.9

postprocess:
  min_track_length: 5
```

### Scenario 4: Fast-Moving Particles

Increase search range:
```yaml
detection:
  diameter: [5, 9, 9]
  minmass: 100.0

tracking:
  search_range_um: 5.0  # Larger range
  memory: 0  # Less memory (fast motion)

postprocess:
  min_track_length: 3  # Shorter tracks OK
  max_velocity_um: 20.0  # Higher velocity limit
```

### Scenario 5: Highly Anisotropic Data (Z >> XY)

Adjust diameter for anisotropy:
```yaml
input:
  voxel_size:
    z_um: 2.0  # Large Z spacing
    y_um: 0.1
    x_um: 0.1

detection:
  # Smaller Z diameter due to worse Z resolution
  diameter: [3, 15, 15]
  minmass: 100.0
```

---

## Validation Checklist

Before running your final analysis, verify:

- [ ] Voxel size matches microscope metadata
- [ ] Detection count per frame is reasonable and stable
- [ ] Detected positions overlay correctly on particles in napari
- [ ] Track count is reasonable (not too many fragmented tracks)
- [ ] Track lengths are appropriate for your analysis
- [ ] No obvious track swapping in visualizations
- [ ] Velocities are physically reasonable

---

## See Also

- [User Guide](user_guide.md) - Step-by-step tutorial
- [examples/sample_config.yaml](../examples/sample_config.yaml) - Complete YAML example
- [trackpy documentation](https://soft-matter.github.io/trackpy/) - Underlying detection/tracking library
