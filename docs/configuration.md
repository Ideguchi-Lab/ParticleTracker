# Configuration Reference

This document provides a comprehensive guide to all configuration parameters in pt3d,
with practical advice on tuning them for your data.

## Table of Contents

- [Overview](#overview)
- [Physical Parameters](#physical-parameters)
- [Detection Parameters](#detection-parameters)
- [Tracking Parameters](#tracking-parameters)
- [Postprocessing Parameters](#postprocessing-parameters)
- [Export Parameters](#export-parameters)
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

## Detection Parameters

### DetectionConfig

Parameters for particle detection using [trackpy](https://soft-matter.github.io/trackpy/).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diameter` | tuple[int, int, int] | **Required** | Feature diameter (dz, dy, dx) |
| `minmass` | float | 0.0 | Minimum integrated brightness |
| `threshold` | float \| None | None | Noise floor threshold |
| `separation` | tuple[int, int, int] \| None | None | Minimum separation between features |
| `invert` | bool | False | Invert for dark particles on light background |
| `preprocess` | bool | True | Use trackpy's bandpass preprocessing |

### diameter (CRITICAL)

The most important detection parameter. Specifies the expected particle size
in pixels as `(dz, dy, dx)`.

**Requirements:**
- All values **must be odd integers** (3, 5, 7, 9, ...)
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
| `adaptive_stop` | float \| None | None | Adaptive search stop threshold |
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

- `adaptive_stop`: Stop adaptive search when subnetwork contains this many particles
- `adaptive_step`: Reduce search_range by this factor (e.g., 0.9 = 90%)

**When to use:**
- Very dense particle fields (>100 particles/frame)
- Tracking is slow due to combinatorial explosion

```python
# For dense scenarios
track_config = TrackingConfig(
    search_range_um=2.0,
    memory=2,
    adaptive_stop=10.0,
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

Minimum number of frames a track must span to be kept. Removes short "stub"
tracks that are likely noise.

**Typical values:**
- 5-10 for general analysis
- 10-20 for diffusion coefficient estimation
- 2 (default) to keep all tracks

```python
postprocess = PostprocessConfig(min_track_length=5)
```

### max_velocity_um

Maximum allowed velocity in µm/frame. Track points exceeding this velocity
are flagged as outliers.

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

## Diffusion Analysis Parameters

For analyzing Brownian motion and estimating diffusion coefficients.

### DiffusionAnalysisConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_track_length` | int | 10 | Minimum frames for analysis |
| `dt` | float | 1.0 | Time step between frames (seconds) |
| `msd` | MSDConfig | (default) | MSD computation settings |
| `interpolate_gaps` | bool | False | Interpolate missing frames |

### MSDConfig

Mean Squared Displacement computation settings.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_lag_fraction` | float | 0.5 | Maximum lag as fraction of track length |
| `fit_range_fraction` | tuple | (0.1, 0.5) | Fitting range for power-law fit |

**Note:** `fit_range_fraction` must satisfy `0 <= start < end <= 1`.

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
   viewer = napari.Viewer()
   viewer.add_image(data)
   viewer.add_points(detections[['z', 'y', 'x']].values)
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
  adaptive_stop: 10.0
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
