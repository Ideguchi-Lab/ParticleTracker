# pt3d - 3D+t Particle Tracking Library

A Python library for 3D+t particle detection and tracking in microscopy data, using [trackpy](https://soft-matter.github.io/trackpy/) as the processing engine and [napari](https://napari.org/) for visualization.

## Features

- **3D Particle Detection**: Detect particles in 3D volumetric data using trackpy
- **Frame-to-Frame Tracking**: Link detections across time frames to form trajectories
- **Anisotropy Correction**: Handle different z vs xy resolution with coordinate scaling
- **napari Integration**: Visualize results interactively with napari plugin
- **Reproducibility**: Save configurations and run metadata for reproducible analysis

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# With napari support
uv pip install -e ".[napari]"

# With development dependencies
uv pip install -e ".[all]"
```

## Quick Start

```python
import numpy as np
from pt3d.config import DetectionConfig, TrackingConfig, VoxelSize
from pt3d.detect import detect_batch
from pt3d.track import link_detections

# Load your 4D data (t, z, y, x)
data = np.load("your_data.npy")

# Configure detection
voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
detect_config = DetectionConfig(
    diameter=(5, 9, 9),  # (dz, dy, dx) - must be odd
    minmass=100.0,
)

# Detect particles
detections = detect_batch(data, detect_config)

# Configure tracking
track_config = TrackingConfig(
    search_range_um=2.0,  # Maximum displacement per frame in µm
    memory=2,
)

# Link detections into tracks
tracks = link_detections(detections, track_config, voxel_size)

print(f"Found {tracks['particle'].nunique()} tracks")
```

## napari Plugin

Launch napari and use the pt3d widget from the Plugins menu:

```python
import napari
viewer = napari.Viewer()
# Plugins > 3D Particle Tracker
napari.run()
```

## Documentation

See [docs/spec.md](docs/spec.md) for the full specification.

## License

Apache License 2.0
