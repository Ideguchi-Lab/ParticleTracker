# pt3d - 3D+t Particle Tracking Library

A Python library for 3D+t particle detection and tracking in microscopy data, using [trackpy](https://soft-matter.github.io/trackpy/) as the processing engine and [napari](https://napari.org/) for visualization.

## Features

- **3D Particle Detection**: Detect particles in 3D volumetric data using trackpy
- **Frame-to-Frame Tracking**: Link detections across time frames to form trajectories
- **Anisotropy Correction**: Handle different z vs xy resolution with coordinate scaling
- **napari Integration**: Visualize results interactively with napari plugin
- **3D Visualization**: Interactive 3D volume rendering with track overlay
- **Reproducibility**: Save configurations and run metadata for reproducible analysis

## Installation

This repository can be installed directly; no Git submodules are required.

```bash
# Using uv (recommended)
uv venv  # Once, from the repository root
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

Launch napari and use the pt3d widgets from the Plugins menu:

```python
import napari
viewer = napari.Viewer()
# Plugins > 3D Particle Tracker      (detection & tracking)
# Plugins > 3D Track Visualization   (3D visualization)
napari.run()
```

### 3D Visualization

Using `data`, `tracks`, and `voxel_size` from the Quick Start above:

```python
import napari
from pt3d.napari import Track3DVisualizationWidget
from pt3d.napari.layers import get_napari_scale, to_napari_tracks

viewer = napari.Viewer()
scale = get_napari_scale(voxel_size)
viewer.add_image(data, name="Volume", scale=scale)
viewer.add_tracks(to_napari_tracks(tracks), name="Tracks", scale=scale)

# Add 3D widget
widget_3d = Track3DVisualizationWidget(viewer)
widget_3d.set_data(image=data, tracks=tracks, voxel_size=voxel_size)
viewer.window.add_dock_widget(widget_3d)
widget_3d.enter_3d_mode()

napari.run()
```

## Documentation

- **[User Guide](docs/user_guide.md)** - Step-by-step tutorial for real data analysis
- **[Configuration Reference](docs/configuration.md)** - Complete parameter tuning guide
- **[Implementation Reference](docs/spec.md)** - Current API, processing behavior, and limitations (Japanese)

## Examples

See the [examples/](examples/) directory for working code:

| Example | Description |
|---------|-------------|
| [starter_guide.py](examples/starter_guide.py) | Complete tutorial with 3D visualization |
| [tracking_pipeline.py](examples/tracking_pipeline.py) | Complete detection → tracking workflow |
| [basic_detection.py](examples/basic_detection.py) | Single frame and batch detection |
| [napari_interactive.py](examples/napari_interactive.py) | Interactive visualization with napari |
| [demo_brownian_motion.py](examples/demo_brownian_motion.py) | Brownian motion simulation |
| [demo_brownian_analysis.py](examples/demo_brownian_analysis.py) | Diffusion coefficient estimation |
| [sample_config.yaml](examples/sample_config.yaml) | YAML configuration template |

## License

[MIT License](LICENSE) — Copyright (c) 2026 Ideguchi-lab
