# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0]

First stable public release. The release date will be recorded when published.

### Added

- Streaming detection and pipeline APIs for processing image volumes one frame at a time
- Diffusion analysis with gap-aware MSD, generalized diffusion coefficients,
  per-particle exponents, and matplotlib visualization
- Software citation metadata in `CITATION.cff`, with a commented paper citation template
- Interactive 3D visualization widget (`Track3DVisualizationWidget`)
  - 3D display mode toggle for napari viewer
  - Volume rendering settings (MIP, attenuated MIP, translucent, ISO)
  - Contrast, gamma, and opacity controls with percentile-based auto-adjustment
  - Track display customization (color by Track ID/Time/Length/Velocity)
  - Detection points visualization settings
  - Camera presets (XY, XZ, YZ, Isometric views)
  - Time navigation with slider and playback controls
- 3D configuration models (`Volume3DConfig`, `Track3DConfig`, `Points3DConfig`)
- 3D layer helper functions (`configure_3d_image_layer`, `configure_3d_tracks_layer`, `configure_3d_points_layer`)
- Camera preset definitions (`CAMERA_PRESETS`, `get_camera_preset`)
- Starter guide example with 3D visualization (`examples/starter_guide.py`)
- User guide documentation (`docs/user_guide.md`)
  - Step-by-step tutorial for real data analysis
  - Working with different file formats
  - Troubleshooting guide for common issues
- Configuration reference documentation (`docs/configuration.md`)
  - Parameter descriptions for core configuration classes
  - Parameter tuning workflow guide
  - Common scenarios with recommended settings
- Enhanced README with documentation links and examples table
- Brownian motion simulation module (`pt3d.synth.brownian`)
  - Fractional Brownian Motion (FBM) trajectory generation
  - Support for normal diffusion (H=0.5), subdiffusion (H<0.5), and superdiffusion (H>0.5)
  - Anisotropic diffusion coefficients
  - Multiple particle populations with different diffusion properties
  - Boundary conditions: reflective, periodic, absorbing
  - MSD computation and diffusion exponent fitting
- Pydantic configuration models for Brownian simulation (`BrownianSimulationConfig`)
- YAML configuration support for simulation parameters
- Demo script for Brownian motion (`examples/demo_brownian_motion.py`)
- YAML configuration runner script (`examples/run_from_yaml.py`)
  - Run tracking pipeline directly from YAML configuration file
  - Optional input path and output directory overrides
- Support for specifying particle diameter in micrometers (`diameter_um`)
  - Alternative to pixel-based `diameter` parameter
  - Automatically converted to pixels using `voxel_size`
  - Enables consistent physical-unit configuration

### Changed

- Changed the project license from Apache-2.0 to MIT, with copyright held by Ideguchi-lab
- Updated package and runtime versions to 1.0.0 and marked the package as stable

### Removed

- Private ilabvis submodule and its dedicated visualization demo; the core
  tracking workflow and napari examples do not require this dependency

### Fixed

- Aligned documentation and docstrings with current input handling, adaptive
  search units, detection schemas, observed track lengths, and unused GUI settings
- Documented exact MSD fitting indices, the slope-one plotting reference,
  generalized diffusion units, and per-frame clipping for the absorbing option
- Replaced the original design draft with a reference to the implemented API
- Corrected explanatory unit labels in the diffusion demos
- Velocity calculation now properly divides by frame difference, correctly handling
  gaps when particles temporarily disappear (memory parameter)
- Clarified physical units: `um`, `um/frame`, and generalized `um^2/s^alpha`
- Fixed `voxel_size=None` documentation in `generate_brownian_particles`
- Fixed `HistogramConfig` docstring (`log_scale_d` not `log_scale_D`)
- Fixed `tracks_visualization_properties` docstring to include all color_by options
- Fixed `normalize_axis_order` docstring example (`tzyx` not `xyzc`)
- Removed duplicate `tests/synth/generators.py` file

## [0.1.0] - 2025-12-25

### Added

- Initial release of pt3d library
- 3D particle detection using trackpy
- Frame-to-frame tracking with coordinate scaling for anisotropy
- Support for ndarray, zarr, and TIFF input formats
- napari plugin for interactive visualization
- Configuration management with Pydantic
- Export to Parquet/CSV with run metadata
- Synthetic data generation for testing
