# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
