"""
Run Tracking Pipeline from YAML Configuration
==============================================

This script runs the pt3d tracking pipeline using a YAML configuration file.
Edit the paths below before running.

Usage:
    1. Copy and modify examples/sample_config.yaml for your data
    2. Edit CONFIG_PATH, INPUT_PATH, OUTPUT_DIR below
    3. Run: python examples/run_from_yaml.py
"""

from pathlib import Path

import yaml

from pt3d import PipelineConfig, run_pipeline
from pt3d.config import ExportConfig

# ============================================================
# EDIT THESE PATHS
# ============================================================
CONFIG_PATH = Path("examples/sample_config.yaml")
INPUT_PATH: Path | None = None  # Set to override input.path in config
OUTPUT_DIR: Path | None = None  # Set to override export.output_dir in config
# ============================================================


def main() -> None:
    """Load config from YAML and run the tracking pipeline."""
    # Load config from YAML
    print(f"Loading configuration from: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        config_dict = yaml.safe_load(f)
    config = PipelineConfig.model_validate(config_dict)

    # Override input path if specified
    if INPUT_PATH is not None:
        config.input.path = INPUT_PATH
        print(f"Input path overridden: {INPUT_PATH}")

    # Override output directory if specified
    if OUTPUT_DIR is not None:
        if config.export is None:
            config.export = ExportConfig(output_dir=OUTPUT_DIR)
        else:
            config.export.output_dir = OUTPUT_DIR
        print(f"Output directory overridden: {OUTPUT_DIR}")

    # Validate input path
    if config.input.path is None:
        raise ValueError("Input path must be specified either in config file or via INPUT_PATH variable")

    # Show configuration summary
    print("\n" + "=" * 60)
    print("Pipeline Configuration")
    print("=" * 60)
    print(f"  Input: {config.input.path}")
    print(f"  Voxel size: {config.input.voxel_size.as_tuple()} µm")
    print(f"  Detection diameter: {config.detection.diameter}")
    print(f"  Search range: {config.tracking.search_range_um} µm")
    print(f"  Min track length: {config.postprocess.min_track_length}")
    if config.export:
        print(f"  Output: {config.export.output_dir}")
    print("=" * 60)

    # Run pipeline
    print("\nRunning pipeline...")
    result = run_pipeline(config.input.path, config)

    # Show results
    print("\n" + "=" * 60)
    print("Pipeline Results")
    print("=" * 60)
    print(f"  Execution time: {result.duration_seconds:.2f} seconds")
    print(f"  Total detections: {result.n_detections}")
    print(f"  Total tracks: {result.n_tracks}")

    if len(result.track_stats) > 0:
        print("\n  Track statistics:")
        print(f"    Mean length: {result.track_stats['length'].mean():.1f} frames")
        print(f"    Max length: {result.track_stats['length'].max()} frames")
        print(f"    Mean velocity: {result.track_stats['mean_velocity_um'].mean():.2f} µm/frame")

    if config.export:
        print(f"\n  Output saved to: {config.export.output_dir}")

    print("=" * 60)


if __name__ == "__main__":
    main()
