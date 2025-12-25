"""Export module for saving results and metadata."""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from pt3d.exceptions import ProcessingError

if TYPE_CHECKING:
    from pt3d.config import ExportConfig, PipelineConfig

logger = logging.getLogger(__name__)


def export_dataframe(
    df: pd.DataFrame,
    path: Path,
    format: str = "parquet",
    overwrite: bool = False,
) -> Path:
    """Export a DataFrame to file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export
    path : Path
        Output path (without extension)
    format : str
        Output format: "parquet" or "csv"
    overwrite : bool
        Allow overwriting existing files

    Returns
    -------
    Path
        Path to the created file
    """
    # Add extension if not present
    if format == "parquet" and not str(path).endswith(".parquet"):
        path = Path(str(path) + ".parquet")
    elif format == "csv" and not str(path).endswith(".csv"):
        path = Path(str(path) + ".csv")

    if path.exists() and not overwrite:
        msg = f"File exists and overwrite=False: {path}"
        raise ProcessingError(msg)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "csv":
        df.to_csv(path, index=False)
    else:
        msg = f"Unknown format: {format}"
        raise ValueError(msg)

    logger.info(f"Exported {len(df)} rows to {path}")
    return path


def export_detections(
    detections: pd.DataFrame,
    config: ExportConfig,
) -> Path:
    """Export detection results.

    Parameters
    ----------
    detections : pd.DataFrame
        Detection DataFrame
    config : ExportConfig
        Export configuration

    Returns
    -------
    Path
        Path to the created file
    """
    path = config.output_dir / "detections"
    return export_dataframe(detections, path, config.format, config.overwrite)


def export_tracks(
    tracks: pd.DataFrame,
    config: ExportConfig,
) -> Path:
    """Export tracking results.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks DataFrame
    config : ExportConfig
        Export configuration

    Returns
    -------
    Path
        Path to the created file
    """
    path = config.output_dir / "tracks"
    return export_dataframe(tracks, path, config.format, config.overwrite)


def export_config(
    config: PipelineConfig,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Export pipeline configuration to YAML.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration
    output_dir : Path
        Output directory
    overwrite : bool
        Allow overwriting existing files

    Returns
    -------
    Path
        Path to the created file
    """
    path = output_dir / "config.yaml"

    if path.exists() and not overwrite:
        msg = f"File exists and overwrite=False: {path}"
        raise ProcessingError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, handling Path objects
    config_dict = config.model_dump(mode="json")

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Exported configuration to {path}")
    return path


def get_environment_info() -> dict[str, Any]:
    """Collect environment information for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary with Python version, platform, and package versions
    """
    import numpy as np
    import pandas as pd
    import trackpy as tp

    import pt3d

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "pt3d": pt3d.__version__,
            "trackpy": tp.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def export_run_info(
    output_dir: Path,
    config: PipelineConfig,
    input_info: dict[str, Any] | None = None,
    results_summary: dict[str, Any] | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    error: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Export run metadata for reproducibility.

    Parameters
    ----------
    output_dir : Path
        Output directory
    config : PipelineConfig
        Pipeline configuration
    input_info : dict[str, Any] | None
        Information about input data (path, shape, dtype, etc.)
    results_summary : dict[str, Any] | None
        Summary of results (n_detections, n_tracks, etc.)
    start_time : datetime | None
        Pipeline start time
    end_time : datetime | None
        Pipeline end time
    error : str | None
        Error message if pipeline failed
    overwrite : bool
        Allow overwriting existing files

    Returns
    -------
    Path
        Path to the created file
    """
    path = output_dir / "run.json"

    if path.exists() and not overwrite:
        msg = f"File exists and overwrite=False: {path}"
        raise ProcessingError(msg)

    path.parent.mkdir(parents=True, exist_ok=True)

    run_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": get_environment_info(),
        "config": config.model_dump(mode="json"),
    }

    if input_info is not None:
        run_info["input"] = input_info

    if results_summary is not None:
        run_info["results"] = results_summary

    if start_time is not None:
        run_info["start_time"] = start_time.isoformat()

    if end_time is not None:
        run_info["end_time"] = end_time.isoformat()
        if start_time is not None:
            duration = (end_time - start_time).total_seconds()
            run_info["duration_seconds"] = duration

    if error is not None:
        run_info["error"] = error

    with open(path, "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    logger.info(f"Exported run info to {path}")
    return path
