"""Configuration models for pt3d pipeline.

All configuration is managed through Pydantic models for validation
and serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class VoxelSize(BaseModel):
    """Physical voxel dimensions in micrometers.

    Attributes
    ----------
    z_um : float
        Z spacing in micrometers
    y_um : float
        Y spacing in micrometers
    x_um : float
        X spacing in micrometers
    """

    z_um: float = Field(gt=0, description="Z spacing in micrometers")
    y_um: float = Field(gt=0, description="Y spacing in micrometers")
    x_um: float = Field(gt=0, description="X spacing in micrometers")

    @property
    def anisotropy_ratio(self) -> float:
        """Return the z/x anisotropy ratio."""
        return self.z_um / self.x_um

    def as_tuple(self) -> tuple[float, float, float]:
        """Return voxel size as (z, y, x) tuple."""
        return (self.z_um, self.y_um, self.x_um)


class InputConfig(BaseModel):
    """Input data configuration.

    Attributes
    ----------
    path : Path | None
        Path to input file (zarr, TIFF, or npy)
    axis_order : str
        Axis order of input data (default: "tzyx")
    dtype : str | None
        Optional dtype to convert input data to
    voxel_size : VoxelSize
        Physical voxel dimensions (required for µm-based tracking)
    """

    path: Path | None = None
    axis_order: str = Field(default="tzyx", description="Axis order of input data")
    dtype: str | None = None
    voxel_size: VoxelSize = Field(description="Physical voxel dimensions in µm")

    @field_validator("axis_order")
    @classmethod
    def validate_axis_order(cls, v: str) -> str:
        """Validate axis order contains only valid axes."""
        v = v.lower()
        valid_axes = set("tzyx")
        if not set(v).issubset(valid_axes):
            invalid = set(v) - valid_axes
            msg = f"Invalid axes in axis_order: {invalid}"
            raise ValueError(msg)
        return v


class DetectionConfig(BaseModel):
    """Detection parameters for trackpy.

    Attributes
    ----------
    diameter : tuple[int, int, int]
        Feature diameter (dz, dy, dx) - must be odd integers
    minmass : float
        Minimum integrated brightness for a particle
    threshold : float | None
        Clip bandpass result below this value (noise floor)
    separation : tuple[int, int, int] | None
        Minimum separation between features
    invert : bool
        Set to True for dark particles on light background
    preprocess : bool
        Use trackpy's built-in preprocessing (bandpass filter)
    """

    diameter: tuple[int, int, int] = Field(
        description="Feature diameter (dz, dy, dx) - must be odd integers"
    )
    minmass: float = Field(default=0.0, ge=0, description="Minimum integrated brightness")
    threshold: float | None = Field(default=None, description="Noise floor threshold")
    separation: tuple[int, int, int] | None = Field(
        default=None, description="Minimum separation between features"
    )
    invert: bool = Field(default=False, description="Invert for dark particles")
    preprocess: bool = Field(default=True, description="Use trackpy preprocessing")

    @field_validator("diameter")
    @classmethod
    def check_diameter_odd(cls, v: tuple[int, int, int]) -> tuple[int, int, int]:
        """Validate that all diameter values are odd integers."""
        for i, d in enumerate(v):
            if d % 2 == 0:
                axis = ["z", "y", "x"][i]
                msg = f"Diameter for {axis} axis must be odd, got {d}"
                raise ValueError(msg)
            if d < 1:
                axis = ["z", "y", "x"][i]
                msg = f"Diameter for {axis} axis must be positive, got {d}"
                raise ValueError(msg)
        return v

    @field_validator("separation")
    @classmethod
    def check_separation_positive(
        cls, v: tuple[int, int, int] | None
    ) -> tuple[int, int, int] | None:
        """Validate that separation values are positive if provided."""
        if v is not None:
            for i, s in enumerate(v):
                if s < 1:
                    axis = ["z", "y", "x"][i]
                    msg = f"Separation for {axis} axis must be positive, got {s}"
                    raise ValueError(msg)
        return v


class TrackingConfig(BaseModel):
    """Tracking (linking) parameters for trackpy.

    All distance parameters are in micrometers (µm).

    Attributes
    ----------
    search_range_um : float
        Maximum displacement per frame in micrometers
    memory : int
        Number of frames a particle can disappear and reappear
    adaptive_stop : float | None
        Stop adaptive search when subnetwork contains this many particles
    adaptive_step : float | None
        Reduce search_range by this factor in adaptive search
    """

    search_range_um: float = Field(
        gt=0, description="Maximum displacement per frame in µm"
    )
    memory: int = Field(default=0, ge=0, description="Frames to remember lost particles")
    adaptive_stop: float | None = Field(
        default=None, description="Adaptive search stop threshold"
    )
    adaptive_step: float | None = Field(
        default=None, description="Adaptive search step factor"
    )

    @model_validator(mode="after")
    def check_adaptive_params(self) -> "TrackingConfig":
        """Validate adaptive search parameters are used together."""
        if (self.adaptive_stop is None) != (self.adaptive_step is None):
            msg = "adaptive_stop and adaptive_step must be used together"
            raise ValueError(msg)
        return self


class PostprocessConfig(BaseModel):
    """Postprocessing configuration.

    Attributes
    ----------
    min_track_length : int
        Minimum number of frames for a valid track
    max_velocity_um : float | None
        Maximum allowed velocity in µm/frame (outlier filter)
    """

    min_track_length: int = Field(
        default=2, ge=1, description="Minimum track length in frames"
    )
    max_velocity_um: float | None = Field(
        default=None, ge=0, description="Maximum velocity for outlier removal"
    )


class ExportConfig(BaseModel):
    """Export configuration.

    Attributes
    ----------
    output_dir : Path
        Directory for output files
    format : Literal["parquet", "csv"]
        Output format for DataFrames
    overwrite : bool
        Allow overwriting existing files
    """

    output_dir: Path = Field(description="Output directory")
    format: Literal["parquet", "csv"] = Field(
        default="parquet", description="Output format"
    )
    overwrite: bool = Field(default=False, description="Allow overwriting files")


class NapariConfig(BaseModel):
    """napari visualization configuration.

    Attributes
    ----------
    image_name : str
        Name for the image layer
    points_name : str
        Name for the points layer
    tracks_name : str
        Name for the tracks layer
    frame_range : tuple[int, int] | None
        Optional subset of frames to display
    """

    image_name: str = Field(default="Volume", description="Image layer name")
    points_name: str = Field(default="Detections", description="Points layer name")
    tracks_name: str = Field(default="Tracks", description="Tracks layer name")
    frame_range: tuple[int, int] | None = Field(
        default=None, description="Frame range for subset display"
    )


class PipelineConfig(BaseModel):
    """Complete pipeline configuration.

    Attributes
    ----------
    input : InputConfig
        Input data configuration
    detection : DetectionConfig
        Detection parameters
    tracking : TrackingConfig
        Tracking parameters
    postprocess : PostprocessConfig
        Postprocessing configuration
    export : ExportConfig | None
        Export configuration (optional)
    napari : NapariConfig
        napari visualization settings
    """

    input: InputConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    export: ExportConfig | None = None
    napari: NapariConfig = Field(default_factory=NapariConfig)
