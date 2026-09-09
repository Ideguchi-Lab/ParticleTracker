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
        Axis order for file inputs (default: "tzyx"). run_pipeline ignores
        this setting for ndarray inputs, which must already be zyx or tzyx.
    dtype : str | None
        Stored metadata only; the pipeline does not convert the input dtype.
        Convert arrays explicitly before passing them to the pipeline.
    voxel_size : VoxelSize
        Physical voxel dimensions (required for um-based tracking)
    """

    path: Path | None = None
    axis_order: str = Field(default="tzyx", description="Axis order of input data")
    dtype: str | None = None
    voxel_size: VoxelSize = Field(description="Physical voxel dimensions in um")

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
    diameter : tuple[int, int, int] | None
        Feature diameter (dz, dy, dx) in pixels - must be odd integers.
        Either diameter or diameter_um must be specified.
    diameter_um : float | None
        Feature diameter in micrometers (isotropic).
        Will be converted to pixels using voxel_size.
        Either diameter or diameter_um must be specified.
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

    diameter: tuple[int, int, int] | None = Field(
        default=None, description="Feature diameter (dz, dy, dx) in pixels - must be odd integers"
    )
    diameter_um: float | None = Field(default=None, gt=0, description="Feature diameter in micrometers (isotropic)")
    minmass: float = Field(default=0.0, ge=0, description="Minimum integrated brightness")
    threshold: float | None = Field(default=None, description="Noise floor threshold")
    separation: tuple[int, int, int] | None = Field(default=None, description="Minimum separation between features")
    invert: bool = Field(default=False, description="Invert for dark particles")
    preprocess: bool = Field(default=True, description="Use trackpy preprocessing")

    @field_validator("diameter")
    @classmethod
    def check_diameter_odd(cls, v: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        """Validate that all diameter values are odd integers."""
        if v is None:
            return v
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
    def check_separation_positive(cls, v: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        """Validate that separation values are positive if provided."""
        if v is not None:
            for i, s in enumerate(v):
                if s < 1:
                    axis = ["z", "y", "x"][i]
                    msg = f"Separation for {axis} axis must be positive, got {s}"
                    raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def check_diameter_specified(self) -> DetectionConfig:
        """Validate that either diameter or diameter_um is specified."""
        if self.diameter is None and self.diameter_um is None:
            msg = "Either 'diameter' (pixels) or 'diameter_um' (micrometers) must be specified"
            raise ValueError(msg)
        if self.diameter is not None and self.diameter_um is not None:
            msg = "Specify either 'diameter' or 'diameter_um', not both"
            raise ValueError(msg)
        return self

    def get_diameter_pixels(self, voxel_size: VoxelSize) -> tuple[int, int, int]:
        """Get diameter in pixels, converting from um if necessary.

        Parameters
        ----------
        voxel_size : VoxelSize
            Physical voxel dimensions for conversion

        Returns
        -------
        tuple[int, int, int]
            Diameter in pixels (dz, dy, dx), guaranteed to be odd integers
        """
        if self.diameter is not None:
            return self.diameter

        # Convert from um to pixels
        assert self.diameter_um is not None
        dz = self.diameter_um / voxel_size.z_um
        dy = self.diameter_um / voxel_size.y_um
        dx = self.diameter_um / voxel_size.x_um

        # Round to an integer (minimum 1), then increase even values by 1.
        def to_odd(v: float) -> int:
            rounded = max(1, round(v))
            if rounded % 2 == 0:
                # Increase the rounded even integer to the next odd integer.
                return rounded + 1
            return rounded

        return (to_odd(dz), to_odd(dy), to_odd(dx))


class TrackingConfig(BaseModel):
    """Tracking (linking) parameters for trackpy.

    search_range_um is in micrometers. adaptive_stop is passed to trackpy
    in scaled coordinate units: one unit is the smallest voxel dimension.

    Attributes
    ----------
    search_range_um : float
        Maximum displacement per frame in micrometers
    memory : int
        Number of frames a particle can disappear and reappear
    adaptive_stop : float | None
        Give up on an oversized subnet when the reduced search range is at
        or below this distance, in scaled coordinate units. Convert a desired
        threshold in um by dividing by min(voxel_size.as_tuple()).
    adaptive_step : float | None
        Reduce search_range by this factor in adaptive search
    """

    search_range_um: float = Field(gt=0, description="Maximum displacement per frame in um")
    memory: int = Field(default=0, ge=0, description="Frames to remember lost particles")
    adaptive_stop: float | None = Field(default=None, description="Adaptive stop distance in scaled coordinate units")
    adaptive_step: float | None = Field(default=None, description="Adaptive search step factor")

    @model_validator(mode="after")
    def check_adaptive_params(self) -> TrackingConfig:
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
        Minimum number of observed points per track, excluding missing frames
    max_velocity_um : float | None
        Maximum allowed velocity in um/frame (outlier filter)
    """

    min_track_length: int = Field(default=2, ge=1, description="Minimum track length in frames")
    max_velocity_um: float | None = Field(default=None, ge=0, description="Maximum velocity for outlier removal")


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
    format: Literal["parquet", "csv"] = Field(default="parquet", description="Output format")
    overwrite: bool = Field(default=False, description="Allow overwriting files")


class NapariConfig(BaseModel):
    """Stored napari preferences, not applied by the pipeline or widgets.

    Set layer names and slice data explicitly when creating napari layers.

    Attributes
    ----------
    image_name : str
        Name for the image layer
    points_name : str
        Name for the points layer
    tracks_name : str
        Name for the tracks layer
    frame_range : tuple[int, int] | None
        Stored frame range only; no automatic display filtering is performed
    """

    image_name: str = Field(default="Volume", description="Image layer name")
    points_name: str = Field(default="Detections", description="Points layer name")
    tracks_name: str = Field(default="Tracks", description="Tracks layer name")
    frame_range: tuple[int, int] | None = Field(default=None, description="Stored frame range; not applied to display")


class Volume3DConfig(BaseModel):
    """3D volume rendering configuration for napari.

    Attributes
    ----------
    rendering_mode : str
        Rendering mode: mip, attenuated_mip, translucent, or iso
    contrast_percentile_low : float
        Lower percentile for contrast limits (0-100)
    contrast_percentile_high : float
        Upper percentile for contrast limits (0-100)
    gamma : float
        Gamma correction value
    opacity : float
        Layer opacity (0-1)
    iso_threshold : float
        Threshold for iso rendering mode (0-1, relative to the contrast limits)
    colormap : str
        Colormap name for volume rendering
    """

    rendering_mode: Literal["mip", "attenuated_mip", "translucent", "iso"] = Field(
        default="mip", description="Volume rendering mode"
    )
    contrast_percentile_low: float = Field(default=1.0, ge=0, le=100, description="Lower percentile for contrast")
    contrast_percentile_high: float = Field(default=99.0, ge=0, le=100, description="Upper percentile for contrast")
    gamma: float = Field(default=1.0, gt=0, description="Gamma correction")
    opacity: float = Field(default=0.5, ge=0, le=1, description="Layer opacity")
    iso_threshold: float = Field(default=0.5, ge=0, le=1, description="ISO threshold (relative)")
    colormap: str = Field(default="gray", description="Colormap for volume")


class Track3DConfig(BaseModel):
    """3D track visualization configuration.

    Attributes
    ----------
    colormap : str
        Colormap for track coloring
    color_by : str
        Property to color tracks by
    tail_length : int
        Number of frames to show in track trail
    show_current_position : bool
        Reserved setting; currently has no effect on rendering
    """

    colormap: str = Field(default="turbo", description="Track colormap")
    color_by: Literal["track_id", "time", "length", "velocity"] = Field(
        default="track_id", description="Property for track coloring"
    )
    tail_length: int = Field(default=10, ge=0, description="Trail length in frames")
    show_current_position: bool = Field(default=True, description="Reserved; current-position highlighting is not applied")


class Points3DConfig(BaseModel):
    """3D detection points visualization configuration.

    Attributes
    ----------
    size : float
        Point size in display units
    face_color : str
        Color for point faces
    opacity : float
        Point opacity (0-1)
    show_current_frame_only : bool
        Reserved setting; currently has no effect on rendering
    """

    size: float = Field(default=5.0, ge=1, description="Point size")
    face_color: str = Field(default="yellow", description="Point face color")
    opacity: float = Field(default=0.8, ge=0, le=1, description="Point opacity")
    show_current_frame_only: bool = Field(default=True, description="Reserved; frame filtering is not applied")


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
