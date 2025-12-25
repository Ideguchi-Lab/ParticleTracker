"""Utility functions for pt3d."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pt3d.config import VoxelSize

# Standard internal axis order: (t, z, y, x)
STANDARD_AXIS_ORDER = "tzyx"


def normalize_axis_order(
    data: NDArray[np.floating],
    axis_order: str,
) -> NDArray[np.floating]:
    """Transpose data to standard axis order (t, z, y, x).

    Parameters
    ----------
    data : NDArray[np.floating]
        Input array with arbitrary axis order
    axis_order : str
        Current axis order as string (e.g., "zyxt", "xyzc")
        Only 't', 'z', 'y', 'x' axes are supported

    Returns
    -------
    NDArray[np.floating]
        Array transposed to (t, z, y, x) order

    Raises
    ------
    ValueError
        If axis_order contains unsupported axes or wrong number of dimensions
    """
    axis_order = axis_order.lower()

    # Validate axis order
    valid_axes = set("tzyx")
    if not set(axis_order).issubset(valid_axes):
        invalid = set(axis_order) - valid_axes
        msg = f"Unsupported axes: {invalid}. Only 't', 'z', 'y', 'x' are supported."
        raise ValueError(msg)

    if len(axis_order) != data.ndim:
        msg = f"Axis order '{axis_order}' has {len(axis_order)} axes but data has {data.ndim} dimensions"
        raise ValueError(msg)

    # Handle 3D data (assume single timepoint)
    if data.ndim == 3:
        if axis_order == "zyx":
            # Add time dimension
            return data[np.newaxis, ...]
        # Transpose to zyx first
        current_order = [axis_order.index(ax) for ax in "zyx"]
        transposed = np.transpose(data, current_order)
        return transposed[np.newaxis, ...]

    # Handle 4D data
    if data.ndim == 4:
        if axis_order == STANDARD_AXIS_ORDER:
            return data
        # Create transpose order
        current_order = [axis_order.index(ax) for ax in STANDARD_AXIS_ORDER]
        return np.transpose(data, current_order)

    msg = f"Data must be 3D or 4D, got {data.ndim}D"
    raise ValueError(msg)


def apply_coordinate_scaling(
    df: pd.DataFrame,
    voxel_size: VoxelSize,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Scale coordinates for isotropic distance calculation.

    Scales z, y, x coordinates to physical units (µm) and then normalizes
    to the finest resolution (typically x_um) for isotropic distance calculation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with z, y, x columns in pixel coordinates
    voxel_size : VoxelSize
        Physical voxel dimensions in micrometers
    inplace : bool, optional
        If True, modify DataFrame in place. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional z_scaled, y_scaled, x_scaled columns
    """
    if not inplace:
        df = df.copy()

    # Find the finest resolution (smallest voxel dimension)
    min_um = min(voxel_size.z_um, voxel_size.y_um, voxel_size.x_um)

    # Scale coordinates to be isotropic
    # After scaling, 1 unit = min_um micrometers in all dimensions
    df["z_scaled"] = df["z"] * (voxel_size.z_um / min_um)
    df["y_scaled"] = df["y"] * (voxel_size.y_um / min_um)
    df["x_scaled"] = df["x"] * (voxel_size.x_um / min_um)

    return df


def pixel_to_um(
    value_px: float,
    pixel_size_um: float,
) -> float:
    """Convert pixel value to micrometers.

    Parameters
    ----------
    value_px : float
        Value in pixels
    pixel_size_um : float
        Pixel size in micrometers

    Returns
    -------
    float
        Value in micrometers
    """
    return value_px * pixel_size_um


def um_to_pixel(
    value_um: float,
    pixel_size_um: float,
) -> float:
    """Convert micrometers to pixels.

    Parameters
    ----------
    value_um : float
        Value in micrometers
    pixel_size_um : float
        Pixel size in micrometers

    Returns
    -------
    float
        Value in pixels
    """
    return value_um / pixel_size_um


def search_range_um_to_scaled(
    search_range_um: float,
    voxel_size: VoxelSize,
) -> float:
    """Convert search range from µm to scaled coordinate units.

    The scaled coordinate system uses the finest resolution as the unit.

    Parameters
    ----------
    search_range_um : float
        Search range in micrometers
    voxel_size : VoxelSize
        Physical voxel dimensions

    Returns
    -------
    float
        Search range in scaled coordinate units
    """
    # Find the finest resolution
    min_um = min(voxel_size.z_um, voxel_size.y_um, voxel_size.x_um)
    # Convert µm to scaled units (where 1 unit = min_um µm)
    return search_range_um / min_um


def validate_odd_diameter(diameter: tuple[int, int, int]) -> None:
    """Validate that all diameter values are odd integers.

    Parameters
    ----------
    diameter : tuple[int, int, int]
        Diameter values (dz, dy, dx)

    Raises
    ------
    ValueError
        If any diameter value is not an odd integer
    """
    for i, d in enumerate(diameter):
        if d % 2 == 0:
            axis = ["z", "y", "x"][i]
            msg = f"Diameter for {axis} axis must be odd, got {d}"
            raise ValueError(msg)
        if d < 1:
            axis = ["z", "y", "x"][i]
            msg = f"Diameter for {axis} axis must be positive, got {d}"
            raise ValueError(msg)
