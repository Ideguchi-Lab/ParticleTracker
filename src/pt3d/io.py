"""Input/Output module for pt3d.

Handles loading data from various formats and normalizing to the
internal standard axis order (t, z, y, x).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pt3d.exceptions import DataError
from pt3d.utils import normalize_axis_order

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_array(
    data: NDArray[np.floating],
    axis_order: str = "tzyx",
) -> NDArray[np.floating]:
    """Load and normalize a numpy array.

    Parameters
    ----------
    data : NDArray[np.floating]
        Input array (3D or 4D)
    axis_order : str
        Current axis order of the data

    Returns
    -------
    NDArray[np.floating]
        Array normalized to (t, z, y, x) order

    Raises
    ------
    DataError
        If data dimensions are invalid
    """
    if data.ndim not in (3, 4):
        msg = f"Data must be 3D or 4D, got {data.ndim}D"
        raise DataError(msg)

    try:
        return normalize_axis_order(data, axis_order)
    except ValueError as e:
        raise DataError(str(e)) from e


def load_npy(
    path: Path | str,
    axis_order: str = "tzyx",
) -> NDArray[np.floating]:
    """Load data from a .npy file.

    Parameters
    ----------
    path : Path | str
        Path to .npy file
    axis_order : str
        Axis order of the saved data

    Returns
    -------
    NDArray[np.floating]
        Array normalized to (t, z, y, x) order
    """
    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        raise DataError(msg)

    data = np.load(path)
    return load_array(data, axis_order)


def load_zarr(
    path: Path | str,
    dataset_key: str | None = None,
    axis_order: str = "tzyx",
) -> NDArray[np.floating]:
    """Load data from a zarr file.

    Parameters
    ----------
    path : Path | str
        Path to zarr directory or file
    dataset_key : str | None
        Key for the dataset within the zarr store
        If None, loads the root array
    axis_order : str
        Axis order of the saved data

    Returns
    -------
    NDArray[np.floating]
        Array normalized to (t, z, y, x) order
    """
    import zarr

    path = Path(path)
    if not path.exists():
        msg = f"Zarr store not found: {path}"
        raise DataError(msg)

    store = zarr.open(path, mode="r")

    if dataset_key is not None:
        if dataset_key not in store:
            msg = f"Dataset '{dataset_key}' not found in zarr store"
            raise DataError(msg)
        data = np.asarray(store[dataset_key])
    else:
        # Try to load as array directly
        if isinstance(store, zarr.Array):
            data = np.asarray(store)
        else:
            msg = "Zarr store is a group; please specify dataset_key"
            raise DataError(msg)

    return load_array(data, axis_order)


def load_tiff(
    path: Path | str,
    axis_order: str = "tzyx",
) -> NDArray[np.floating]:
    """Load data from a TIFF file.

    Parameters
    ----------
    path : Path | str
        Path to TIFF file
    axis_order : str
        Axis order of the saved data

    Returns
    -------
    NDArray[np.floating]
        Array normalized to (t, z, y, x) order
    """
    import tifffile

    path = Path(path)
    if not path.exists():
        msg = f"TIFF file not found: {path}"
        raise DataError(msg)

    data = tifffile.imread(path)
    return load_array(data, axis_order)


def load_data(
    source: Path | str | NDArray[np.floating],
    axis_order: str = "tzyx",
    dataset_key: str | None = None,
) -> NDArray[np.floating]:
    """Load data from various sources.

    Automatically detects the source type and loads accordingly.

    Parameters
    ----------
    source : Path | str | NDArray[np.floating]
        Input source: file path or numpy array
    axis_order : str
        Axis order of the data
    dataset_key : str | None
        For zarr files, the dataset key

    Returns
    -------
    NDArray[np.floating]
        Array normalized to (t, z, y, x) order

    Raises
    ------
    DataError
        If source type is not recognized
    """
    # Direct array input
    if isinstance(source, np.ndarray):
        return load_array(source, axis_order)

    # File path input
    path = Path(source)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return load_npy(path, axis_order)
    if suffix == ".zarr" or path.is_dir():
        return load_zarr(path, dataset_key, axis_order)
    if suffix in (".tif", ".tiff"):
        return load_tiff(path, axis_order)

    msg = f"Unsupported file format: {suffix}"
    raise DataError(msg)
