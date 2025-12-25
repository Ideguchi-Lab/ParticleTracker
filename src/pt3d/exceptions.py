"""Custom exceptions for pt3d."""


class Pt3dError(Exception):
    """Base exception for pt3d."""


class ConfigError(Pt3dError):
    """Exception raised for configuration errors.

    Examples:
        - Diameter values are not odd integers
        - Invalid axis order specification
        - Missing required configuration fields
    """


class DataError(Pt3dError):
    """Exception raised for input data errors.

    Examples:
        - Incorrect array dimensions
        - Invalid data types
        - Missing required axes
    """


class ProcessingError(Pt3dError):
    """Exception raised for processing failures.

    Examples:
        - Detection failed for a frame
        - Tracking failed due to algorithm errors
        - File I/O errors during processing
    """
