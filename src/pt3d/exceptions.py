"""Custom exceptions for pt3d."""


class Pt3dError(Exception):
    """Base exception for pt3d."""


class ConfigError(Pt3dError):
    """Reserved configuration exception; not raised by current config models.

    Pydantic model validation raises pydantic.ValidationError instead.
    """


class DataError(Pt3dError):
    """Exception raised for input data errors.

    Examples:
        - Incorrect array dimensions
        - Unsupported file formats or missing files
        - Missing required axes
    """


class ProcessingError(Pt3dError):
    """Exception raised for processing failures.

    Examples:
        - Detection failed for a frame
        - Tracking failed due to algorithm errors
        - Export refused because a file exists and overwrite is disabled

    Native I/O exceptions are not all wrapped in this exception.
    """
