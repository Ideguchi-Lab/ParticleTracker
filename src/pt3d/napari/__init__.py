"""napari plugin for pt3d."""

from pt3d.napari.layers import (
    CAMERA_PRESETS,
    compute_xy_max_projection,
    configure_3d_image_layer,
    configure_3d_points_layer,
    configure_3d_tracks_layer,
    get_camera_preset,
    get_napari_scale,
    merge_track_stats_to_tracks,
    to_napari_points,
    to_napari_tracks,
    tracks_visualization_properties,
)
from pt3d.napari.widgets import (
    MainWidget,
    Track3DVisualizationWidget,
    TrackVisualizationWidget,
)

__all__ = [
    "CAMERA_PRESETS",
    "MainWidget",
    "Track3DVisualizationWidget",
    "TrackVisualizationWidget",
    "compute_xy_max_projection",
    "configure_3d_image_layer",
    "configure_3d_points_layer",
    "configure_3d_tracks_layer",
    "get_camera_preset",
    "get_napari_scale",
    "merge_track_stats_to_tracks",
    "to_napari_points",
    "to_napari_tracks",
    "tracks_visualization_properties",
]
