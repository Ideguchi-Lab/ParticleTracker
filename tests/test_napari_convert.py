"""Tests for napari conversion functions."""

import numpy as np
import pandas as pd
import pytest

from pt3d.config import VoxelSize
from pt3d.napari.layers import (
    compute_xy_max_projection,
    get_napari_scale,
    merge_track_stats_to_tracks,
    points_properties_from_detections,
    to_napari_points,
    to_napari_tracks,
    tracks_properties_from_tracks,
    tracks_visualization_properties,
)


class TestToNapariPoints:
    def test_4d_points(self, sample_detections):
        points = to_napari_points(sample_detections, include_frame=True)

        assert points.shape == (len(sample_detections), 4)
        assert points.dtype == np.float64
        # Check order is [frame, z, y, x]
        np.testing.assert_array_equal(
            points[:, 0], sample_detections["frame"].values
        )

    def test_3d_points(self, sample_detections):
        points = to_napari_points(sample_detections, include_frame=False)

        assert points.shape == (len(sample_detections), 3)
        # Check order is [z, y, x]
        np.testing.assert_array_equal(points[:, 0], sample_detections["z"].values)

    def test_empty_detections(self):
        empty_df = pd.DataFrame(columns=["frame", "z", "y", "x"])

        points_4d = to_napari_points(empty_df, include_frame=True)
        assert points_4d.shape == (0, 4)

        points_3d = to_napari_points(empty_df, include_frame=False)
        assert points_3d.shape == (0, 3)


class TestToNapariTracks:
    def test_tracks_format(self, sample_tracks):
        tracks_array = to_napari_tracks(sample_tracks)

        assert tracks_array.shape == (len(sample_tracks), 5)
        assert tracks_array.dtype == np.float64
        # Check order is [track_id, t, z, y, x]
        # Data should be sorted by (particle, frame)

    def test_tracks_sorting(self):
        # Create unsorted tracks
        df = pd.DataFrame(
            {
                "particle": [1, 0, 1, 0],
                "frame": [1, 0, 0, 1],
                "z": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0],
                "x": [1.0, 2.0, 3.0, 4.0],
            }
        )

        tracks_array = to_napari_tracks(df)

        # Should be sorted by (particle, frame)
        assert tracks_array[0, 0] == 0  # particle 0 first
        assert tracks_array[0, 1] == 0  # frame 0
        assert tracks_array[1, 0] == 0  # particle 0
        assert tracks_array[1, 1] == 1  # frame 1
        assert tracks_array[2, 0] == 1  # particle 1
        assert tracks_array[2, 1] == 0  # frame 0

    def test_empty_tracks(self):
        empty_df = pd.DataFrame(columns=["particle", "frame", "z", "y", "x"])
        tracks_array = to_napari_tracks(empty_df)
        assert tracks_array.shape == (0, 5)


class TestGetNapariScale:
    def test_4d_scale(self):
        voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        scale = get_napari_scale(voxel_size, include_time=True)

        assert scale == (1.0, 0.8, 0.2, 0.2)

    def test_3d_scale(self):
        voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        scale = get_napari_scale(voxel_size, include_time=False)

        assert scale == (0.8, 0.2, 0.2)


class TestPointsProperties:
    def test_default_properties(self, sample_detections):
        properties = points_properties_from_detections(sample_detections)

        assert "mass" in properties
        assert "size" in properties
        assert len(properties["mass"]) == len(sample_detections)

    def test_custom_properties(self):
        df = pd.DataFrame(
            {
                "frame": [0, 1],
                "z": [10.0, 20.0],
                "y": [10.0, 20.0],
                "x": [10.0, 20.0],
                "custom_prop": [1.0, 2.0],
            }
        )

        properties = points_properties_from_detections(df, ["custom_prop"])

        assert "custom_prop" in properties
        assert len(properties) == 1

    def test_empty_detections(self):
        empty_df = pd.DataFrame(columns=["frame", "z", "y", "x"])
        properties = points_properties_from_detections(empty_df)
        assert properties == {}


class TestTracksProperties:
    def test_properties(self, sample_tracks):
        sample_tracks["mass"] = 100.0
        properties = tracks_properties_from_tracks(sample_tracks)

        assert "mass" in properties
        assert len(properties["mass"]) == len(sample_tracks)


class TestTracksVisualizationProperties:
    """Tests for track visualization property generation."""

    def test_color_by_track_id(self, sample_tracks):
        properties, color_by = tracks_visualization_properties(sample_tracks, color_by="track_id")

        assert "track_id" in properties
        assert color_by == "track_id"
        assert len(properties["track_id"]) == len(sample_tracks)

    def test_color_by_time(self, sample_tracks):
        properties, color_by = tracks_visualization_properties(sample_tracks, color_by="time")

        assert "time" in properties
        assert color_by == "time"

    def test_color_by_with_stats(self, sample_tracks):
        from pt3d.postprocess import compute_track_stats

        voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        stats = compute_track_stats(sample_tracks, voxel_size)

        properties, color_by = tracks_visualization_properties(
            sample_tracks, stats, color_by="length"
        )

        assert "length" in properties
        assert color_by == "length"

    def test_fallback_color_by(self, sample_tracks):
        # Request a property that doesn't exist without stats
        _properties, color_by = tracks_visualization_properties(
            sample_tracks, color_by="velocity"
        )

        # Should fallback to track_id
        assert color_by == "track_id"

    def test_empty_tracks(self):
        empty_df = pd.DataFrame(columns=["particle", "frame", "z", "y", "x"])
        properties, color_by = tracks_visualization_properties(empty_df)

        assert properties == {}
        assert color_by == "track_id"


class TestMergeTrackStats:
    """Tests for merging track statistics into tracks."""

    def test_merge_stats(self, sample_tracks):
        from pt3d.postprocess import compute_track_stats

        voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        stats = compute_track_stats(sample_tracks, voxel_size)

        merged = merge_track_stats_to_tracks(sample_tracks, stats)

        assert "length" in merged.columns
        assert len(merged) == len(sample_tracks)
        # Each point in a track should have the same length value
        track0_lengths = merged[merged["particle"] == 0]["length"]
        assert track0_lengths.nunique() == 1

    def test_empty_tracks(self):
        empty_tracks = pd.DataFrame(columns=["particle", "frame", "z", "y", "x"])
        empty_stats = pd.DataFrame(columns=["particle", "length"])

        result = merge_track_stats_to_tracks(empty_tracks, empty_stats)
        assert len(result) == 0


class TestComputeXYMaxProjection:
    """Tests for XY maximum intensity projection."""

    def test_projection_shape(self):
        # 4D input: (T=5, Z=10, Y=32, X=32)
        image_data = np.random.rand(5, 10, 32, 32).astype(np.float64)

        proj, _ = compute_xy_max_projection(image_data)

        assert proj.shape == (5, 32, 32)  # (T, Y, X)

    def test_projection_values(self):
        # Create data where max along Z is known
        image_data = np.zeros((2, 3, 4, 4), dtype=np.float64)
        image_data[0, 0, 2, 2] = 1.0  # Max at z=0
        image_data[0, 2, 2, 2] = 2.0  # Max at z=2

        proj, _ = compute_xy_max_projection(image_data)

        assert proj[0, 2, 2] == 2.0  # Should be max value

    def test_projection_with_tracks(self, sample_tracks):
        image_data = np.random.rand(3, 32, 64, 64).astype(np.float64)

        _proj, proj_tracks = compute_xy_max_projection(image_data, sample_tracks)

        assert proj_tracks is not None
        # Z coordinate should be set to 0
        assert (proj_tracks["z"] == 0.0).all()
        # Other columns should be preserved
        assert len(proj_tracks) == len(sample_tracks)

    def test_projection_without_tracks(self):
        image_data = np.random.rand(3, 10, 32, 32).astype(np.float64)

        _proj, proj_tracks = compute_xy_max_projection(image_data)

        assert proj_tracks is None

    def test_invalid_dimensions(self):
        # 3D data should raise error
        image_data = np.random.rand(10, 32, 32)

        with pytest.raises(ValueError, match="Expected 4D"):
            compute_xy_max_projection(image_data)
