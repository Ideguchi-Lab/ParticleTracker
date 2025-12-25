"""Tests for napari conversion functions."""

import numpy as np
import pandas as pd
import pytest

from pt3d.config import VoxelSize
from pt3d.napari.layers import (
    get_napari_scale,
    points_properties_from_detections,
    to_napari_points,
    to_napari_tracks,
    tracks_properties_from_tracks,
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
