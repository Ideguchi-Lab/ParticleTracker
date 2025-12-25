"""Tests for tracking module."""

import numpy as np
import pandas as pd
import pytest

from pt3d.config import TrackingConfig, VoxelSize
from pt3d.exceptions import ProcessingError
from pt3d.track import link_detections, relabel_tracks


class TestLinkDetections:
    def test_basic_linking(self, sample_detections):
        voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        config = TrackingConfig(search_range_um=1.0)

        tracks = link_detections(sample_detections, config, voxel_size)

        assert "particle" in tracks.columns
        # Should find 3 tracks (one per particle trajectory)
        assert tracks["particle"].nunique() == 3

    def test_with_memory(self, sample_detections):
        voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        config = TrackingConfig(search_range_um=1.0, memory=2)

        tracks = link_detections(sample_detections, config, voxel_size)

        assert "particle" in tracks.columns

    def test_empty_detections(self):
        empty_df = pd.DataFrame(columns=["frame", "z", "y", "x"])
        voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        config = TrackingConfig(search_range_um=1.0)

        tracks = link_detections(empty_df, config, voxel_size)

        assert len(tracks) == 0
        assert "particle" in tracks.columns

    def test_missing_columns_raises_error(self):
        bad_df = pd.DataFrame({"z": [10.0], "y": [20.0]})  # Missing x and frame
        voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        config = TrackingConfig(search_range_um=1.0)

        with pytest.raises(ProcessingError):
            link_detections(bad_df, config, voxel_size)

    def test_anisotropy_correction(self):
        # Create detections where z motion is large in pixels but small in µm
        detections = pd.DataFrame(
            {
                "frame": [0, 1],
                "z": [10.0, 15.0],  # 5 pixel change
                "y": [20.0, 20.0],
                "x": [30.0, 30.0],
            }
        )
        # With z_um=0.8, 5 pixels = 4.0 µm
        voxel_size = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        config = TrackingConfig(search_range_um=5.0)  # > 4.0 µm

        tracks = link_detections(detections, config, voxel_size)

        # Should link as one track
        assert tracks["particle"].nunique() == 1


class TestRelabelTracks:
    def test_dense_relabeling(self, sample_tracks):
        # Create gaps in particle IDs
        sample_tracks["particle"] = [0, 5, 10, 0, 5, 10, 0, 5, 10]

        relabeled = relabel_tracks(sample_tracks, mode="dense")

        # Should be consecutive starting from 0
        unique_ids = sorted(relabeled["particle"].unique())
        assert unique_ids == [0, 1, 2]

    def test_sorted_relabeling(self):
        # Particle 5 appears first, then 0, then 10
        df = pd.DataFrame(
            {
                "frame": [0, 1, 2],
                "particle": [5, 0, 10],
                "z": [10.0, 20.0, 30.0],
                "y": [10.0, 20.0, 30.0],
                "x": [10.0, 20.0, 30.0],
            }
        )

        relabeled = relabel_tracks(df, mode="sorted")

        # Order should be based on first appearance
        assert relabeled[relabeled["frame"] == 0]["particle"].iloc[0] == 0
        assert relabeled[relabeled["frame"] == 1]["particle"].iloc[0] == 1
        assert relabeled[relabeled["frame"] == 2]["particle"].iloc[0] == 2

    def test_missing_particle_column_raises_error(self):
        df = pd.DataFrame({"frame": [0], "z": [10.0], "y": [20.0], "x": [30.0]})

        with pytest.raises(ProcessingError):
            relabel_tracks(df)

    def test_unknown_mode_raises_error(self, sample_tracks):
        with pytest.raises(ValueError):
            relabel_tracks(sample_tracks, mode="unknown")
