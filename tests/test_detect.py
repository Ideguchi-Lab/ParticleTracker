"""Tests for detection module."""

import numpy as np
import pandas as pd
import pytest

from pt3d.config import DetectionConfig
from pt3d.detect import detect_batch, detect_frame, detect_single_frame, detect_streaming
from pt3d.exceptions import ProcessingError


class TestDetectFrame:
    def test_detect_synthetic_particles(self, synthetic_volume):
        config = DetectionConfig(
            diameter=(5, 9, 9),
            minmass=0.1,
        )
        detections = detect_frame(synthetic_volume, config)

        # Should detect approximately 3 particles
        assert len(detections) >= 2
        assert len(detections) <= 5

        # Check required columns
        assert "z" in detections.columns
        assert "y" in detections.columns
        assert "x" in detections.columns

    def test_detect_empty_volume(self):
        volume = np.zeros((32, 64, 64), dtype=np.float64)
        config = DetectionConfig(
            diameter=(5, 9, 9),
            minmass=100.0,
        )
        detections = detect_frame(volume, config)

        assert len(detections) == 0

    def test_wrong_dimensions_raises_error(self):
        volume_2d = np.zeros((64, 64), dtype=np.float64)
        config = DetectionConfig(diameter=(5, 9, 9))

        with pytest.raises(ProcessingError):
            detect_frame(volume_2d, config)


class TestDetectBatch:
    def test_detect_4d_data(self, synthetic_4d_data):
        volumes, _ = synthetic_4d_data
        config = DetectionConfig(
            diameter=(5, 9, 9),
            minmass=0.1,
        )

        detections = detect_batch(volumes, config)

        # Should detect particles in multiple frames
        assert len(detections) > 0
        assert "frame" in detections.columns
        assert detections["frame"].nunique() > 1

    def test_frame_range(self, synthetic_4d_data):
        volumes, _ = synthetic_4d_data
        config = DetectionConfig(
            diameter=(5, 9, 9),
            minmass=0.1,
        )

        # Process only first 3 frames
        detections = detect_batch(volumes, config, frame_range=(0, 3))

        assert detections["frame"].max() < 3
        assert detections["frame"].min() >= 0

    def test_wrong_dimensions_raises_error(self):
        volume_3d = np.zeros((32, 64, 64), dtype=np.float64)
        config = DetectionConfig(diameter=(5, 9, 9))

        with pytest.raises(ProcessingError):
            detect_batch(volume_3d, config)

    def test_invalid_frame_range_raises_error(self, synthetic_4d_data):
        volumes, _ = synthetic_4d_data
        config = DetectionConfig(diameter=(5, 9, 9))

        with pytest.raises(ProcessingError):
            detect_batch(volumes, config, frame_range=(5, 3))


class TestDetectStreaming:
    def test_processes_each_volume_and_preserves_source_frame_numbers(self, monkeypatch):
        volumes = [
            np.full((2, 3, 4), fill_value=frame, dtype=np.float64)
            for frame in range(3)
        ]
        located_features = [
            pd.DataFrame({"z": [1.0], "y": [2.0], "x": [3.0]}),
            pd.DataFrame(columns=["z", "y", "x"]),
            pd.DataFrame({"z": [4.0, 5.0], "y": [6.0, 7.0], "x": [8.0, 9.0]}),
        ]
        seen_frames = []

        def fake_locate(volume, **_kwargs):
            seen_frames.append(int(volume[0, 0, 0]))
            return located_features[len(seen_frames) - 1]

        monkeypatch.setattr("pt3d.detect.tp.locate", fake_locate)
        config = DetectionConfig(diameter=(3, 3, 3))

        detections = detect_streaming(iter(volumes), config)

        assert seen_frames == [0, 1, 2]
        np.testing.assert_array_equal(detections["frame"].to_numpy(), [0, 2, 2])
        np.testing.assert_allclose(
            detections[["z", "y", "x"]].to_numpy(),
            [[1.0, 2.0, 3.0], [4.0, 6.0, 8.0], [5.0, 7.0, 9.0]],
        )

    def test_rejects_non_3d_volume_at_its_source_frame(self, monkeypatch):
        monkeypatch.setattr(
            "pt3d.detect.tp.locate",
            lambda _volume, **_kwargs: pd.DataFrame(columns=["z", "y", "x"]),
        )
        volumes = [np.zeros((2, 3, 4)), np.zeros((3, 4))]
        config = DetectionConfig(diameter=(3, 3, 3))

        with pytest.raises(ProcessingError, match=r"Volume must be 3D, got 2D at frame 1"):
            detect_streaming(iter(volumes), config)


class TestDetectSingleFrame:
    def test_detect_single_frame(self, synthetic_4d_data):
        volumes, _ = synthetic_4d_data
        config = DetectionConfig(
            diameter=(5, 9, 9),
            minmass=0.1,
        )

        detections = detect_single_frame(volumes, frame_index=2, config=config)

        assert len(detections) > 0
        assert "frame" in detections.columns
        assert (detections["frame"] == 2).all()

    def test_invalid_frame_index_raises_error(self, synthetic_4d_data):
        volumes, _ = synthetic_4d_data
        config = DetectionConfig(diameter=(5, 9, 9))

        with pytest.raises(ProcessingError):
            detect_single_frame(volumes, frame_index=100, config=config)
