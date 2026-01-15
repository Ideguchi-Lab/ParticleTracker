"""Tests for configuration models."""

import pytest
from pydantic import ValidationError

from pt3d.config import (
    DetectionConfig,
    InputConfig,
    PostprocessConfig,
    TrackingConfig,
    VoxelSize,
)


class TestVoxelSize:
    def test_valid_voxel_size(self):
        vs = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        assert vs.z_um == 0.8
        assert vs.y_um == 0.2
        assert vs.x_um == 0.2

    def test_anisotropy_ratio(self):
        vs = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        assert vs.anisotropy_ratio == 4.0

    def test_as_tuple(self):
        vs = VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2)
        assert vs.as_tuple() == (0.8, 0.2, 0.2)

    def test_negative_value_rejected(self):
        with pytest.raises(ValidationError):
            VoxelSize(z_um=-0.8, y_um=0.2, x_um=0.2)

    def test_zero_value_rejected(self):
        with pytest.raises(ValidationError):
            VoxelSize(z_um=0, y_um=0.2, x_um=0.2)


class TestDetectionConfig:
    def test_valid_diameter(self):
        config = DetectionConfig(diameter=(5, 9, 9))
        assert config.diameter == (5, 9, 9)

    def test_even_diameter_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            DetectionConfig(diameter=(4, 9, 9))
        assert "odd" in str(exc_info.value).lower()

    def test_all_even_diameter_rejected(self):
        with pytest.raises(ValidationError):
            DetectionConfig(diameter=(4, 8, 10))

    def test_negative_diameter_rejected(self):
        with pytest.raises(ValidationError):
            DetectionConfig(diameter=(-5, 9, 9))

    def test_default_values(self):
        config = DetectionConfig(diameter=(5, 9, 9))
        assert config.minmass == 0.0
        assert config.threshold is None
        assert config.separation is None
        assert config.invert is False
        assert config.preprocess is True


class TestTrackingConfig:
    def test_valid_config(self):
        config = TrackingConfig(search_range_um=2.0)
        assert config.search_range_um == 2.0
        assert config.memory == 0

    def test_with_memory(self):
        config = TrackingConfig(search_range_um=2.0, memory=3)
        assert config.memory == 3

    def test_negative_search_range_rejected(self):
        with pytest.raises(ValidationError):
            TrackingConfig(search_range_um=-1.0)

    def test_adaptive_params_together(self):
        # Both must be provided or neither
        config = TrackingConfig(search_range_um=2.0, adaptive_stop=10.0, adaptive_step=0.5)
        assert config.adaptive_stop == 10.0

    def test_adaptive_params_incomplete_rejected(self):
        with pytest.raises(ValidationError):
            TrackingConfig(search_range_um=2.0, adaptive_stop=10.0)


class TestInputConfig:
    def test_valid_axis_order(self):
        config = InputConfig(axis_order="tzyx", voxel_size=VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2))
        assert config.axis_order == "tzyx"

    def test_invalid_axis_rejected(self):
        with pytest.raises(ValidationError):
            InputConfig(
                axis_order="tcyx",  # 'c' is invalid
                voxel_size=VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2),
            )

    def test_case_insensitive_axis(self):
        config = InputConfig(axis_order="TZYX", voxel_size=VoxelSize(z_um=0.8, y_um=0.2, x_um=0.2))
        assert config.axis_order == "tzyx"


class TestPostprocessConfig:
    def test_defaults(self):
        config = PostprocessConfig()
        assert config.min_track_length == 2
        assert config.max_velocity_um is None

    def test_custom_values(self):
        config = PostprocessConfig(min_track_length=5, max_velocity_um=10.0)
        assert config.min_track_length == 5
        assert config.max_velocity_um == 10.0
