"""napari widgets for interactive particle tracking.

This module provides Qt widgets for the napari plugin interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


class VoxelSizeWidget(QWidget):
    """Widget for entering voxel size."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Z spacing
        layout.addWidget(QLabel("Z:"))
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0.01, 100.0)
        self.z_spin.setValue(0.8)
        self.z_spin.setSuffix(" µm")
        self.z_spin.setDecimals(3)
        layout.addWidget(self.z_spin)

        # Y spacing
        layout.addWidget(QLabel("Y:"))
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(0.01, 100.0)
        self.y_spin.setValue(0.2)
        self.y_spin.setSuffix(" µm")
        self.y_spin.setDecimals(3)
        layout.addWidget(self.y_spin)

        # X spacing
        layout.addWidget(QLabel("X:"))
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(0.01, 100.0)
        self.x_spin.setValue(0.2)
        self.x_spin.setSuffix(" µm")
        self.x_spin.setDecimals(3)
        layout.addWidget(self.x_spin)

        self.setLayout(layout)

    def get_values(self) -> tuple[float, float, float]:
        """Get voxel size values (z, y, x) in µm."""
        return (self.z_spin.value(), self.y_spin.value(), self.x_spin.value())


class DiameterWidget(QWidget):
    """Widget for entering detection diameter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Z diameter
        layout.addWidget(QLabel("Z:"))
        self.z_spin = QSpinBox()
        self.z_spin.setRange(3, 51)
        self.z_spin.setSingleStep(2)
        self.z_spin.setValue(5)
        layout.addWidget(self.z_spin)

        # Y diameter
        layout.addWidget(QLabel("Y:"))
        self.y_spin = QSpinBox()
        self.y_spin.setRange(3, 51)
        self.y_spin.setSingleStep(2)
        self.y_spin.setValue(9)
        layout.addWidget(self.y_spin)

        # X diameter
        layout.addWidget(QLabel("X:"))
        self.x_spin = QSpinBox()
        self.x_spin.setRange(3, 51)
        self.x_spin.setSingleStep(2)
        self.x_spin.setValue(9)
        layout.addWidget(self.x_spin)

        self.setLayout(layout)

    def get_values(self) -> tuple[int, int, int]:
        """Get diameter values (dz, dy, dx)."""
        return (self.z_spin.value(), self.y_spin.value(), self.x_spin.value())


class DetectionWidget(QWidget):
    """Widget for detection parameter adjustment."""

    def __init__(self, napari_viewer: "napari.Viewer", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        # Image layer selector
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Image:"))
        self.layer_combo = QComboBox()
        self._update_layer_combo()
        layer_layout.addWidget(self.layer_combo)
        layout.addLayout(layer_layout)

        # Diameter
        diameter_group = QGroupBox("Diameter (must be odd)")
        self.diameter_widget = DiameterWidget()
        diameter_layout = QVBoxLayout()
        diameter_layout.addWidget(self.diameter_widget)
        diameter_group.setLayout(diameter_layout)
        layout.addWidget(diameter_group)

        # Minmass
        minmass_layout = QHBoxLayout()
        minmass_layout.addWidget(QLabel("Min mass:"))
        self.minmass_spin = QDoubleSpinBox()
        self.minmass_spin.setRange(0, 1e9)
        self.minmass_spin.setValue(100.0)
        self.minmass_spin.setDecimals(1)
        minmass_layout.addWidget(self.minmass_spin)
        layout.addLayout(minmass_layout)

        # Threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0, 1e6)
        self.threshold_spin.setValue(0)
        self.threshold_spin.setSpecialValueText("Auto")
        threshold_layout.addWidget(self.threshold_spin)
        layout.addLayout(threshold_layout)

        # Invert checkbox
        self.invert_check = QCheckBox("Invert (dark particles)")
        layout.addWidget(self.invert_check)

        # Preprocess checkbox
        self.preprocess_check = QCheckBox("Use trackpy preprocessing")
        self.preprocess_check.setChecked(True)
        layout.addWidget(self.preprocess_check)

        # Frame range
        range_group = QGroupBox("Frame Range")
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Start:"))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 10000)
        range_layout.addWidget(self.start_frame_spin)
        range_layout.addWidget(QLabel("End:"))
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(1, 10001)
        self.end_frame_spin.setValue(10)
        range_layout.addWidget(self.end_frame_spin)
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)

        # Run button
        self.run_button = QPushButton("Run Detection")
        self.run_button.clicked.connect(self._run_detection)
        layout.addWidget(self.run_button)

        layout.addStretch()
        self.setLayout(layout)

    def _update_layer_combo(self) -> None:
        """Update layer combo box with available image layers."""
        self.layer_combo.clear()
        for layer in self.viewer.layers:
            if hasattr(layer, "data") and hasattr(layer.data, "ndim"):
                if layer.data.ndim >= 3:
                    self.layer_combo.addItem(layer.name)

    def _run_detection(self) -> None:
        """Execute detection on selected image layer."""
        from pt3d.config import DetectionConfig
        from pt3d.detect import detect_batch
        from pt3d.napari.layers import to_napari_points

        layer_name = self.layer_combo.currentText()
        if not layer_name:
            logger.warning("No image layer selected")
            return

        layer = self.viewer.layers[layer_name]
        data = layer.data

        # Ensure 4D
        if data.ndim == 3:
            data = data[None, ...]

        # Get detection config
        config = DetectionConfig(
            diameter=self.diameter_widget.get_values(),
            minmass=self.minmass_spin.value(),
            threshold=self.threshold_spin.value() if self.threshold_spin.value() > 0 else None,
            invert=self.invert_check.isChecked(),
            preprocess=self.preprocess_check.isChecked(),
        )

        # Frame range
        start = self.start_frame_spin.value()
        end = min(self.end_frame_spin.value(), data.shape[0])
        frame_range = (start, end) if start > 0 or end < data.shape[0] else None

        try:
            detections = detect_batch(data, config, frame_range)
            logger.info(f"Detected {len(detections)} particles")

            if len(detections) > 0:
                points = to_napari_points(detections)
                self.viewer.add_points(
                    points,
                    name="Detections",
                    size=5,
                    face_color="yellow",
                )
        except Exception as e:
            logger.exception(f"Detection failed: {e}")


class TrackingWidget(QWidget):
    """Widget for tracking parameter adjustment."""

    def __init__(self, napari_viewer: "napari.Viewer", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        # Voxel size
        voxel_group = QGroupBox("Voxel Size")
        self.voxel_widget = VoxelSizeWidget()
        voxel_layout = QVBoxLayout()
        voxel_layout.addWidget(self.voxel_widget)
        voxel_group.setLayout(voxel_layout)
        layout.addWidget(voxel_group)

        # Search range
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Search range:"))
        self.search_range_spin = QDoubleSpinBox()
        self.search_range_spin.setRange(0.01, 100.0)
        self.search_range_spin.setValue(2.0)
        self.search_range_spin.setSuffix(" µm")
        self.search_range_spin.setDecimals(2)
        range_layout.addWidget(self.search_range_spin)
        layout.addLayout(range_layout)

        # Memory
        memory_layout = QHBoxLayout()
        memory_layout.addWidget(QLabel("Memory:"))
        self.memory_spin = QSpinBox()
        self.memory_spin.setRange(0, 10)
        self.memory_spin.setValue(2)
        self.memory_spin.setSuffix(" frames")
        memory_layout.addWidget(self.memory_spin)
        layout.addLayout(memory_layout)

        # Min track length
        min_len_layout = QHBoxLayout()
        min_len_layout.addWidget(QLabel("Min track length:"))
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 100)
        self.min_length_spin.setValue(5)
        self.min_length_spin.setSuffix(" frames")
        min_len_layout.addWidget(self.min_length_spin)
        layout.addLayout(min_len_layout)

        # Run button
        self.run_button = QPushButton("Run Tracking")
        self.run_button.clicked.connect(self._run_tracking)
        layout.addWidget(self.run_button)

        layout.addStretch()
        self.setLayout(layout)

    def _run_tracking(self) -> None:
        """Execute tracking on current detections."""
        from pt3d.config import PostprocessConfig, TrackingConfig, VoxelSize
        from pt3d.napari.layers import to_napari_tracks
        from pt3d.postprocess import filter_stubs
        from pt3d.track import link_detections

        # Find detections layer
        detections_layer = None
        for layer in self.viewer.layers:
            if layer.name == "Detections" and hasattr(layer, "data"):
                detections_layer = layer
                break

        if detections_layer is None:
            logger.warning("No Detections layer found. Run detection first.")
            return

        import pandas as pd

        # Convert points to DataFrame
        points = detections_layer.data
        if points.shape[1] == 4:
            detections = pd.DataFrame(points, columns=["frame", "z", "y", "x"])
        else:
            detections = pd.DataFrame(points, columns=["z", "y", "x"])
            detections["frame"] = 0

        # Get config
        z_um, y_um, x_um = self.voxel_widget.get_values()
        voxel_size = VoxelSize(z_um=z_um, y_um=y_um, x_um=x_um)

        track_config = TrackingConfig(
            search_range_um=self.search_range_spin.value(),
            memory=self.memory_spin.value(),
        )

        try:
            tracks = link_detections(detections, track_config, voxel_size)
            tracks = filter_stubs(tracks, self.min_length_spin.value())

            logger.info(f"Found {tracks['particle'].nunique()} tracks")

            if len(tracks) > 0:
                tracks_data = to_napari_tracks(tracks)
                self.viewer.add_tracks(
                    tracks_data,
                    name="Tracks",
                )
        except Exception as e:
            logger.exception(f"Tracking failed: {e}")


class MainWidget(QWidget):
    """Main widget combining detection, tracking, and export."""

    def __init__(self, napari_viewer: "napari.Viewer") -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        # Title
        title = QLabel("3D Particle Tracker")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Detection section
        detect_group = QGroupBox("Detection")
        self.detection_widget = DetectionWidget(self.viewer)
        detect_layout = QVBoxLayout()
        detect_layout.addWidget(self.detection_widget)
        detect_group.setLayout(detect_layout)
        layout.addWidget(detect_group)

        # Tracking section
        track_group = QGroupBox("Tracking")
        self.tracking_widget = TrackingWidget(self.viewer)
        track_layout = QVBoxLayout()
        track_layout.addWidget(self.tracking_widget)
        track_group.setLayout(track_layout)
        layout.addWidget(track_group)

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)
