"""napari widgets for interactive particle tracking.

This module provides Qt widgets for the napari plugin interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari
    import numpy as np
    import pandas as pd

    from pt3d.config import VoxelSize

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

    def __init__(self, napari_viewer: napari.Viewer, parent: QWidget | None = None) -> None:
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

    # Signal emitted when tracking completes: (tracks_df, voxel_size)
    tracking_completed = Signal(object, object)

    def __init__(self, napari_viewer: napari.Viewer, parent: QWidget | None = None) -> None:
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
        from pt3d.config import TrackingConfig, VoxelSize
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

                # Emit signal with tracks and voxel size for visualization
                self.tracking_completed.emit(tracks, voxel_size)
        except Exception as e:
            logger.exception(f"Tracking failed: {e}")


class TrackVisualizationWidget(QWidget):
    """Widget for track visualization settings and statistics display."""

    def __init__(self, napari_viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer
        self._current_tracks: pd.DataFrame | None = None
        self._current_stats: pd.DataFrame | None = None
        self._voxel_size: VoxelSize | None = None
        self._original_image_data: np.ndarray | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        # Display mode section
        display_group = QGroupBox("Display Mode")
        display_layout = QVBoxLayout()

        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["2D Slice + Time Slider", "XY Max Projection Overview"])
        self.display_mode_combo.currentIndexChanged.connect(self._on_display_mode_changed)
        display_layout.addWidget(self.display_mode_combo)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Track coloring section
        color_group = QGroupBox("Track Coloring")
        color_layout = QVBoxLayout()

        color_by_layout = QHBoxLayout()
        color_by_layout.addWidget(QLabel("Color by:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["Track ID", "Time (frame)", "Track Length"])
        self.color_by_combo.currentIndexChanged.connect(self._update_track_colors)
        color_by_layout.addWidget(self.color_by_combo)
        color_layout.addLayout(color_by_layout)

        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["turbo", "viridis", "plasma", "magma", "hsv"])
        self.colormap_combo.currentIndexChanged.connect(self._update_track_colors)
        cmap_layout.addWidget(self.colormap_combo)
        color_layout.addLayout(cmap_layout)

        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        # Track selection section
        select_group = QGroupBox("Track Selection")
        select_layout = QVBoxLayout()

        track_id_layout = QHBoxLayout()
        track_id_layout.addWidget(QLabel("Highlight Track:"))
        self.track_id_spin = QSpinBox()
        self.track_id_spin.setRange(-1, 10000)
        self.track_id_spin.setValue(-1)
        self.track_id_spin.setSpecialValueText("All")
        self.track_id_spin.valueChanged.connect(self._highlight_track)
        track_id_layout.addWidget(self.track_id_spin)
        select_layout.addLayout(track_id_layout)

        self.show_others_check = QCheckBox("Show other tracks (dimmed)")
        self.show_others_check.setChecked(True)
        self.show_others_check.stateChanged.connect(self._highlight_track)
        select_layout.addWidget(self.show_others_check)

        select_group.setLayout(select_layout)
        layout.addWidget(select_group)

        # Statistics display section
        stats_group = QGroupBox("Track Statistics")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(120)
        stats_layout.addWidget(self.stats_text)

        self.export_stats_btn = QPushButton("Export Statistics...")
        self.export_stats_btn.clicked.connect(self._export_statistics)
        stats_layout.addWidget(self.export_stats_btn)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Apply button
        self.apply_button = QPushButton("Update Visualization")
        self.apply_button.clicked.connect(self._apply_visualization)
        layout.addWidget(self.apply_button)

        layout.addStretch()
        self.setLayout(layout)

    def set_tracks(self, tracks: pd.DataFrame, voxel_size: VoxelSize) -> None:
        """Set current tracks and compute statistics."""

        from pt3d.postprocess import compute_track_stats

        self._current_tracks = tracks
        self._voxel_size = voxel_size

        if len(tracks) > 0:
            self._current_stats = compute_track_stats(tracks, voxel_size)
            self._update_stats_display()

            # Update track ID spinner range
            max_id = int(tracks["particle"].max())
            self.track_id_spin.setMaximum(max_id)

        # Store original image data for projection
        for layer in self.viewer.layers:
            if hasattr(layer, "data") and hasattr(layer.data, "ndim"):
                if layer.data.ndim == 4:
                    self._original_image_data = layer.data
                    break

    def _update_stats_display(self) -> None:
        """Update statistics text display."""
        if self._current_stats is None or len(self._current_stats) == 0:
            self.stats_text.setText("No tracks available")
            return

        stats = self._current_stats
        summary = (
            f"Total Tracks: {len(stats)}\n"
            f"Mean Track Length: {stats['length'].mean():.1f} frames\n"
            f"Max Track Length: {stats['length'].max()} frames\n"
            f"Mean Velocity: {stats['mean_velocity_um'].mean():.2f} µm/frame\n"
            f"Max Velocity: {stats['max_velocity_um'].max():.2f} µm/frame"
        )
        self.stats_text.setText(summary)

    def _on_display_mode_changed(self) -> None:
        """Handle display mode change."""
        mode = self.display_mode_combo.currentIndex()
        if mode == 0:
            self._show_slice_mode()
        else:
            self._show_max_projection_mode()

    def _show_slice_mode(self) -> None:
        """Display 2D slice + time slider mode."""
        # Remove max projection layer if exists
        for layer in list(self.viewer.layers):
            if layer.name == "XY Max Projection" or layer.name == "Tracks (Max Proj)":
                self.viewer.layers.remove(layer)

        # Make original layers visible
        for layer in self.viewer.layers:
            if layer.name in ["Particles", "Tracks"]:
                layer.visible = True

    def _show_max_projection_mode(self) -> None:
        """Display XY max projection overview."""

        from pt3d.napari.layers import (
            compute_xy_max_projection,
            to_napari_tracks,
            tracks_visualization_properties,
        )

        if self._original_image_data is None or self._voxel_size is None:
            logger.warning("No image data available for projection")
            return

        # Compute max projection
        proj_image, proj_tracks = compute_xy_max_projection(self._original_image_data, self._current_tracks)

        # Hide original layers
        for layer in self.viewer.layers:
            if layer.name in ["Particles", "Tracks"]:
                layer.visible = False

        # Add or update max projection layer
        proj_scale = (1.0, self._voxel_size.y_um, self._voxel_size.x_um)

        existing_proj = None
        for layer in self.viewer.layers:
            if layer.name == "XY Max Projection":
                existing_proj = layer
                break

        if existing_proj is not None:
            existing_proj.data = proj_image
        else:
            self.viewer.add_image(
                proj_image,
                name="XY Max Projection",
                scale=proj_scale,
                colormap="gray",
            )

        # Add projected tracks
        if proj_tracks is not None and len(proj_tracks) > 0:
            # Remove existing projected tracks
            for layer in list(self.viewer.layers):
                if layer.name == "Tracks (Max Proj)":
                    self.viewer.layers.remove(layer)

            tracks_data = to_napari_tracks(proj_tracks)
            properties, color_by = tracks_visualization_properties(
                proj_tracks, self._current_stats, self._get_color_by_key()
            )

            # For 3D (T, Y, X), use 3D scale
            track_scale = (1.0, self._voxel_size.y_um, self._voxel_size.x_um)
            self.viewer.add_tracks(
                tracks_data,
                name="Tracks (Max Proj)",
                scale=track_scale,
                properties=properties,
                color_by=color_by,
                colormap=self.colormap_combo.currentText(),
            )

    def _get_color_by_key(self) -> str:
        """Get color_by key from combo box selection."""
        index = self.color_by_combo.currentIndex()
        mapping = {0: "track_id", 1: "time", 2: "length"}
        return mapping.get(index, "track_id")

    def _update_track_colors(self) -> None:
        """Update track layer coloring based on selected property."""
        from pt3d.napari.layers import tracks_visualization_properties

        if self._current_tracks is None or len(self._current_tracks) == 0:
            return

        color_by = self._get_color_by_key()
        colormap = self.colormap_combo.currentText()

        # Update Tracks layer
        for layer in self.viewer.layers:
            if layer.name == "Tracks" and hasattr(layer, "color_by"):
                properties, actual_color_by = tracks_visualization_properties(
                    self._current_tracks, self._current_stats, color_by
                )
                layer.properties = properties
                layer.color_by = actual_color_by
                layer.colormap = colormap
                break

    def _highlight_track(self) -> None:
        """Highlight selected track ID."""
        if self._current_tracks is None:
            return

        track_id = self.track_id_spin.value()
        show_others = self.show_others_check.isChecked()

        # Find Tracks layer
        tracks_layer = None
        for layer in self.viewer.layers:
            if layer.name == "Tracks":
                tracks_layer = layer
                break

        if tracks_layer is None:
            return

        if track_id == -1:
            # Show all tracks
            tracks_layer.visible = True
        else:
            # Currently napari Tracks layer doesn't support filtering directly,
            # so we log the info for now
            logger.info(f"Highlighting track {track_id} (show others: {show_others})")

    def _export_statistics(self) -> None:
        """Export track statistics to file."""
        if self._current_stats is None or len(self._current_stats) == 0:
            logger.warning("No statistics to export")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Statistics",
            "track_statistics.csv",
            "CSV Files (*.csv);;Parquet Files (*.parquet)",
        )

        if filepath:
            if filepath.endswith(".parquet"):
                self._current_stats.to_parquet(filepath)
            else:
                if not filepath.endswith(".csv"):
                    filepath += ".csv"
                self._current_stats.to_csv(filepath, index=False)
            logger.info(f"Exported statistics to {filepath}")

    def _apply_visualization(self) -> None:
        """Apply current visualization settings."""
        self._on_display_mode_changed()
        self._update_track_colors()


class MainWidget(QWidget):
    """Main widget combining detection, tracking, visualization, and export."""

    def __init__(self, napari_viewer: napari.Viewer) -> None:
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

        # Visualization section (NEW)
        vis_group = QGroupBox("Visualization")
        self.visualization_widget = TrackVisualizationWidget(self.viewer)
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.visualization_widget)
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # Connect tracking completion to visualization
        self.tracking_widget.tracking_completed.connect(self._on_tracking_completed)

        # Status
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setLayout(layout)

    def _on_tracking_completed(self, tracks: pd.DataFrame, voxel_size: VoxelSize) -> None:
        """Handle tracking completion to update visualization widget."""
        self.visualization_widget.set_tracks(tracks, voxel_size)
        self.status_label.setText(f"Tracking complete: {tracks['particle'].nunique()} tracks")
