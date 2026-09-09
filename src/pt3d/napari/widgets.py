"""napari widgets for interactive particle tracking.

This module provides Qt widgets for the napari plugin interface.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari
    import numpy as np
    import pandas as pd

    from pt3d.config import Points3DConfig, Track3DConfig, Volume3DConfig, VoxelSize

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
        """Log the selected ID; individual-track highlighting is not implemented.

        Selecting All restores the Tracks layer's visibility.
        """
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


class Track3DVisualizationWidget(QWidget):
    """Widget for interactive 3D visualization of tracks, volumes, and detection points.

    This widget provides controls for:
    - 3D display mode toggle
    - Volume rendering settings (MIP, attenuated MIP, translucent, ISO)
    - Track visualization (color by, colormap, trail length)
    - Detection points display
    - Camera presets and time navigation
    """

    visualization_updated = Signal()

    def __init__(self, napari_viewer: napari.Viewer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = napari_viewer
        self._current_tracks: pd.DataFrame | None = None
        self._current_stats: pd.DataFrame | None = None
        self._voxel_size: VoxelSize | None = None
        self._is_3d_mode = False
        self._play_timer: QTimer | None = None
        self._max_frames = 100
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup the widget UI layout."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("3D Track Visualization")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # 3D mode toggle
        self.mode_3d_check = QCheckBox("Enable 3D Display")
        self.mode_3d_check.stateChanged.connect(self._on_3d_mode_toggled)
        layout.addWidget(self.mode_3d_check)

        # Volume Rendering group
        layout.addWidget(self._setup_volume_group())

        # Track Display group
        layout.addWidget(self._setup_track_group())

        # Detection Points group
        layout.addWidget(self._setup_points_group())

        # Camera group
        layout.addWidget(self._setup_camera_group())

        # Time Navigation group
        layout.addWidget(self._setup_time_group())

        # Apply button
        self.apply_button = QPushButton("Update Visualization")
        self.apply_button.clicked.connect(self._apply_visualization)
        layout.addWidget(self.apply_button)

        layout.addStretch()
        self.setLayout(layout)

    def _setup_volume_group(self) -> QGroupBox:
        """Setup volume rendering controls."""
        group = QGroupBox("Volume Rendering")
        layout = QVBoxLayout()

        # Rendering mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Rendering:"))
        self.rendering_combo = QComboBox()
        self.rendering_combo.addItems(["mip", "attenuated_mip", "translucent", "iso"])
        self.rendering_combo.currentIndexChanged.connect(self._on_volume_settings_changed)
        mode_layout.addWidget(self.rendering_combo)
        layout.addLayout(mode_layout)

        # Contrast percentile
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_low_spin = QDoubleSpinBox()
        self.contrast_low_spin.setRange(0, 100)
        self.contrast_low_spin.setValue(1.0)
        self.contrast_low_spin.setSuffix("%")
        self.contrast_low_spin.valueChanged.connect(self._on_volume_settings_changed)
        contrast_layout.addWidget(self.contrast_low_spin)
        contrast_layout.addWidget(QLabel("-"))
        self.contrast_high_spin = QDoubleSpinBox()
        self.contrast_high_spin.setRange(0, 100)
        self.contrast_high_spin.setValue(99.0)
        self.contrast_high_spin.setSuffix("%")
        self.contrast_high_spin.valueChanged.connect(self._on_volume_settings_changed)
        contrast_layout.addWidget(self.contrast_high_spin)
        layout.addLayout(contrast_layout)

        # Gamma
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(10, 300)  # 0.1 to 3.0
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self._on_volume_settings_changed)
        gamma_layout.addWidget(self.gamma_slider)
        self.gamma_label = QLabel("1.0")
        gamma_layout.addWidget(self.gamma_label)
        layout.addLayout(gamma_layout)

        # Opacity
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.volume_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_opacity_slider.setRange(0, 100)
        self.volume_opacity_slider.setValue(50)
        self.volume_opacity_slider.valueChanged.connect(self._on_volume_settings_changed)
        opacity_layout.addWidget(self.volume_opacity_slider)
        self.volume_opacity_label = QLabel("0.5")
        opacity_layout.addWidget(self.volume_opacity_label)
        layout.addLayout(opacity_layout)

        # Colormap
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.volume_colormap_combo = QComboBox()
        self.volume_colormap_combo.addItems(["gray", "viridis", "magma", "plasma", "inferno"])
        self.volume_colormap_combo.currentIndexChanged.connect(self._on_volume_settings_changed)
        cmap_layout.addWidget(self.volume_colormap_combo)
        layout.addLayout(cmap_layout)

        group.setLayout(layout)
        return group

    def _setup_track_group(self) -> QGroupBox:
        """Setup track visualization controls."""
        group = QGroupBox("Track Display")
        layout = QVBoxLayout()

        # Color by
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color by:"))
        self.track_color_by_combo = QComboBox()
        self.track_color_by_combo.addItems(["Track ID", "Time (frame)", "Track Length", "Velocity"])
        self.track_color_by_combo.currentIndexChanged.connect(self._on_track_settings_changed)
        color_layout.addWidget(self.track_color_by_combo)
        layout.addLayout(color_layout)

        # Colormap
        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.track_colormap_combo = QComboBox()
        self.track_colormap_combo.addItems(["turbo", "viridis", "plasma", "magma", "hsv"])
        self.track_colormap_combo.currentIndexChanged.connect(self._on_track_settings_changed)
        cmap_layout.addWidget(self.track_colormap_combo)
        layout.addLayout(cmap_layout)

        # Trail length
        trail_layout = QHBoxLayout()
        trail_layout.addWidget(QLabel("Trail:"))
        self.trail_slider = QSlider(Qt.Orientation.Horizontal)
        self.trail_slider.setRange(0, 100)
        self.trail_slider.setValue(10)
        self.trail_slider.valueChanged.connect(self._on_track_settings_changed)
        trail_layout.addWidget(self.trail_slider)
        self.trail_label = QLabel("10 frames")
        trail_layout.addWidget(self.trail_label)
        layout.addLayout(trail_layout)

        # Show current position
        self.show_position_check = QCheckBox("Highlight current position")
        self.show_position_check.setChecked(True)
        layout.addWidget(self.show_position_check)

        group.setLayout(layout)
        return group

    def _setup_points_group(self) -> QGroupBox:
        """Setup detection points controls."""
        group = QGroupBox("Detection Points")
        layout = QVBoxLayout()

        # Size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.points_size_spin = QSpinBox()
        self.points_size_spin.setRange(1, 50)
        self.points_size_spin.setValue(5)
        self.points_size_spin.valueChanged.connect(self._on_points_settings_changed)
        size_layout.addWidget(self.points_size_spin)
        layout.addLayout(size_layout)

        # Color
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color:"))
        self.points_color_combo = QComboBox()
        self.points_color_combo.addItems(["yellow", "red", "green", "cyan", "magenta", "white"])
        self.points_color_combo.currentIndexChanged.connect(self._on_points_settings_changed)
        color_layout.addWidget(self.points_color_combo)
        layout.addLayout(color_layout)

        # Opacity
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.points_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.points_opacity_slider.setRange(0, 100)
        self.points_opacity_slider.setValue(80)
        self.points_opacity_slider.valueChanged.connect(self._on_points_settings_changed)
        opacity_layout.addWidget(self.points_opacity_slider)
        self.points_opacity_label = QLabel("0.8")
        opacity_layout.addWidget(self.points_opacity_label)
        layout.addLayout(opacity_layout)

        # Current frame only
        self.current_frame_only_check = QCheckBox("Show current frame only")
        self.current_frame_only_check.setChecked(True)
        layout.addWidget(self.current_frame_only_check)

        group.setLayout(layout)
        return group

    def _setup_camera_group(self) -> QGroupBox:
        """Setup camera preset controls."""
        group = QGroupBox("Camera")
        layout = QVBoxLayout()

        # Preset buttons
        preset_layout = QHBoxLayout()
        self.camera_button_group = QButtonGroup(self)

        presets = [("XY", "xy"), ("XZ", "xz"), ("YZ", "yz"), ("Iso", "isometric")]
        for i, (label, preset_id) in enumerate(presets):
            btn = QRadioButton(label)
            btn.setProperty("preset_id", preset_id)
            if i == 3:  # Isometric default
                btn.setChecked(True)
            self.camera_button_group.addButton(btn, i)
            preset_layout.addWidget(btn)

        self.camera_button_group.buttonClicked.connect(self._on_camera_preset_clicked)
        layout.addLayout(preset_layout)

        group.setLayout(layout)
        return group

    def _setup_time_group(self) -> QGroupBox:
        """Setup time navigation controls."""
        group = QGroupBox("Time Navigation")
        layout = QVBoxLayout()

        # Frame slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 100)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        slider_layout.addWidget(self.frame_slider)
        self.frame_label = QLabel("0 / 100")
        slider_layout.addWidget(self.frame_label)
        layout.addLayout(slider_layout)

        # Play controls
        play_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        play_layout.addWidget(self.play_button)

        play_layout.addWidget(QLabel("Speed:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setSuffix(" fps")
        play_layout.addWidget(self.fps_spin)

        layout.addLayout(play_layout)

        group.setLayout(layout)
        return group

    def set_data(
        self,
        image: np.ndarray | None = None,
        tracks: pd.DataFrame | None = None,
        detections: pd.DataFrame | None = None,
        voxel_size: VoxelSize | None = None,
    ) -> None:
        """Store tracking metadata and set the time-slider range.

        Image, points, and tracks layers must already be added to the viewer.
        This method does not create layers or update their data or scale.

        Parameters
        ----------
        image : np.ndarray | None
            4D image data (T, Z, Y, X), used only to set the time-slider range
        tracks : pd.DataFrame | None
            Tracks DataFrame
        detections : pd.DataFrame | None
            Reserved argument; currently unused
        voxel_size : VoxelSize | None
            Physical voxel dimensions
        """
        from pt3d.postprocess import compute_track_stats

        self._voxel_size = voxel_size
        self._current_tracks = tracks

        # Compute track statistics if tracks provided
        if tracks is not None and voxel_size is not None and len(tracks) > 0:
            self._current_stats = compute_track_stats(tracks, voxel_size)

        # Update frame slider range based on image data
        if image is not None and image.ndim >= 4:
            self._max_frames = image.shape[0] - 1
            self.frame_slider.setRange(0, self._max_frames)
            self.frame_label.setText(f"0 / {self._max_frames}")

    def enter_3d_mode(self) -> None:
        """Switch napari viewer to 3D display mode."""
        self.viewer.dims.ndisplay = 3
        self._is_3d_mode = True
        self._apply_volume_settings()
        self._apply_track_settings()
        self._apply_points_settings()
        self._apply_camera_preset("isometric")
        self.mode_3d_check.setChecked(True)

    def exit_3d_mode(self) -> None:
        """Return to 2D slice display mode."""
        self.viewer.dims.ndisplay = 2
        self._is_3d_mode = False
        self.mode_3d_check.setChecked(False)

        # Reset image rendering to default
        for layer in self.viewer.layers:
            import napari.layers

            if isinstance(layer, napari.layers.Image):
                layer.rendering = "mip"

    def _on_3d_mode_toggled(self, state: int) -> None:
        """Handle 3D mode checkbox state change."""
        # Qt.CheckState.Checked is 2 for both Qt5 and Qt6
        if state == 2:
            self.enter_3d_mode()
        else:
            self.exit_3d_mode()

    def _get_volume_config(self) -> Volume3DConfig:
        """Get current volume rendering configuration."""
        from pt3d.config import Volume3DConfig

        gamma = self.gamma_slider.value() / 100.0
        opacity = self.volume_opacity_slider.value() / 100.0

        return Volume3DConfig(
            rendering_mode=self.rendering_combo.currentText(),  # type: ignore[arg-type]
            contrast_percentile_low=self.contrast_low_spin.value(),
            contrast_percentile_high=self.contrast_high_spin.value(),
            gamma=gamma,
            opacity=opacity,
            colormap=self.volume_colormap_combo.currentText(),
        )

    def _get_track_config(self) -> Track3DConfig:
        """Get current track visualization configuration."""
        from pt3d.config import Track3DConfig

        color_by_map = {0: "track_id", 1: "time", 2: "length", 3: "velocity"}
        color_by = color_by_map.get(self.track_color_by_combo.currentIndex(), "track_id")

        return Track3DConfig(
            colormap=self.track_colormap_combo.currentText(),
            color_by=color_by,  # type: ignore[arg-type]
            tail_length=self.trail_slider.value(),
            show_current_position=self.show_position_check.isChecked(),
        )

    def _get_points_config(self) -> Points3DConfig:
        """Get current points visualization configuration."""
        from pt3d.config import Points3DConfig

        opacity = self.points_opacity_slider.value() / 100.0

        return Points3DConfig(
            size=float(self.points_size_spin.value()),
            face_color=self.points_color_combo.currentText(),
            opacity=opacity,
            show_current_frame_only=self.current_frame_only_check.isChecked(),
        )

    def _apply_volume_settings(self) -> None:
        """Apply current volume rendering settings to image layers."""
        import napari.layers

        from pt3d.napari.layers import configure_3d_image_layer

        config = self._get_volume_config()

        # Update labels
        self.gamma_label.setText(f"{config.gamma:.1f}")
        self.volume_opacity_label.setText(f"{config.opacity:.1f}")

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                configure_3d_image_layer(layer, config)

    def _apply_track_settings(self) -> None:
        """Apply current track visualization settings."""
        import napari.layers

        from pt3d.napari.layers import configure_3d_tracks_layer

        config = self._get_track_config()

        # Update labels
        self.trail_label.setText(f"{config.tail_length} frames")

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Tracks):
                configure_3d_tracks_layer(layer, config, self._current_tracks, self._current_stats)

    def _apply_points_settings(self) -> None:
        """Apply current points visualization settings."""
        import napari.layers

        from pt3d.napari.layers import configure_3d_points_layer

        config = self._get_points_config()

        # Update labels
        self.points_opacity_label.setText(f"{config.opacity:.1f}")

        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Points):
                configure_3d_points_layer(layer, config)

    def _apply_camera_preset(self, preset: str) -> None:
        """Apply camera preset to viewer."""
        from pt3d.napari.layers import get_camera_preset

        preset_info = get_camera_preset(preset)
        angles = preset_info["angles"]

        # Apply camera angles
        self.viewer.camera.angles = angles

    def _on_volume_settings_changed(self) -> None:
        """Handle volume settings change."""
        if self._is_3d_mode:
            self._apply_volume_settings()

    def _on_track_settings_changed(self) -> None:
        """Handle track settings change."""
        if self._is_3d_mode:
            self._apply_track_settings()

    def _on_points_settings_changed(self) -> None:
        """Handle points settings change."""
        if self._is_3d_mode:
            self._apply_points_settings()

    def _on_camera_preset_clicked(self) -> None:
        """Handle camera preset button click."""
        button = self.camera_button_group.checkedButton()
        if button is not None:
            preset = button.property("preset_id")
            if isinstance(preset, str):
                self._apply_camera_preset(preset)

    def _on_frame_changed(self, value: int) -> None:
        """Handle frame slider value change."""
        # Update label
        self.frame_label.setText(f"{value} / {self._max_frames}")

        # Update viewer's current step for time dimension
        current_step = list(self.viewer.dims.current_step)
        if len(current_step) > 0:
            current_step[0] = value
            self.viewer.dims.current_step = tuple(current_step)

    def _toggle_play(self) -> None:
        """Toggle play/pause of time animation."""
        if self._play_timer is None:
            # Start playing
            self._play_timer = QTimer()
            self._play_timer.timeout.connect(self._advance_frame)
            interval = int(1000 / self.fps_spin.value())
            self._play_timer.start(interval)
            self.play_button.setText("Stop")
        else:
            # Stop playing
            self._play_timer.stop()
            self._play_timer = None
            self.play_button.setText("Play")

    def _advance_frame(self) -> None:
        """Advance to next frame during playback."""
        current = self.frame_slider.value()
        next_frame = (current + 1) % (self._max_frames + 1)
        self.frame_slider.setValue(next_frame)

    def _apply_visualization(self) -> None:
        """Apply all current visualization settings."""
        if self._is_3d_mode:
            self._apply_volume_settings()
            self._apply_track_settings()
            self._apply_points_settings()
        self.visualization_updated.emit()


class MainWidget(QWidget):
    """Combine detection, tracking, visualization, and track-statistics export.

    Detection and tracking run synchronously. Full pipeline export of
    detections, tracks, configuration, and run metadata is available through
    pt3d.pipeline, not through this widget.
    """

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
