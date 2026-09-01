"""Non-modal dock controls for committed M11 P32 visualization results."""

from __future__ import annotations

from typing import Any

import numpy as np

from dfn_cave_studio.models.spatial_grid import VoxelCellState
from dfn_cave_studio.services.m11_service import M11SecondVoxelizationService
from dfn_cave_studio.ui.qt_adapter import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSignalBlocker,
    QSlider,
    QSpinBox,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QTimer,
    QVBoxLayout,
    QWidget,
)
from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager
from dfn_cave_studio.visualization.m11_voxel_renderer import M11DisplayConfig, M11VoxelRenderer
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES


class M11VisualizationPanel(QDockWidget):
    """Dockable, session-only visualization controls for committed M11 results."""

    def __init__(self, layer_manager: M11LayerManager | None, parent=None) -> None:
        super().__init__("M11 Visualization", parent)
        self.layer_manager = layer_manager
        self.renderer = M11VoxelRenderer()
        self.project = None
        self.workflow = None
        self._axis_controls: dict[str, dict[str, Any]] = {}
        self._axis_timers: dict[str, QTimer] = {}
        self._syncing_axes: set[str] = set()
        self._syncing_arbitrary = False
        self._syncing_cutaway = False
        self._syncing_box = False
        self._box_interaction_active = False
        self._arbitrary_timer = QTimer(self)
        self._arbitrary_timer.setSingleShot(True)
        self._arbitrary_timer.setInterval(80)
        self._arbitrary_timer.timeout.connect(lambda: self._render_arbitrary(False, False))
        self._cutaway_timer = QTimer(self)
        self._cutaway_timer.setSingleShot(True)
        self._cutaway_timer.setInterval(80)
        self._cutaway_timer.timeout.connect(lambda: self._render_cutaway(False, False))
        self._box_timer = QTimer(self)
        self._box_timer.setSingleShot(True)
        self._box_timer.setInterval(100)
        self._box_timer.timeout.connect(lambda: self._render_box_cutaway(False, False))
        self._build_ui()
        self.visibilityChanged.connect(self._on_visibility_changed)

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        self.status_label = QLabel("Open a project with a completed M11 result.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.controls = QWidget()
        controls_layout = QVBoxLayout(self.controls)
        selection = QGroupBox("Committed M11 Result")
        selection_form = QFormLayout(selection)
        self.realization_combo = QComboBox()
        self.field_combo = QComboBox()
        self.field_combo.addItem("P32 explicit intersection", "p32_explicit_intersection")
        self.field_combo.addItem("P32 subgrid", "p32_subgrid")
        self.field_combo.addItem("P32 total", "p32_total")
        self.set_combo = QComboBox()
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItem("Exact Cell Colours", "exact")
        self.interpolation_combo.addItem("Smooth Display", "smooth")
        self.grid_lines = QCheckBox("Show Grid Lines")
        self.range_mode = QComboBox()
        self.range_mode.addItem("Global Range", "global")
        self.range_mode.addItem("Manual Range", "manual")
        self.range_min = self._number(-1e12, 1e12)
        self.range_max = self._number(-1e12, 1e12)
        self.range_max.setValue(1.0)
        self.color_mode = QComboBox()
        self.color_mode.addItem("Continuous Gradient", "continuous")
        self.color_mode.addItem("Discrete Bands", "discrete")
        self.bands = QSpinBox()
        self.bands.setRange(5, 30)
        self.bands.setValue(10)
        selection_form.addRow("Realization", self.realization_combo)
        selection_form.addRow("Field", self.field_combo)
        selection_form.addRow("Joint Set", self.set_combo)
        selection_form.addRow("Display", self.interpolation_combo)
        selection_form.addRow(self.grid_lines)
        selection_form.addRow("Colour range", self.range_mode)
        range_row = QHBoxLayout()
        range_row.addWidget(self.range_min)
        range_row.addWidget(self.range_max)
        selection_form.addRow("Manual min/max", range_row)
        selection_form.addRow("Colour mapping", self.color_mode)
        selection_form.addRow("Bands", self.bands)
        note = QLabel("Display interpolation only — scientific voxel values unchanged")
        note.setWordWrap(True)
        selection_form.addRow(note)
        controls_layout.addWidget(selection)

        surface = QGroupBox("Outer Surface Cloud")
        surface_layout = QHBoxLayout(surface)
        self.surface_button = QPushButton("Render / Update Surface")
        self.surface_remove = QPushButton("Remove Surface")
        surface_layout.addWidget(self.surface_button)
        surface_layout.addWidget(self.surface_remove)
        controls_layout.addWidget(surface)

        sections = QGroupBox("Orthogonal Sections")
        sections_layout = QVBoxLayout(sections)
        for axis in "xyz":
            row = QHBoxLayout()
            enabled = QCheckBox(axis.upper())
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(500)
            coordinate = self._number(-1e12, 1e12)
            coordinate.setSuffix(" m")
            handle = QCheckBox("Handle")
            handle.setChecked(True)
            snapshot = QPushButton("Keep Snapshot")
            position = QLabel("centre")
            row.addWidget(enabled)
            row.addWidget(slider, 1)
            row.addWidget(coordinate)
            row.addWidget(handle)
            row.addWidget(snapshot)
            row.addWidget(position)
            sections_layout.addLayout(row)
            self._axis_controls[axis] = {
                "enabled": enabled,
                "slider": slider,
                "coordinate": coordinate,
                "handle": handle,
                "snapshot": snapshot,
                "position": position,
            }
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(80)
            timer.timeout.connect(lambda axis=axis: self._render_axis(axis, create_widget=False))
            self._axis_timers[axis] = timer
            enabled.toggled.connect(lambda checked, axis=axis: self._toggle_axis(axis, checked))
            slider.valueChanged.connect(lambda value, axis=axis: self._axis_slider_changed(axis, value))
            coordinate.valueChanged.connect(lambda value, axis=axis: self._axis_coordinate_changed(axis, value))
            handle.toggled.connect(lambda visible, axis=axis: self._set_axis_handle_visible(axis, visible))
            snapshot.clicked.connect(lambda _=False, axis=axis: self._render_axis(axis, snapshot=True))
        self.reset_sections = QPushButton("Reset XYZ to Centre")
        self.reset_sections.clicked.connect(self._reset_sections)
        sections_layout.addWidget(self.reset_sections)
        controls_layout.addWidget(sections)

        arbitrary = QGroupBox("Arbitrary Section")
        arbitrary_form = QFormLayout(arbitrary)
        self.arbitrary_enabled = QCheckBox("Enable arbitrary section")
        self.arbitrary_handle = QCheckBox("Show plane handle")
        self.arbitrary_handle.setChecked(True)
        self.arbitrary_origin = [self._number(-1e12, 1e12) for _ in range(3)]
        self.arbitrary_normal = [self._number(-1e6, 1e6) for _ in range(3)]
        self.arbitrary_normal[2].setValue(1.0)
        origin_row = QHBoxLayout()
        normal_row = QHBoxLayout()
        for control in self.arbitrary_origin:
            origin_row.addWidget(control)
        for control in self.arbitrary_normal:
            normal_row.addWidget(control)
        self.render_arbitrary = QPushButton("Render / Update Arbitrary Section")
        self.snapshot_arbitrary = QPushButton("Keep Snapshot")
        self.reset_arbitrary = QPushButton("Reset")
        arbitrary_orientations = QHBoxLayout()
        self.arbitrary_orientations: dict[str, QPushButton] = {}
        for axis in "xyz":
            button = QPushButton(axis.upper())
            self.arbitrary_orientations[axis] = button
            arbitrary_orientations.addWidget(button)
        arbitrary_form.addRow(self.arbitrary_enabled, self.arbitrary_handle)
        arbitrary_form.addRow("Origin X/Y/Z", origin_row)
        arbitrary_form.addRow("Normal Nx/Ny/Nz", normal_row)
        arbitrary_form.addRow(self.render_arbitrary, self.snapshot_arbitrary)
        arbitrary_form.addRow(self.reset_arbitrary, arbitrary_orientations)
        controls_layout.addWidget(arbitrary)

        cutaway = QGroupBox("Cutaway / Clip Plane")
        cutaway_form = QFormLayout(cutaway)
        self.cutaway_enabled = QCheckBox("Enable cutaway")
        self.cutaway_handle = QCheckBox("Show plane handle")
        self.cutaway_handle.setChecked(True)
        self.cutaway_flip = QCheckBox("Flip Side")
        self.cutaway_origin = [self._number(-1e12, 1e12) for _ in range(3)]
        self.cutaway_normal = [self._number(-1e6, 1e6) for _ in range(3)]
        self.cutaway_normal[0].setValue(1.0)
        cutaway_origin_row = QHBoxLayout()
        cutaway_normal_row = QHBoxLayout()
        for control in self.cutaway_origin:
            cutaway_origin_row.addWidget(control)
        for control in self.cutaway_normal:
            cutaway_normal_row.addWidget(control)
        self.render_cutaway = QPushButton("Render / Update Cutaway")
        self.snapshot_cutaway = QPushButton("Keep Snapshot")
        self.reset_cutaway = QPushButton("Reset")
        cutaway_orientations = QHBoxLayout()
        self.cutaway_orientations: dict[str, QPushButton] = {}
        for axis in "xyz":
            button = QPushButton(axis.upper())
            self.cutaway_orientations[axis] = button
            cutaway_orientations.addWidget(button)
        cutaway_form.addRow(self.cutaway_enabled, self.cutaway_handle)
        cutaway_form.addRow("Origin X/Y/Z", cutaway_origin_row)
        cutaway_form.addRow("Normal Nx/Ny/Nz", cutaway_normal_row)
        cutaway_form.addRow(self.cutaway_flip)
        cutaway_form.addRow(self.render_cutaway, self.snapshot_cutaway)
        cutaway_form.addRow(self.reset_cutaway, cutaway_orientations)
        cutaway_note = QLabel("Display clipping only — scientific calculation domain and P32 arrays are unchanged.")
        cutaway_note.setWordWrap(True)
        cutaway_form.addRow(cutaway_note)
        controls_layout.addWidget(cutaway)

        clipping = QGroupBox("Display Clipping Box — retains box interior")
        clipping_form = QFormLayout(clipping)
        self.box_enabled = QCheckBox("Enable Box Cutaway")
        self.box_handle = QCheckBox("Show Box Handle")
        self.box_handle.setChecked(True)
        self.box_snap = QCheckBox("Snap Box to Voxel Faces")
        self.box_snap.setChecked(True)
        self.box_bounds = [self._number(-1e12, 1e12) for _ in range(6)]
        for control in self.box_bounds:
            control.setDecimals(12)
        for labels, start in (("X min/max", 0), ("Y min/max", 2), ("Z min/max", 4)):
            row = QHBoxLayout()
            row.addWidget(self.box_bounds[start])
            row.addWidget(self.box_bounds[start + 1])
            clipping_form.addRow(labels, row)
        self.render_box = QPushButton("Render / Update Box Cutaway")
        self.snapshot_box = QPushButton("Keep Snapshot")
        self.reset_box_model = QPushButton("Reset to Model Bounds")
        self.reset_box_centre = QPushButton("Reset to Centre")
        clipping_form.addRow(self.box_enabled, self.box_handle)
        clipping_form.addRow(self.box_snap)
        clipping_form.addRow(self.render_box, self.snapshot_box)
        clipping_form.addRow(self.reset_box_model, self.reset_box_centre)
        clipping_note = QLabel(
            "Volume-first display cutaway. Exact+Snap shows source voxel faces; continuous cuts hide triangulation "
            "edges. Scientific calculation range and P32 arrays are unchanged."
        )
        clipping_note.setWordWrap(True)
        clipping_form.addRow(clipping_note)
        controls_layout.addWidget(clipping)

        layers = QGroupBox("M11 Rendered Layers")
        layers_layout = QVBoxLayout(layers)
        self.layers_table = QTableWidget(0, 6)
        self.layers_table.setHorizontalHeaderLabels(
            ["Visible", "Mode", "Field", "Set", "Axis/Slot", "Interpolation"]
        )
        layers_layout.addWidget(self.layers_table)
        layer_buttons = QHBoxLayout()
        self.toggle_layer = QPushButton("Show/Hide")
        self.remove_layer = QPushButton("Remove")
        self.clear_current = QPushButton("Clear Current View")
        self.clear_layers = QPushButton("Clear All M11")
        self.layer_opacity = self._number(0.0, 1.0)
        self.layer_opacity.setSingleStep(0.05)
        self.layer_opacity.setValue(1.0)
        layer_buttons.addWidget(self.toggle_layer)
        layer_buttons.addWidget(self.remove_layer)
        layer_buttons.addWidget(self.clear_current)
        layer_buttons.addWidget(self.clear_layers)
        layer_buttons.addWidget(QLabel("Opacity"))
        layer_buttons.addWidget(self.layer_opacity)
        layers_layout.addLayout(layer_buttons)
        controls_layout.addWidget(layers)
        layout.addWidget(self.controls)
        layout.addStretch(1)
        scroll.setWidget(content)
        self.setWidget(scroll)

        self.surface_button.clicked.connect(self._render_surface)
        self.surface_remove.clicked.connect(self._remove_surface)
        self.render_arbitrary.clicked.connect(lambda: self._render_arbitrary(False, True))
        self.snapshot_arbitrary.clicked.connect(lambda: self._render_arbitrary(True, False))
        self.arbitrary_enabled.toggled.connect(self._toggle_arbitrary)
        self.arbitrary_handle.toggled.connect(self._set_arbitrary_handle_visible)
        for control in (*self.arbitrary_origin, *self.arbitrary_normal):
            control.valueChanged.connect(self._arbitrary_inputs_changed)
        self.reset_arbitrary.clicked.connect(self._reset_arbitrary)
        for axis, button in self.arbitrary_orientations.items():
            button.clicked.connect(lambda _=False, axis=axis: self._set_arbitrary_orientation(axis))
        self.render_cutaway.clicked.connect(lambda: self._render_cutaway(False, True))
        self.snapshot_cutaway.clicked.connect(lambda: self._render_cutaway(True, False))
        self.cutaway_enabled.toggled.connect(self._toggle_cutaway)
        self.cutaway_handle.toggled.connect(self._set_cutaway_handle_visible)
        self.cutaway_flip.toggled.connect(self._cutaway_flip_changed)
        for control in (*self.cutaway_origin, *self.cutaway_normal):
            control.valueChanged.connect(self._cutaway_inputs_changed)
        self.reset_cutaway.clicked.connect(self._reset_cutaway)
        for axis, button in self.cutaway_orientations.items():
            button.clicked.connect(lambda _=False, axis=axis: self._set_cutaway_orientation(axis))
        self.render_box.clicked.connect(self._activate_box_cutaway)
        self.snapshot_box.clicked.connect(lambda: self._render_box_cutaway(True, False))
        self.box_enabled.toggled.connect(self._toggle_box_cutaway)
        self.box_handle.toggled.connect(self._set_box_handle_visible)
        self.box_snap.toggled.connect(self._box_snap_changed)
        for control in self.box_bounds:
            control.valueChanged.connect(self._box_inputs_changed)
        self.reset_box_model.clicked.connect(self._reset_box_model_bounds)
        self.reset_box_centre.clicked.connect(self._reset_box_to_centre)
        self.grid_lines.toggled.connect(self._box_display_changed)
        self.toggle_layer.clicked.connect(self._toggle_selected)
        self.remove_layer.clicked.connect(self._remove_selected)
        self.clear_current.clicked.connect(self._remove_selected)
        self.clear_layers.clicked.connect(self._clear_all)
        self.layer_opacity.valueChanged.connect(self._set_selected_opacity)
        self.layers_table.itemSelectionChanged.connect(self._selected_layer_changed)
        self.realization_combo.currentIndexChanged.connect(self._realization_changed)

    @staticmethod
    def _number(minimum: float, maximum: float) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(6)
        return control

    def set_context(self, project: Any, workflow: Any) -> None:
        """Rebind the dock to a project and show only committed, valid M11 results."""
        self._clear_interaction_widgets()
        self.renderer.clear_cache()
        self.project = project
        self.workflow = workflow
        self.refresh()

    def refresh(self) -> None:
        """Refresh committed result choices without duplicating callbacks or widgets."""
        previous = self.realization_combo.currentData()
        with QSignalBlocker(self.realization_combo):
            self.realization_combo.clear()
            valid = (
                self.project is not None
                and self.workflow is not None
                and self.workflow.is_step_done("second_voxelization")
            )
            if valid:
                service = M11SecondVoxelizationService(self.project)
                for result in self.project.m11_state.results:
                    if result.complete and service.is_valid(result):
                        self.realization_combo.addItem(result.realization_id, result.realization_id)
            index = self.realization_combo.findData(previous)
            if index >= 0:
                self.realization_combo.setCurrentIndex(index)
        available = self.realization_combo.count() > 0 and self.layer_manager is not None
        if not available and self.layer_manager is not None:
            self.layer_manager.clear_m11_layers()
        self.controls.setEnabled(available)
        self.status_label.setText(
            "Committed M11 result ready. Slice and clipping controls are display-only."
            if available
            else "No valid committed M11 result. Complete M11.1 or recompute stale results first."
        )
        self._populate_sets()
        self._reset_ranges()
        self._refresh_layers()

    def _result(self):
        realization_id = self.realization_combo.currentData()
        if self.project is None or realization_id is None:
            return None
        return next(
            (item for item in self.project.m11_state.results if item.realization_id == realization_id), None
        )

    def _metadata(self):
        return None if self.project is None else self.project.m9_state.parameter_field_metadata

    def _populate_sets(self) -> None:
        previous = self.set_combo.currentData()
        with QSignalBlocker(self.set_combo):
            self.set_combo.clear()
            self.set_combo.addItem("All Sets", None)
            metadata = self._metadata()
            if metadata is not None:
                for set_id in metadata.set_ids:
                    self.set_combo.addItem(f"Joint Set {set_id}", int(set_id))
            index = self.set_combo.findData(previous)
            if index >= 0:
                self.set_combo.setCurrentIndex(index)

    def _field(self) -> tuple[str, int | None]:
        base = str(self.field_combo.currentData())
        set_id = self.set_combo.currentData()
        return (base if set_id is None else f"{base}_set_{set_id}", set_id)

    def _base_config(self, **updates) -> M11DisplayConfig:
        values = {
            "interpolation_mode": self.interpolation_combo.currentData(),
            "show_grid_lines": self.grid_lines.isChecked(),
            "range_mode": self.range_mode.currentData(),
            "manual_min": self.range_min.value(),
            "manual_max": self.range_max.value(),
            "color_mode": self.color_mode.currentData(),
            "contour_bands": self.bands.value(),
        }
        values.update(updates)
        return M11DisplayConfig(**values)

    def _render_config(self, config: M11DisplayConfig) -> Any:
        result, metadata = self._result(), self._metadata()
        if result is None or metadata is None or self.layer_manager is None:
            return None
        field, set_id = self._field()
        try:
            record = self.renderer.render(self.layer_manager, metadata, result, field, set_id, config)
            self._refresh_layers()
            return record
        except Exception as exc:  # noqa: BLE001 - Qt boundary reports VTK/PyVista failures
            QMessageBox.critical(self, "M11 Visualization Failed", str(exc))
            return None

    def _render_surface(self) -> None:
        self._render_config(self._base_config(display_mode="outer_surface", interactive_slot="surface"))

    def _remove_surface(self) -> None:
        record = self._prepared_record(self._base_config(display_mode="outer_surface", interactive_slot="surface"))
        if record is not None and self.layer_manager is not None:
            self.layer_manager.remove(record.layer_id)
            self._refresh_layers()

    def _axis_bounds(self, axis: str) -> tuple[float, float]:
        metadata = self._metadata()
        index = {"x": 0, "y": 1, "z": 2}[axis]
        lower = metadata.origin[index]
        return lower, lower + metadata.shape[index] * metadata.spacing[index]

    def _axis_slider_changed(self, axis: str, value: int) -> None:
        lower, upper = self._axis_bounds(axis)
        coordinate = lower + (upper - lower) * value / 1000.0
        self._sync_axis_position(axis, coordinate)

    def _axis_coordinate_changed(self, axis: str, value: float) -> None:
        self._sync_axis_position(axis, value)

    def _sync_axis_position(self, axis: str, value: float) -> None:
        """Synchronize one orthogonal coordinate across controls, widget, and actor."""
        if axis in self._syncing_axes:
            return
        self._syncing_axes.add(axis)
        try:
            lower, upper = self._axis_bounds(axis)
            value = min(max(float(value), lower), upper)
            coordinate = self._axis_controls[axis]["coordinate"]
            slider = self._axis_controls[axis]["slider"]
            with QSignalBlocker(coordinate):
                coordinate.setValue(value)
            with QSignalBlocker(slider):
                slider.setValue(round(1000 * (value - lower) / (upper - lower)))
            self._update_axis_label(axis)
            record = self._live_record(f"xyz_{axis}", "orthogonal_section")
            if record is not None and self.layer_manager is not None:
                self.layer_manager.set_widget_plane(
                    record.layer_id,
                    self._orthogonal_origin(axis, value),
                    self._axis_normal(axis),
                )
            if self._axis_controls[axis]["enabled"].isChecked():
                self._axis_timers[axis].start()
        finally:
            self._syncing_axes.discard(axis)

    @staticmethod
    def _axis_normal(axis: str) -> tuple[float, float, float]:
        index = {"x": 0, "y": 1, "z": 2}[axis]
        return tuple(1.0 if item == index else 0.0 for item in range(3))

    def _orthogonal_origin(self, axis: str, coordinate: float) -> tuple[float, float, float]:
        metadata = self._metadata()
        origin = [
            metadata.origin[index] + metadata.shape[index] * metadata.spacing[index] / 2.0
            for index in range(3)
        ]
        origin[{"x": 0, "y": 1, "z": 2}[axis]] = float(coordinate)
        return tuple(origin)

    def _live_record(self, slot: str, display_mode: str):
        if self.layer_manager is None:
            return None
        result = self._result()
        if result is None:
            return None
        field, set_id = self._field()
        interpolation = self.interpolation_combo.currentData()
        return next(
            (
                record
                for record in self.layer_manager.list_layers()
                if record.realization_id == result.realization_id
                and record.field_name == field
                and record.joint_set_id == set_id
                and record.display_mode == display_mode
                and record.interpolation_mode == interpolation
                and record.plane_id == f"slot_{slot}"
            ),
            None,
        )

    def _update_axis_label(self, axis: str) -> None:
        metadata = self._metadata()
        coordinate = self._axis_controls[axis]["coordinate"].value()
        index = {"x": 0, "y": 1, "z": 2}[axis]
        floating = (coordinate - metadata.origin[index]) / metadata.spacing[index] - 0.5
        nearest = round(floating)
        suffix = f"voxel {nearest}" if 0 <= nearest < metadata.shape[index] and abs(floating - nearest) < 1e-6 else "continuous"
        self._axis_controls[axis]["position"].setText(f"{coordinate:.4g} m; {suffix}")

    def _toggle_axis(self, axis: str, enabled: bool) -> None:
        if enabled:
            self._render_axis(axis, create_widget=True)
            return
        prepared = self._prepared_record(self._axis_config(axis, create_widget=False))
        if prepared is not None and self.layer_manager is not None:
            self.layer_manager.remove(prepared.layer_id)
            self._refresh_layers()

    def _axis_config(self, axis: str, *, create_widget: bool, snapshot: bool = False) -> M11DisplayConfig:
        coordinate = self._axis_controls[axis]["coordinate"].value()
        return self._base_config(
            display_mode="orthogonal_section",
            axis=axis,
            section_coordinate=coordinate,
            interactive_slot="" if snapshot else f"xyz_{axis}",
            interactive_plane=create_widget and self._axis_controls[axis]["handle"].isChecked(),
            interaction_callback=lambda normal, origin, axis=axis: self._axis_widget_changed(axis, origin),
        )

    def _render_axis(self, axis: str, *, create_widget: bool = False, snapshot: bool = False) -> None:
        if not snapshot and not self._axis_controls[axis]["enabled"].isChecked():
            return
        self._render_config(self._axis_config(axis, create_widget=create_widget, snapshot=snapshot))

    def _axis_widget_changed(self, axis: str, origin: Any) -> None:
        if axis in self._syncing_axes:
            return
        index = {"x": 0, "y": 1, "z": 2}[axis]
        self._sync_axis_position(axis, float(origin[index]))

    def _set_axis_handle_visible(self, axis: str, visible: bool) -> None:
        record = self._live_record(f"xyz_{axis}", "orthogonal_section")
        if record is not None and self.layer_manager is not None:
            self.layer_manager.set_widgets_visible(record.layer_id, visible)

    def _reset_sections(self) -> None:
        for axis, value in zip("xyz", self._model_centre()):
            self._sync_axis_position(axis, value)

    def _render_arbitrary(self, snapshot: bool, create_widget: bool) -> None:
        config = self._base_config(
            display_mode="arbitrary_plane",
            plane_origin=tuple(control.value() for control in self.arbitrary_origin),
            plane_normal=tuple(control.value() for control in self.arbitrary_normal),
            interactive_slot="" if snapshot else "arbitrary",
            interactive_plane=create_widget and self.arbitrary_handle.isChecked(),
            interaction_callback=self._arbitrary_widget_changed,
        )
        self._render_config(config)

    def _arbitrary_widget_changed(self, normal: Any, origin: Any) -> None:
        if self._syncing_arbitrary:
            return
        self._syncing_arbitrary = True
        try:
            for control, value in zip(self.arbitrary_origin, origin):
                with QSignalBlocker(control):
                    control.setValue(float(value))
            for control, value in zip(self.arbitrary_normal, normal):
                with QSignalBlocker(control):
                    control.setValue(float(value))
            if self.arbitrary_enabled.isChecked():
                self._arbitrary_timer.start()
        finally:
            self._syncing_arbitrary = False

    def _arbitrary_inputs_changed(self, _value: float = 0.0) -> None:
        if self._syncing_arbitrary:
            return
        self._syncing_arbitrary = True
        try:
            origin = tuple(control.value() for control in self.arbitrary_origin)
            normal_input = tuple(control.value() for control in self.arbitrary_normal)
            try:
                normal = self.renderer.normalize_plane_normal(normal_input)
                origin = self.renderer._validate_plane_origin(origin)
            except ValueError:
                return
            record = self._live_record("arbitrary", "arbitrary_plane")
            if record is not None and self.layer_manager is not None:
                self.layer_manager.set_widget_plane(record.layer_id, origin, normal)
            if self.arbitrary_enabled.isChecked():
                self._arbitrary_timer.start()
        finally:
            self._syncing_arbitrary = False

    def _set_arbitrary_orientation(self, axis: str) -> None:
        normal = self._axis_normal(axis)
        self._syncing_arbitrary = True
        try:
            for control, value in zip(self.arbitrary_normal, normal):
                with QSignalBlocker(control):
                    control.setValue(value)
        finally:
            self._syncing_arbitrary = False
        self._arbitrary_inputs_changed()

    def _reset_arbitrary(self) -> None:
        centre = self._model_centre()
        self._syncing_arbitrary = True
        try:
            for control, value in zip(self.arbitrary_origin, centre):
                with QSignalBlocker(control):
                    control.setValue(value)
            for control, value in zip(self.arbitrary_normal, (0.0, 0.0, 1.0)):
                with QSignalBlocker(control):
                    control.setValue(value)
        finally:
            self._syncing_arbitrary = False
        self._arbitrary_inputs_changed()

    def _toggle_arbitrary(self, enabled: bool) -> None:
        if enabled:
            self._render_arbitrary(False, True)
            return
        record = self._live_record("arbitrary", "arbitrary_plane")
        if record is not None and self.layer_manager is not None:
            self.layer_manager.remove(record.layer_id)
            self._refresh_layers()

    def _set_arbitrary_handle_visible(self, visible: bool) -> None:
        record = self._live_record("arbitrary", "arbitrary_plane")
        if record is not None and self.layer_manager is not None:
            self.layer_manager.set_widgets_visible(record.layer_id, visible)

    def _render_cutaway(self, snapshot: bool, create_widget: bool) -> None:
        config = self._base_config(
            display_mode="cutaway",
            plane_origin=tuple(control.value() for control in self.cutaway_origin),
            plane_normal=tuple(control.value() for control in self.cutaway_normal),
            flip_side=self.cutaway_flip.isChecked(),
            interactive_slot="" if snapshot else "cutaway",
            interactive_plane=create_widget and self.cutaway_handle.isChecked(),
            interaction_callback=self._cutaway_widget_changed,
        )
        self._render_config(config)

    def _cutaway_widget_changed(self, normal: Any, origin: Any) -> None:
        if self._syncing_cutaway:
            return
        self._syncing_cutaway = True
        try:
            for control, value in zip(self.cutaway_origin, origin):
                with QSignalBlocker(control):
                    control.setValue(float(value))
            for control, value in zip(self.cutaway_normal, normal):
                with QSignalBlocker(control):
                    control.setValue(float(value))
            if self.cutaway_enabled.isChecked():
                self._cutaway_timer.start()
        finally:
            self._syncing_cutaway = False

    def _cutaway_inputs_changed(self, _value: float = 0.0) -> None:
        if self._syncing_cutaway:
            return
        self._syncing_cutaway = True
        try:
            origin = tuple(control.value() for control in self.cutaway_origin)
            normal_input = tuple(control.value() for control in self.cutaway_normal)
            try:
                normal = self.renderer.normalize_plane_normal(normal_input)
                origin = self.renderer._validate_plane_origin(origin)
            except ValueError:
                return
            record = self._live_record("cutaway", "cutaway")
            if record is not None and self.layer_manager is not None:
                self.layer_manager.set_widget_plane(record.layer_id, origin, normal)
            if self.cutaway_enabled.isChecked():
                self._cutaway_timer.start()
        finally:
            self._syncing_cutaway = False

    def _toggle_cutaway(self, enabled: bool) -> None:
        if enabled:
            self._render_cutaway(False, True)
            return
        record = self._live_record("cutaway", "cutaway")
        if record is not None and self.layer_manager is not None:
            self.layer_manager.remove(record.layer_id)
            self._refresh_layers()

    def _set_cutaway_handle_visible(self, visible: bool) -> None:
        record = self._live_record("cutaway", "cutaway")
        if record is not None and self.layer_manager is not None:
            self.layer_manager.set_widgets_visible(record.layer_id, visible)

    def _cutaway_flip_changed(self, _flipped: bool) -> None:
        if self.cutaway_enabled.isChecked():
            self._cutaway_timer.start()

    def _set_cutaway_orientation(self, axis: str) -> None:
        normal = self._axis_normal(axis)
        self._syncing_cutaway = True
        try:
            for control, value in zip(self.cutaway_normal, normal):
                with QSignalBlocker(control):
                    control.setValue(value)
        finally:
            self._syncing_cutaway = False
        self._cutaway_inputs_changed()

    def _reset_cutaway(self) -> None:
        centre = self._model_centre()
        self._syncing_cutaway = True
        try:
            for control, value in zip(self.cutaway_origin, centre):
                with QSignalBlocker(control):
                    control.setValue(value)
            for control, value in zip(self.cutaway_normal, (1.0, 0.0, 0.0)):
                with QSignalBlocker(control):
                    control.setValue(value)
            with QSignalBlocker(self.cutaway_flip):
                self.cutaway_flip.setChecked(False)
        finally:
            self._syncing_cutaway = False
        self._cutaway_inputs_changed()

    def _analysis_bounds(self) -> tuple[float, float, float, float, float, float]:
        return M11VoxelRenderer.analysis_bounds(self._metadata())

    def _model_bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return the effective valid-mask bounds, falling back to the Analysis Domain."""
        metadata, result = self._metadata(), self._result()
        if metadata is None or result is None:
            return self._analysis_bounds()
        states = np.asarray(result.arrays.get("cell_state"))
        hidden_codes = [
            CELL_STATE_CODES[VoxelCellState.OUTSIDE_MODEL],
            CELL_STATE_CODES[VoxelCellState.NO_DATA],
            CELL_STATE_CODES[VoxelCellState.EXCAVATION],
        ]
        valid = ~np.isin(states, hidden_codes)
        indices = np.argwhere(valid)
        if indices.size == 0:
            return self._analysis_bounds()
        low = indices.min(axis=0)
        high = indices.max(axis=0) + 1
        minimum = np.asarray(metadata.origin) + low * np.asarray(metadata.spacing)
        maximum = np.asarray(metadata.origin) + high * np.asarray(metadata.spacing)
        return (
            float(minimum[0]), float(maximum[0]),
            float(minimum[1]), float(maximum[1]),
            float(minimum[2]), float(maximum[2]),
        )

    def _model_centre(self) -> tuple[float, float, float]:
        bounds = self._model_bounds()
        return (
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        )

    def _reset_ranges(self) -> None:
        if self._metadata() is None:
            return
        bounds = self._model_bounds()
        analysis = self._analysis_bounds()
        for axis in range(3):
            for control in self.box_bounds[2 * axis : 2 * axis + 2]:
                control.setRange(analysis[2 * axis], analysis[2 * axis + 1])
        for control, value in zip(self.box_bounds, bounds):
            with QSignalBlocker(control):
                control.setValue(value)
        centre = self._model_centre()
        for control, value in zip(self.arbitrary_origin, centre):
            with QSignalBlocker(control):
                control.setValue(value)
        for control, value in zip(self.cutaway_origin, centre):
            with QSignalBlocker(control):
                control.setValue(value)
        for axis, value in zip("xyz", centre):
            lower, upper = self._axis_bounds(axis)
            with QSignalBlocker(self._axis_controls[axis]["coordinate"]):
                self._axis_controls[axis]["coordinate"].setValue(value)
            with QSignalBlocker(self._axis_controls[axis]["slider"]):
                self._axis_controls[axis]["slider"].setValue(round(1000 * (value - lower) / (upper - lower)))
            self._update_axis_label(axis)

    def _box_config(self, *, create_widget: bool, snapshot: bool = False) -> M11DisplayConfig:
        bounds = self._current_box_bounds(
            apply_snap=self.box_snap.isChecked() and not self._box_interaction_active
        )
        return self._base_config(
            display_mode="box_cutaway",
            box_bounds=bounds,
            snap_box_to_voxel_faces=self.box_snap.isChecked(),
            interactive_slot="" if snapshot else "box_cutaway",
            interactive_plane=create_widget and self.box_handle.isChecked(),
            interaction_callback=self._box_widget_changed,
            interaction_end_callback=self._box_widget_ended,
            box_preview_continuous=self._box_interaction_active,
        )

    def _current_box_bounds(
        self, *, apply_snap: bool | None = None
    ) -> tuple[float, float, float, float, float, float]:
        metadata = self._metadata()
        bounds = M11VoxelRenderer.validate_box_bounds(
            tuple(control.value() for control in self.box_bounds),
            self._analysis_bounds(),
        )
        should_snap = self.box_snap.isChecked() if apply_snap is None else bool(apply_snap)
        if should_snap:
            bounds = M11VoxelRenderer.snap_box_bounds(metadata, bounds)
        return bounds

    def _render_box_cutaway(self, snapshot: bool, create_widget: bool) -> None:
        try:
            bounds = self._current_box_bounds(
                apply_snap=self.box_snap.isChecked() and not self._box_interaction_active
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Box Cutaway", str(exc))
            return
        self._set_box_controls(bounds)
        self._render_config(self._box_config(create_widget=create_widget, snapshot=snapshot))

    def _activate_box_cutaway(self) -> None:
        if self.box_enabled.isChecked():
            self._render_box_cutaway(False, True)
        else:
            self.box_enabled.setChecked(True)

    def _toggle_box_cutaway(self, enabled: bool) -> None:
        if enabled:
            before = len(self.layer_manager.list_layers()) if self.layer_manager is not None else 0
            self._render_box_cutaway(False, True)
            after = len(self.layer_manager.list_layers()) if self.layer_manager is not None else 0
            if after == before and self._box_live_record() is None:
                with QSignalBlocker(self.box_enabled):
                    self.box_enabled.setChecked(False)
            return
        record = self._box_live_record()
        if record is not None and self.layer_manager is not None:
            self.layer_manager.remove(record.layer_id)
            self._refresh_layers()

    def _box_inputs_changed(self, _value: float = 0.0) -> None:
        if self._syncing_box:
            return
        self._box_timer.stop()
        self._box_interaction_active = False
        try:
            bounds = self._current_box_bounds()
        except ValueError:
            return
        self._sync_box(bounds, schedule_render=True, update_widget=True)

    def _box_widget_changed(self, box: Any, widget: Any | None = None) -> None:
        del widget
        if self._syncing_box:
            return
        try:
            bounds = self._box_callback_bounds(box)
            bounds = M11VoxelRenderer.validate_box_bounds(bounds, self._analysis_bounds())
        except ValueError:
            return
        self._box_interaction_active = True
        # Re-placing or snapping vtkBoxWidget during InteractionEvent breaks
        # its active mouse pick.  Keep a continuous preview until the end event.
        self._sync_box(bounds, schedule_render=True, update_widget=False)

    def _box_widget_ended(self, box: Any, widget: Any | None = None) -> None:
        del widget
        if self._syncing_box:
            return
        try:
            bounds = self._box_callback_bounds(box)
            bounds = M11VoxelRenderer.validate_box_bounds(bounds, self._analysis_bounds())
            if self.box_snap.isChecked():
                bounds = M11VoxelRenderer.snap_box_bounds(self._metadata(), bounds)
        except ValueError:
            self._box_interaction_active = False
            return
        self._box_timer.stop()
        self._box_interaction_active = False
        self._sync_box(bounds, schedule_render=False, update_widget=True)
        if self.box_enabled.isChecked():
            # EndInteraction is the final transaction boundary.  It bypasses
            # an older debounce request and renders exactly the final bounds.
            self._render_box_cutaway(False, False)

    @staticmethod
    def _box_callback_bounds(box: Any) -> tuple[float, float, float, float, float, float]:
        values = getattr(box, "bounds", None)
        if values is None:
            raise ValueError("Box widget callback did not provide world bounds")
        return tuple(float(item) for item in values)

    def _sync_box(
        self,
        bounds: tuple[float, float, float, float, float, float],
        *,
        schedule_render: bool,
        update_widget: bool,
    ) -> None:
        if self._syncing_box:
            return
        self._syncing_box = True
        try:
            self._set_box_controls(bounds)
            record = self._box_live_record()
            if update_widget and record is not None and self.layer_manager is not None:
                self.layer_manager.set_widget_bounds(record.layer_id, bounds)
            if schedule_render and self.box_enabled.isChecked():
                self._box_timer.start()
        finally:
            self._syncing_box = False

    def _set_box_controls(self, bounds: tuple[float, float, float, float, float, float]) -> None:
        for control, value in zip(self.box_bounds, bounds):
            with QSignalBlocker(control):
                control.setValue(value)

    def _box_live_record(self):
        if self.layer_manager is None:
            return None
        result = self._result()
        if result is None:
            return None
        field, set_id = self._field()
        interpolation = self.interpolation_combo.currentData()
        return next(
            (
                record
                for record in self.layer_manager.list_layers()
                if record.realization_id == result.realization_id
                and record.field_name == field
                and record.joint_set_id == set_id
                and record.display_mode == "box_cutaway"
                and record.interpolation_mode == interpolation
                and record.plane_id.startswith("slot_box_cutaway_")
            ),
            None,
        )

    def _remove_box_live_records(self) -> None:
        if self.layer_manager is None:
            return
        ids = [
            record.layer_id
            for record in self.layer_manager.list_layers()
            if record.display_mode == "box_cutaway" and record.plane_id.startswith("slot_box_cutaway_")
        ]
        for layer_id in ids:
            self.layer_manager.remove(layer_id)

    def _box_snap_changed(self, _enabled: bool) -> None:
        self._box_timer.stop()
        self._box_interaction_active = False
        try:
            bounds = self._current_box_bounds()
        except ValueError:
            return
        self._set_box_controls(bounds)
        if self.box_enabled.isChecked():
            self._remove_box_live_records()
            self._render_box_cutaway(False, True)

    def _box_display_changed(self, _enabled: bool) -> None:
        if self.box_enabled.isChecked():
            self._box_timer.start()

    def _set_box_handle_visible(self, visible: bool) -> None:
        record = self._box_live_record()
        if record is not None and self.layer_manager is not None:
            self.layer_manager.set_widgets_visible(record.layer_id, visible)

    def _reset_box_model_bounds(self) -> None:
        self._box_timer.stop()
        self._box_interaction_active = False
        bounds = self._analysis_bounds()
        self._sync_box(bounds, schedule_render=False, update_widget=True)
        if self.box_enabled.isChecked():
            self._render_box_cutaway(False, False)

    def _reset_box_to_centre(self) -> None:
        bounds = self._model_bounds()
        centred: list[float] = []
        for axis in range(3):
            low, high = bounds[2 * axis], bounds[2 * axis + 1]
            quarter = (high - low) / 4.0
            centred.extend((low + quarter, high - quarter))
        values = tuple(centred)
        if self.box_snap.isChecked():
            values = M11VoxelRenderer.snap_box_bounds(self._metadata(), values)
        self._sync_box(values, schedule_render=True, update_widget=True)

    def _render_enabled_views(self) -> None:
        surface = self._prepared_record(
            self._base_config(display_mode="outer_surface", interactive_slot="surface")
        )
        if surface is not None and self.layer_manager is not None and self.layer_manager.contains(surface.layer_id):
            self._render_surface()
        for axis in "xyz":
            if self._axis_controls[axis]["enabled"].isChecked():
                self._render_axis(axis, create_widget=False)
        if self.arbitrary_enabled.isChecked():
            self._render_arbitrary(False, False)
        if self.cutaway_enabled.isChecked():
            self._render_cutaway(False, False)
        if self.box_enabled.isChecked():
            self._render_box_cutaway(False, False)

    def _prepared_record(self, config: M11DisplayConfig):
        result, metadata = self._result(), self._metadata()
        if result is None or metadata is None:
            return None
        field, set_id = self._field()
        try:
            return self.renderer.prepare_display(metadata, result, field, set_id, config)
        except ValueError:
            return None

    def _refresh_layers(self) -> None:
        records = [] if self.layer_manager is None else self.layer_manager.list_layers()
        self.layers_table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = (
                "Yes" if record.visible else "No",
                record.display_mode,
                record.field_name,
                "All" if record.joint_set_id is None else str(record.joint_set_id),
                record.plane_id or f"{record.axis}:{record.slice_index}",
                record.interpolation_mode,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record.layer_id)
                self.layers_table.setItem(row, column, item)

    def _selected_layer_id(self) -> str | None:
        row = self.layers_table.currentRow()
        item = self.layers_table.item(row, 0) if row >= 0 else None
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def _toggle_selected(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is None or self.layer_manager is None:
            return
        record = self.layer_manager.get(layer_id)
        if record is not None:
            self.layer_manager.set_visible(layer_id, not record.visible)
            self._refresh_layers()

    def _remove_selected(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is not None and self.layer_manager is not None:
            self.layer_manager.remove(layer_id)
            self._refresh_layers()

    def _selected_layer_changed(self) -> None:
        layer_id = self._selected_layer_id()
        record = None if layer_id is None or self.layer_manager is None else self.layer_manager.get(layer_id)
        if record is not None:
            with QSignalBlocker(self.layer_opacity):
                self.layer_opacity.setValue(record.opacity)

    def _set_selected_opacity(self, value: float) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is not None and self.layer_manager is not None:
            self.layer_manager.set_opacity(layer_id, value)

    def _clear_all(self) -> None:
        if self.layer_manager is not None:
            self.layer_manager.clear_m11_layers()
            self._refresh_layers()

    def _realization_changed(self) -> None:
        self._clear_interaction_widgets()
        self.renderer.clear_cache()
        self._reset_ranges()
        if self.isVisible():
            self._restore_interaction_widgets()

    def _clear_interaction_widgets(self) -> None:
        for timer in (*self._axis_timers.values(), self._arbitrary_timer, self._cutaway_timer, self._box_timer):
            timer.stop()
        self._box_interaction_active = False
        if self.layer_manager is not None:
            self.layer_manager.clear_widgets()
            self.layer_manager.clear_control_widgets()

    def _on_visibility_changed(self, visible: bool) -> None:
        if not visible:
            self._clear_interaction_widgets()
        else:
            self.refresh()
            self._restore_interaction_widgets()

    def _restore_interaction_widgets(self) -> None:
        """Restore enabled handles after reopening the dock without duplicating actors."""
        if not self.controls.isEnabled():
            return
        for axis in "xyz":
            if self._axis_controls[axis]["enabled"].isChecked():
                self._render_axis(axis, create_widget=True)
        if self.arbitrary_enabled.isChecked():
            self._render_arbitrary(False, True)
        if self.cutaway_enabled.isChecked():
            self._render_cutaway(False, True)
        if self.box_enabled.isChecked():
            self._render_box_cutaway(False, True)
