"""Transactional UI for M11.1 exact second voxelization."""

from __future__ import annotations

from typing import Any

from dfn_cave_studio.services.m11_service import M11SecondVoxelizationService
from dfn_cave_studio.ui.dialog_geometry import fit_dialog_to_screen
from dfn_cave_studio.ui.qt_adapter import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSize,
    QSpinBox,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QThreadPool,
    QVBoxLayout,
    QWidget,
)
from dfn_cave_studio.visualization.m11_layer_manager import M11LayerManager
from dfn_cave_studio.visualization.m11_voxel_renderer import M11DisplayConfig, M11VoxelRenderer
from dfn_cave_studio.voxel.second_voxelization import SecondVoxelizationConfig
from dfn_cave_studio.workers.m11_worker import M11SecondVoxelizationWorker


class M11SecondVoxelizationDialog(QDialog):
    """Compute, inspect, render, and transactionally commit one M11.1 result."""

    def __init__(
        self,
        project: Any,
        workflow: Any,
        layer_manager: M11LayerManager | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.workflow = workflow
        self.layer_manager = layer_manager
        self.service = M11SecondVoxelizationService(project)
        self.renderer = M11VoxelRenderer()
        self.committed_changes = False
        self._pending_result = None
        self._worker: M11SecondVoxelizationWorker | None = None
        self.setWindowTitle("M11.1 Exact Second Voxelization")
        self._build_ui()
        self.setMinimumSize(620, 420)
        fit_dialog_to_screen(self, QSize(980, 760))
        self._populate_realizations()
        self._refresh_layers()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, 1)
        notice = QLabel(
            "Computes exact circle-disk/voxel intersection P32. "
            "Connectivity, percolation, and block analysis are not implemented in M11.1."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)

        controls = QGroupBox("Calculation")
        form = QFormLayout(controls)
        self.realization_combo = QComboBox()
        self.worker_combo = QComboBox()
        for worker_count in (1, 2, 4, 8):
            self.worker_combo.addItem(str(worker_count), worker_count)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 65_536)
        self.batch_size.setValue(65_536)
        form.addRow("M10 realization", self.realization_combo)
        form.addRow("Workers", self.worker_combo)
        form.addRow("Fractures per batch", self.batch_size)
        layout.addWidget(controls)

        actions = QHBoxLayout()
        self.compute_button = QPushButton("Compute Exact Intersections")
        self.cancel_button = QPushButton("Cancel Calculation")
        self.cancel_button.setEnabled(False)
        self.compute_button.clicked.connect(self._start)
        self.cancel_button.clicked.connect(self._cancel)
        actions.addWidget(self.compute_button)
        actions.addWidget(self.cancel_button)
        self.progress = QProgressBar()
        self.status = QLabel("Ready")

        view = QGroupBox("M11.2 Three-dimensional P32 Cloud (m^-1)")
        view_form = QFormLayout(view)
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItem("Voxel Cells", "voxel_cells")
        self.display_mode_combo.addItem("Outer Surface Cloud", "outer_surface")
        self.display_mode_combo.addItem("Orthogonal Section", "orthogonal_section")
        self.display_mode_combo.addItem("Arbitrary Plane", "arbitrary_plane")
        self.display_mode_combo.setCurrentIndex(0)
        self.field_combo = QComboBox()
        self.field_combo.addItem("P32 explicit intersection", "p32_explicit_intersection")
        self.field_combo.addItem("P32 subgrid", "p32_subgrid")
        self.field_combo.addItem("P32 total", "p32_total")
        self.set_combo = QComboBox()
        self.axis_combo = QComboBox()
        for axis in "xyz":
            self.axis_combo.addItem(axis.upper(), axis)
        self.slice_fraction = QDoubleSpinBox()
        self.slice_fraction.setRange(0.0, 1.0)
        self.slice_fraction.setSingleStep(0.05)
        self.slice_fraction.setValue(0.5)
        self.section_coordinate = QLabel("—")
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItem("Exact Cell Colours", "exact")
        self.interpolation_combo.addItem("Smooth Display", "smooth")
        self.interpolation_notice = QLabel(
            "Display interpolation only — scientific voxel values unchanged"
        )
        self.interpolation_notice.setWordWrap(True)
        self.show_grid_lines = QCheckBox("Show Grid Lines")
        self.range_mode_combo = QComboBox()
        self.range_mode_combo.addItem("Global range", "global")
        self.range_mode_combo.addItem("Manual min/max", "manual")
        self.manual_min = QDoubleSpinBox()
        self.manual_max = QDoubleSpinBox()
        for control in (self.manual_min, self.manual_max):
            control.setRange(-1e12, 1e12)
            control.setDecimals(8)
        self.manual_min.setValue(0.0)
        self.manual_max.setValue(1.0)
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItem("Continuous gradient", "continuous")
        self.color_mode_combo.addItem("Discrete contour bands", "discrete")
        self.contour_bands = QSpinBox()
        self.contour_bands.setRange(5, 30)
        self.contour_bands.setValue(10)
        metadata = self.project.m9_state.parameter_field_metadata
        centre = (0.0, 0.0, 0.0)
        if metadata is not None:
            centre = tuple(
                metadata.origin[index] + metadata.shape[index] * metadata.spacing[index] / 2.0
                for index in range(3)
            )
        self.plane_origin = []
        self.plane_normal = []
        plane_origin_row = QHBoxLayout()
        plane_normal_row = QHBoxLayout()
        for index, label in enumerate(("X", "Y", "Z")):
            origin_control = QDoubleSpinBox()
            origin_control.setRange(-1e12, 1e12)
            origin_control.setDecimals(6)
            origin_control.setPrefix(f"{label}=")
            origin_control.setValue(centre[index])
            self.plane_origin.append(origin_control)
            plane_origin_row.addWidget(origin_control)
            normal_control = QDoubleSpinBox()
            normal_control.setRange(-1e6, 1e6)
            normal_control.setDecimals(6)
            normal_control.setPrefix(f"N{label}=")
            normal_control.setValue(1.0 if index == 2 else 0.0)
            self.plane_normal.append(normal_control)
            plane_normal_row.addWidget(normal_control)
        self.interactive_plane = QCheckBox("Interactive plane widget")
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.0, 1.0)
        self.opacity.setValue(1.0)
        self.render_button = QPushButton("Render Cloud")
        self.render_button.clicked.connect(self._render)
        view_form.addRow("Display mode", self.display_mode_combo)
        view_form.addRow("Joint set", self.set_combo)
        view_form.addRow("Field", self.field_combo)
        view_form.addRow("Axis", self.axis_combo)
        view_form.addRow("Slice position", self.slice_fraction)
        view_form.addRow("Section coordinate/index", self.section_coordinate)
        view_form.addRow("Plane origin", plane_origin_row)
        view_form.addRow("Plane normal", plane_normal_row)
        view_form.addRow(self.interactive_plane)
        view_form.addRow("Display interpolation", self.interpolation_combo)
        view_form.addRow(self.interpolation_notice)
        view_form.addRow(self.show_grid_lines)
        view_form.addRow("Colour range", self.range_mode_combo)
        manual_row = QHBoxLayout()
        manual_row.addWidget(self.manual_min)
        manual_row.addWidget(self.manual_max)
        view_form.addRow("Manual min/max", manual_row)
        view_form.addRow("Colour mapping", self.color_mode_combo)
        view_form.addRow("Contour bands", self.contour_bands)
        view_form.addRow("Opacity", self.opacity)
        view_form.addRow(self.render_button)
        inspect_row = QHBoxLayout()
        self.inspect_i = QSpinBox()
        self.inspect_j = QSpinBox()
        self.inspect_k = QSpinBox()
        shape = metadata.shape if metadata is not None else (1, 1, 1)
        for control, limit, label in zip(
            (self.inspect_i, self.inspect_j, self.inspect_k), shape, ("i", "j", "k")
        ):
            control.setRange(0, max(0, limit - 1))
            control.setPrefix(f"{label}=")
            inspect_row.addWidget(control)
        self.inspect_button = QPushButton("Inspect Voxel")
        self.inspect_button.clicked.connect(self._inspect_voxel)
        inspect_row.addWidget(self.inspect_button)
        view_form.addRow("Voxel index", inspect_row)
        self.inspect_label = QLabel("Select a completed result and inspect a voxel.")
        self.inspect_label.setWordWrap(True)
        view_form.addRow(self.inspect_label)
        layout.addWidget(view)

        self.display_mode_combo.currentIndexChanged.connect(self._update_display_controls)
        self.axis_combo.currentIndexChanged.connect(self._update_section_coordinate)
        self.slice_fraction.valueChanged.connect(self._update_section_coordinate)
        self.range_mode_combo.currentIndexChanged.connect(self._update_display_controls)
        self.color_mode_combo.currentIndexChanged.connect(self._update_display_controls)
        self._update_display_controls()

        layers = QGroupBox("Rendered Layers (session-only display state)")
        layers_layout = QVBoxLayout(layers)
        self.layer_table = QTableWidget(0, 11)
        self.layer_table.setHorizontalHeaderLabels(
            [
                "Visible",
                "Layer name",
                "Realization",
                "Display Mode",
                "Exact/Smooth",
                "Field",
                "Joint Set",
                "Axis",
                "Slice",
                "Coordinate",
                "Opacity",
            ]
        )
        layers_layout.addWidget(self.layer_table)
        layer_actions = QHBoxLayout()
        self.toggle_layer_button = QPushButton("Show/Hide Selected")
        self.remove_layer_button = QPushButton("Remove Selected")
        self.clear_current_button = QPushButton("Clear Current Slice")
        self.clear_all_button = QPushButton("Clear All M11 Layers")
        self.toggle_layer_button.clicked.connect(self._toggle_selected_layer)
        self.remove_layer_button.clicked.connect(self._remove_selected_layer)
        self.clear_current_button.clicked.connect(self._clear_current_slice)
        self.clear_all_button.clicked.connect(self._clear_all_layers)
        for button in (
            self.toggle_layer_button,
            self.remove_layer_button,
            self.clear_current_button,
            self.clear_all_button,
        ):
            layer_actions.addWidget(button)
        layers_layout.addLayout(layer_actions)
        layout.addWidget(layers)

        self.opacity.valueChanged.connect(self._update_selected_opacity)

        self.summary = QTableWidget(0, 2)
        self.summary.setHorizontalHeaderLabels(["Metric", "Value"])
        layout.addWidget(self.summary)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        fixed_status = QVBoxLayout()
        fixed_status.addWidget(self.progress)
        self.status.setWordWrap(True)
        fixed_status.addWidget(self.status)
        fixed_actions = QHBoxLayout()
        fixed_actions.addWidget(self.compute_button)
        fixed_actions.addWidget(self.cancel_button)
        fixed_actions.addStretch(1)
        fixed_actions.addWidget(self.buttons)
        fixed_status.addLayout(fixed_actions)
        outer.addLayout(fixed_status)

    def _populate_realizations(self) -> None:
        self.realization_combo.clear()
        for realization in self.project.m10_state.realizations:
            if realization.complete:
                self.realization_combo.addItem(
                    f"{realization.realization_id} ({realization.fracture_count:,} fractures)", realization.realization_id
                )
        metadata = self.project.m9_state.parameter_field_metadata
        self.set_combo.clear()
        self.set_combo.addItem("All sets", None)
        if metadata is not None:
            for set_id in metadata.set_ids:
                self.set_combo.addItem(f"Joint Set {set_id}", int(set_id))
        self._refresh_summary(self._current_result())

    def _start(self) -> None:
        realization_id = self.realization_combo.currentData()
        if realization_id is None or self._worker is not None:
            return
        if self._pending_result is not None and self.layer_manager is not None:
            self.layer_manager.clear_pending(self._pending_result.realization_id)
            self._pending_result = None
        config = SecondVoxelizationConfig(
            worker_count=int(self.worker_combo.currentData()),
            fracture_batch_size=self.batch_size.value(),
        )
        worker = M11SecondVoxelizationWorker(self.project, str(realization_id), config)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)
        worker.signals.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self._set_running(True)
        QThreadPool.globalInstance().start(worker)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancellation requested; stopping calculation and worker processes...")

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(current)
        self.status.setText(message)

    def _on_finished(self, result: Any) -> None:
        self._pending_result = result
        self._worker = None
        self._set_running(False)
        self.status.setText(
            f"Complete: {result.positive_intersection_count:,} positive pairs from "
            f"{result.candidate_pair_count:,} candidates. Press OK to commit."
        )
        self._refresh_summary(result)

    def _on_failed(self, message: str) -> None:
        self._worker = None
        self._set_running(False)
        QMessageBox.critical(self, "M11 Second Voxelization Failed", message)
        self.status.setText("Failed; project state was not changed.")

    def _on_cancelled(self) -> None:
        self._worker = None
        self._set_running(False)
        self.status.setText("Cancelled; no partial result was committed.")

    def _set_running(self, running: bool) -> None:
        from dfn_cave_studio.ui.i18n import language_manager

        language_manager().set_busy(f"m11-voxelization-{id(self)}", running)
        self.compute_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.buttons.setEnabled(not running)

    def _current_result(self):
        if self._pending_result is not None:
            return self._pending_result
        realization_id = self.realization_combo.currentData()
        return self.project.m11_state.result_for(str(realization_id)) if realization_id is not None else None

    def _refresh_summary(self, result: Any) -> None:
        if result is None:
            self.summary.setRowCount(0)
            return
        conservation = result.conservation
        rejection = (
            result.rejected_candidate_count / result.candidate_pair_count if result.candidate_pair_count else 0.0
        )
        values = [
            ("Candidate pairs", f"{result.candidate_pair_count:,}"),
            ("Positive intersections", f"{result.positive_intersection_count:,}"),
            ("Candidate rejection ratio", f"{rejection:.3%}"),
            ("Exact area in generation domain", f"{conservation.target_area_total:.9g} m^2"),
            ("Exact area in analysis domain", f"{conservation.analysis_domain_target_area_total:.9g} m^2"),
            ("Area assigned to analysis voxels", f"{conservation.intersection_area_total:.9g} m^2"),
            ("Maximum per-fracture error", f"{conservation.maximum_absolute_error:.3g} m^2"),
            ("Over-tolerance fractures", str(conservation.over_tolerance_count)),
            ("Sparse result bytes", f"{result.sparse_bytes:,}"),
            ("All M11 array bytes", f"{result.total_array_bytes:,}"),
        ]
        self.summary.setRowCount(len(values))
        for row, (name, value) in enumerate(values):
            self.summary.setItem(row, 0, QTableWidgetItem(name))
            self.summary.setItem(row, 1, QTableWidgetItem(value))

    def _render(self) -> None:
        result = self._current_result()
        metadata = self.project.m9_state.parameter_field_metadata
        if result is None or metadata is None or self.layer_manager is None:
            return
        field_name, set_id = self._current_field()
        try:
            self.renderer.render(
                self.layer_manager,
                metadata,
                result,
                field_name,
                set_id,
                self._display_config(),
                pending=result is self._pending_result,
            )
            self._refresh_layers()
        except Exception as exc:
            QMessageBox.critical(self, "M11 Slice Render Failed", str(exc))

    def _selected_layer_id(self) -> str | None:
        row = self.layer_table.currentRow()
        if row < 0:
            return None
        item = self.layer_table.item(row, 0)
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def _refresh_layers(self) -> None:
        records = [] if self.layer_manager is None else self.layer_manager.list_layers()
        self.layer_table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = (
                "Yes" if record.visible else "No",
                record.layer_id,
                record.realization_id,
                record.display_mode,
                record.interpolation_mode,
                record.field_name,
                "All" if record.joint_set_id is None else str(record.joint_set_id),
                record.axis.upper(),
                str(record.slice_index),
                f"{record.coordinate:.6g}",
                f"{record.opacity:.3f}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record.layer_id)
                self.layer_table.setItem(row, column, item)

    def _toggle_selected_layer(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is None or self.layer_manager is None:
            return
        record = self.layer_manager.get(layer_id)
        if record is not None:
            self.layer_manager.set_visible(layer_id, not record.visible)
            self._refresh_layers()

    def _remove_selected_layer(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is None or self.layer_manager is None:
            return
        self.layer_manager.remove(layer_id)
        self._refresh_layers()

    def _clear_current_slice(self) -> None:
        result = self._current_result()
        metadata = self.project.m9_state.parameter_field_metadata
        if result is None or metadata is None or self.layer_manager is None:
            return
        field_name, set_id = self._current_field()
        try:
            prepared = self.renderer.prepare_display(
                metadata, result, field_name, set_id, self._display_config()
            )
        except ValueError:
            return
        self.layer_manager.remove(prepared.layer_id)
        self._refresh_layers()

    def _clear_all_layers(self) -> None:
        if self.layer_manager is None:
            return
        self.layer_manager.clear_m11_layers()
        self._refresh_layers()

    def _update_selected_opacity(self, opacity: float) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is None or self.layer_manager is None:
            return
        if self.layer_manager.set_opacity(layer_id, opacity):
            self._refresh_layers()

    def _current_field(self) -> tuple[str, int | None]:
        base = str(self.field_combo.currentData())
        set_id = self.set_combo.currentData()
        return (base if set_id is None else f"{base}_set_{set_id}", set_id)

    def _display_config(self) -> M11DisplayConfig:
        return M11DisplayConfig(
            display_mode=self.display_mode_combo.currentData(),
            interpolation_mode=self.interpolation_combo.currentData(),
            axis=str(self.axis_combo.currentData()),
            fraction=self.slice_fraction.value(),
            plane_origin=tuple(control.value() for control in self.plane_origin),
            plane_normal=tuple(control.value() for control in self.plane_normal),
            interactive_plane=self.interactive_plane.isChecked(),
            show_grid_lines=self.show_grid_lines.isChecked(),
            range_mode=self.range_mode_combo.currentData(),
            manual_min=self.manual_min.value(),
            manual_max=self.manual_max.value(),
            color_mode=self.color_mode_combo.currentData(),
            contour_bands=self.contour_bands.value(),
            opacity=self.opacity.value(),
        )

    def _update_display_controls(self) -> None:
        mode = self.display_mode_combo.currentData()
        orthogonal = mode == "orthogonal_section"
        arbitrary = mode == "arbitrary_plane"
        self.axis_combo.setEnabled(orthogonal)
        self.slice_fraction.setEnabled(orthogonal)
        for control in (*self.plane_origin, *self.plane_normal):
            control.setEnabled(arbitrary)
        self.interactive_plane.setEnabled(arbitrary)
        manual = self.range_mode_combo.currentData() == "manual"
        self.manual_min.setEnabled(manual)
        self.manual_max.setEnabled(manual)
        self.contour_bands.setEnabled(self.color_mode_combo.currentData() == "discrete")
        self._update_section_coordinate()

    def _update_section_coordinate(self) -> None:
        metadata = self.project.m9_state.parameter_field_metadata
        if metadata is None or self.display_mode_combo.currentData() != "orthogonal_section":
            self.section_coordinate.setText("—")
            return
        index, coordinate = self.renderer.slice_location(
            metadata, str(self.axis_combo.currentData()), self.slice_fraction.value()
        )
        self.section_coordinate.setText(f"{coordinate:.6g} m (index {index})")

    def _inspect_voxel(self) -> None:
        result = self._current_result()
        metadata = self.project.m9_state.parameter_field_metadata
        if result is None or metadata is None:
            return
        index = (self.inspect_i.value(), self.inspect_j.value(), self.inspect_k.value())
        set_id = self.set_combo.currentData()
        suffix = "" if set_id is None else f"_set_{set_id}"
        explicit = float(result.arrays[f"p32_explicit_intersection{suffix}"][index])
        subgrid = float(result.arrays[f"p32_subgrid{suffix}"][index])
        total = float(result.arrays[f"p32_total{suffix}"][index])
        count = int(result.arrays[f"intersecting_fracture_count{suffix}"][index])
        state = int(result.arrays["cell_state"][index])
        self.inspect_label.setText(
            f"Voxel {index}: state_code={state}; intersection area-derived P32={explicit:.6g} m^-1; "
            f"subgrid={subgrid:.6g} m^-1; total={total:.6g} m^-1; intersecting fractures={count}."
        )

    def accept(self) -> None:
        if self._worker is not None:
            return
        if self._pending_result is not None:
            self.service.commit(self._pending_result)
            self.committed_changes = True
            self.workflow.complete_step("second_voxelization")
            if self.layer_manager is not None:
                self.layer_manager.mark_committed(self._pending_result.realization_id)
        if self.layer_manager is not None:
            self.layer_manager.clear_widgets()
        self.renderer.clear_cache()
        super().accept()

    def reject(self) -> None:
        if self._worker is not None:
            self._cancel()
            return
        if self._pending_result is not None and self.layer_manager is not None:
            self.layer_manager.clear_pending(self._pending_result.realization_id)
        self._pending_result = None
        if self.layer_manager is not None:
            self.layer_manager.clear_widgets()
        self.renderer.clear_cache()
        super().reject()
