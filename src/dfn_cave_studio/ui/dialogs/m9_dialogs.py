"""Operational M9 dialogs; all scientific work is delegated to services."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings
from dfn_cave_studio.services.m9_service import M9Service
from dfn_cave_studio.ui.qt_adapter import (
    Qt,
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QThreadPool,
    QVBoxLayout,
)
from dfn_cave_studio.workers.m9_worker import M9Worker


class _M9Dialog(QDialog):
    """Transactional base: Cancel restores M9 state and workflow."""

    def __init__(self, project, workflow, parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.workflow = workflow
        self.service = M9Service(project)
        self._state_before = deepcopy(project.m9_state)
        self._workflow_before = workflow.to_dict()
        self.committed_changes = False
        self._worker = None

    def reject(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.project.m9_state = self._state_before
        self.workflow.from_dict(self._workflow_before)
        super().reject()

    def _buttons(self) -> QDialogButtonBox:
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        return box

    def _fail(self, message: str) -> None:
        QMessageBox.critical(self, "M9 operation failed", message)


class M9DensityDialog(_M9Dialog):
    """Configure and calculate calibration P10/P32."""

    def __init__(self, project, workflow, parent=None) -> None:
        super().__init__(project, workflow, parent)
        self.setWindowTitle("M9 Fracture Density Model")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.interval_mode = QComboBox()
        self.interval_mode.addItems(["fixed", "domain"])
        self.interval_length = QDoubleSpinBox()
        self.interval_length.setRange(0.1, 10000)
        self.interval_length.setValue(project.m9_state.density_settings.interval_length)
        self.method = QComboBox()
        self.method.addItems([DensityMethod.GLOBAL_CONSTANT.value, DensityMethod.IDW.value])
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(project.m9_state.random_seed)
        current = project.m9_state.density_settings
        self.interval_mode.setCurrentText(current.interval_mode)
        self.method.setCurrentText(current.method.value)
        self.power = QDoubleSpinBox(); self.power.setRange(0.1, 10); self.power.setValue(current.power)
        self.radius = QDoubleSpinBox(); self.radius.setRange(0, 1e9); self.radius.setSpecialValueText("Unlimited"); self.radius.setValue(current.search_radius or 0)
        self.min_neighbors = QSpinBox(); self.min_neighbors.setRange(1, 1000); self.min_neighbors.setValue(current.min_neighbors)
        self.max_neighbors = QSpinBox(); self.max_neighbors.setRange(1, 1000); self.max_neighbors.setValue(current.max_neighbors)
        self.anisotropy_x = QDoubleSpinBox(); self.anisotropy_x.setRange(0.001, 1e6); self.anisotropy_x.setValue(current.anisotropy_x)
        self.anisotropy_y = QDoubleSpinBox(); self.anisotropy_y.setRange(0.001, 1e6); self.anisotropy_y.setValue(current.anisotropy_y)
        self.anisotropy_z = QDoubleSpinBox(); self.anisotropy_z.setRange(0.001, 1e6); self.anisotropy_z.setValue(current.anisotropy_z)
        self.global_fallback = QCheckBox("Use labelled domain-global fallback"); self.global_fallback.setChecked(current.global_fallback)
        self.monte_carlo_samples = QSpinBox()
        self.monte_carlo_samples.setRange(100, 10_000_000)
        self.monte_carlo_samples.setValue(current.monte_carlo_samples)
        self.low_observability_threshold = QDoubleSpinBox()
        self.low_observability_threshold.setDecimals(4)
        self.low_observability_threshold.setRange(0.0001, 1.0)
        self.low_observability_threshold.setValue(current.low_observability_threshold)
        form.addRow("Interval mode", self.interval_mode)
        form.addRow("Interval length (m)", self.interval_length)
        form.addRow("Spatial model", self.method)
        form.addRow("Random seed", self.seed)
        form.addRow("IDW power", self.power)
        form.addRow("Search radius (m)", self.radius)
        form.addRow("Min neighbours", self.min_neighbors)
        form.addRow("Max neighbours", self.max_neighbors)
        form.addRow("Distance scale X", self.anisotropy_x)
        form.addRow("Distance scale Y", self.anisotropy_y)
        form.addRow("Distance scale Z", self.anisotropy_z)
        form.addRow(self.global_fallback)
        form.addRow("Fisher Monte Carlo samples", self.monte_carlo_samples)
        form.addRow("Low-observability threshold", self.low_observability_threshold)
        layout.addLayout(form)
        self.calculate_button = QPushButton("Calculate P10 / P32")
        self.calculate_button.clicked.connect(self._calculate)
        layout.addWidget(self.calculate_button)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Domain", "Set", "N", "Raw L", "Effective L", "P32", "Observability"])
        layout.addWidget(self.table)
        layout.addWidget(self._buttons())
        self._refresh()

    def _calculate(self) -> None:
        try:
            settings = DensitySettings(
                interval_mode=self.interval_mode.currentText(),
                interval_length=self.interval_length.value(),
                method=self.method.currentText(),
                power=self.power.value(),
                search_radius=self.radius.value() or None,
                min_neighbors=self.min_neighbors.value(),
                max_neighbors=self.max_neighbors.value(),
                anisotropy_x=self.anisotropy_x.value(),
                anisotropy_y=self.anisotropy_y.value(),
                anisotropy_z=self.anisotropy_z.value(),
                global_fallback=self.global_fallback.isChecked(),
                monte_carlo_samples=self.monte_carlo_samples.value(),
                low_observability_threshold=self.low_observability_threshold.value(),
            )
            self.project.m9_state.random_seed = self.seed.value()
            self.calculate_button.setEnabled(False)
            self.progress.setVisible(True)
            self._worker = M9Worker(
                lambda progress, cancelled: self.service.calculate_density(
                    settings,
                    progress=progress,
                    cancelled=cancelled,
                )
            )
            self._worker.signals.progress.connect(lambda current, total: self.progress.setRange(0, total))
            self._worker.signals.progress.connect(lambda current, total: self.progress.setValue(current))
            self._worker.signals.finished.connect(self._calculation_done)
            self._worker.signals.cancelled.connect(self._calculation_cancelled)
            self._worker.signals.failed.connect(self._calculation_failed)
            QThreadPool.globalInstance().start(self._worker)
        except Exception as exc:
            self._fail(str(exc))

    def _calculation_done(self, _result) -> None:
        self._worker = None
        self.calculate_button.setEnabled(True)
        self.progress.setVisible(False)
        self.committed_changes = True
        self._refresh()

    def _calculation_cancelled(self) -> None:
        self._worker = None
        self.calculate_button.setEnabled(True)
        self.progress.setVisible(False)

    def _calculation_failed(self, message: str) -> None:
        self._calculation_cancelled()
        self._fail(message)

    def _refresh(self) -> None:
        state = self.project.m9_state
        calibration = sum(item.role == "calibration" for item in state.p10_intervals)
        validation = sum(item.role == "validation" for item in state.p10_intervals)
        self.summary.setText(f"P10 intervals: calibration {calibration}, validation {validation}; P32 estimates {len(state.p32_estimates)}")
        self.table.setRowCount(len(state.p32_estimates))
        for row, estimate in enumerate(state.p32_estimates):
            values = [estimate.domain_id, estimate.set_id, estimate.fracture_count, f"{estimate.raw_sample_length:.3f}", f"{estimate.effective_sample_length:.3f}", estimate.p32, estimate.observability.value]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem("" if value is None else str(value)))

    def accept(self) -> None:
        if self._worker is not None:
            self._fail("Wait for density calculation to finish or cancel the dialog")
            return
        if self.committed_changes:
            self.workflow.complete_step("density")
            self.workflow.mark_ready("size")
        super().accept()


class M9SizeDialog(_M9Dialog):
    """Fit measured sizes or define explicitly assumed size parameters."""

    def __init__(self, project, workflow, parent=None) -> None:
        super().__init__(project, workflow, parent)
        self.setWindowTitle("M9 Fracture Size Distribution")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Automatic fitting is allowed only for radius, diameter, trace_length, or mapped_length."))
        limitation = QLabel(
            "Automatic truncated-distribution fits are EXPERIMENTAL: sample extrema are used as truncation "
            "bounds, and the truncated lognormal is not a full truncated-likelihood optimization."
        )
        limitation.setWordWrap(True)
        layout.addWidget(limitation)
        fit_row = QHBoxLayout()
        self.measurement_field = QComboBox(); self.measurement_field.addItems(["radius", "diameter", "trace_length", "mapped_length"])
        self.fit_button = QPushButton("Fit real calibration measurements")
        self.fit_button.clicked.connect(self._fit)
        fit_row.addWidget(self.measurement_field); fit_row.addWidget(self.fit_button)
        layout.addLayout(fit_row)
        row = QHBoxLayout()
        self.distribution = QComboBox()
        self.distribution.addItems(["fixed", "uniform", "truncated_lognormal", "truncated_power_law", "truncated_exponential"])
        self.lower = QDoubleSpinBox(); self.lower.setRange(0.0001, 1e6); self.lower.setValue(1.0)
        self.upper = QDoubleSpinBox(); self.upper.setRange(0.0001, 1e6); self.upper.setValue(5.0)
        self.manual_source = QComboBox(); self.manual_source.addItems(["user_defined", "assumed"])
        row.addWidget(self.distribution); row.addWidget(self.lower); row.addWidget(self.upper); row.addWidget(self.manual_source)
        layout.addLayout(row)
        self.apply_button = QPushButton("Apply user-defined size model")
        self.apply_button.clicked.connect(self._apply)
        layout.addWidget(self.apply_button)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        layout.addWidget(self._buttons())
        self._refresh()

    def _apply(self) -> None:
        dtype = self.distribution.currentText()
        lower, upper = self.lower.value(), self.upper.value()
        if upper <= lower:
            self._fail("Maximum radius must be greater than minimum radius")
            return
        parameters = {
            "fixed": {"radius": lower},
            "uniform": {},
            "truncated_lognormal": {"mu": float(np.log((lower + upper) / 2)), "sigma": 0.5},
            "truncated_power_law": {"exponent": 2.5},
            "truncated_exponential": {"rate": 2 / (lower + upper)},
        }[dtype]
        try:
            self.service.set_assumed_sizes(
                dtype,
                parameters,
                lower,
                upper,
                user_defined=self.manual_source.currentText() == "user_defined",
            )
            self.committed_changes = True
            self._refresh()
        except Exception as exc:
            self._fail(str(exc))

    def _fit(self) -> None:
        try:
            self.service.fit_sizes(self.measurement_field.currentText())
            self.committed_changes = True
            self._refresh()
        except Exception as exc:
            self._fail(str(exc))

    def _refresh(self) -> None:
        models = self.project.m9_state.size_models
        sources = sorted({item.source.value for item in models})
        experimental = sum(item.fit_status == "experimental" for item in models)
        self.summary.setText(
            f"Size models: {len(models)}; sources: {', '.join(sources) if sources else 'not set'}; "
            f"experimental fits: {experimental}"
        )

    def accept(self) -> None:
        if self.committed_changes:
            self.workflow.complete_step("size")
            self.workflow.mark_ready("parameter_field")
        super().accept()


class M9ParameterFieldDialog(_M9Dialog):
    """Generate and inspect the first voxelized input parameter field."""

    def __init__(self, project, workflow, parent=None) -> None:
        super().__init__(project, workflow, parent)
        self.setWindowTitle("M9 First Voxel Parameter Field")
        self._layer_manager = self._resolve_layer_manager(parent)
        layout = QVBoxLayout(self)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.field = QComboBox()
        self.field.currentTextChanged.connect(self._refresh_slice)
        layout.addWidget(self.field)
        controls = QHBoxLayout()
        self.axis = QComboBox(); self.axis.addItems(["x", "y", "z"])
        self.slice_fraction = QDoubleSpinBox(); self.slice_fraction.setRange(0, 1); self.slice_fraction.setSingleStep(0.05); self.slice_fraction.setValue(0.5)
        self.opacity = QDoubleSpinBox(); self.opacity.setRange(0, 1); self.opacity.setSingleStep(0.05); self.opacity.setValue(0.85)
        self.show_boreholes = QCheckBox("Show boreholes"); self.show_boreholes.setChecked(True)
        self.show_domains = QCheckBox("Show domains"); self.show_domains.setChecked(True)
        self.show_boreholes.toggled.connect(lambda visible: self._set_overlay_visibility("borehole", visible))
        self.show_domains.toggled.connect(lambda visible: self._set_overlay_visibility("domain", visible))
        self.opacity.valueChanged.connect(self._update_current_opacity)
        controls.addWidget(self.axis); controls.addWidget(self.slice_fraction); controls.addWidget(self.opacity)
        controls.addWidget(self.show_boreholes); controls.addWidget(self.show_domains)
        layout.addLayout(controls)
        self.slice_summary = QLabel("No field generated")
        layout.addWidget(self.slice_summary)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        row = QHBoxLayout()
        self.generate_button = QPushButton("Generate parameter field")
        self.cancel_button = QPushButton("Cancel computation")
        self.render_button = QPushButton("Render slice")
        self.export_button = QPushButton("Export M9 package")
        self.screenshot_button = QPushButton("Save PNG")
        self.generate_button.clicked.connect(self._generate)
        self.cancel_button.clicked.connect(self._cancel)
        self.render_button.clicked.connect(self._render_slice)
        self.export_button.clicked.connect(self._export)
        self.screenshot_button.clicked.connect(self._screenshot)
        self.cancel_button.setEnabled(False)
        row.addWidget(self.generate_button); row.addWidget(self.cancel_button); row.addWidget(self.render_button)
        row.addWidget(self.export_button); row.addWidget(self.screenshot_button)
        layout.addLayout(row)
        layout.addWidget(self._create_layer_group())
        layout.addWidget(self._buttons())
        self._refresh()
        self._refresh_layers_table()

    @staticmethod
    def _resolve_layer_manager(parent):
        if parent is None:
            return None
        getter = getattr(parent, "_get_m9_layer_manager", None)
        if getter is not None:
            return getter()
        plotter = getattr(parent, "_plotter", None)
        if plotter is None:
            return None
        from dfn_cave_studio.visualization.m9_layer_manager import M9LayerManager

        return M9LayerManager(plotter)

    def _create_layer_group(self) -> QGroupBox:
        group = QGroupBox("Rendered Layers / 已渲染图层")
        layout = QVBoxLayout(group)
        layout.addWidget(QLabel("Rendered slice layers are session-only display state."))
        self.layers_table = QTableWidget(0, 7)
        self.layers_table.setHorizontalHeaderLabels(
            ["Visible", "Layer name", "Field", "Axis", "Slice", "Coordinate", "Opacity"]
        )
        self.layers_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.layers_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header = self.layers_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.layers_table.itemChanged.connect(self._layer_item_changed)
        layout.addWidget(self.layers_table)
        buttons = QHBoxLayout()
        self.toggle_layer_button = QPushButton("Show/Hide Selected")
        self.remove_layer_button = QPushButton("Remove Selected")
        self.clear_current_button = QPushButton("Clear Current Slice")
        self.clear_all_button = QPushButton("Clear All M9 Layers")
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
            buttons.addWidget(button)
        layout.addLayout(buttons)
        return group

    def _generate(self) -> None:
        try:
            from dfn_cave_studio.voxel.parameter_field import ParameterFieldBuilder

            estimated = ParameterFieldBuilder.estimate_bytes(
                self.project.spatial_grid_config.analysis_domain,
                self.project.voxel_config,
                len(self.project.joint_sets),
            )
            if estimated > ParameterFieldBuilder().memory_warning_bytes:
                QMessageBox.warning(
                    self,
                    "Large parameter field",
                    f"Estimated allocation is {estimated / 1024**3:.2f} GiB. Reduce the grid resolution if needed.",
                )
        except AttributeError:
            self._fail("A confirmed M8 voxel grid is required")
            return
        self.generate_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._worker = M9Worker(lambda progress, cancelled: self.service.build_parameter_field(progress=progress, cancelled=cancelled))
        self._worker.signals.progress.connect(lambda current, total: self.progress.setRange(0, total))
        self._worker.signals.progress.connect(lambda current, total: self.progress.setValue(current))
        self._worker.signals.finished.connect(self._done)
        self._worker.signals.cancelled.connect(self._cancelled)
        self._worker.signals.failed.connect(self._failed)
        QThreadPool.globalInstance().start(self._worker)

    def _done(self, _result) -> None:
        self._worker = None
        self.generate_button.setEnabled(True); self.cancel_button.setEnabled(False)
        self.committed_changes = True
        self._refresh()

    def _cancelled(self) -> None:
        self._worker = None
        self.generate_button.setEnabled(True); self.cancel_button.setEnabled(False)

    def _failed(self, message: str) -> None:
        self._cancelled(); self._fail(message)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _refresh(self) -> None:
        metadata = self.project.m9_state.parameter_field_metadata
        self.field.clear()
        if metadata is None:
            self.summary.setText("Parameter field not generated")
            return
        self.summary.setText(f"Shape {metadata.shape}; {np.prod(metadata.shape):,} voxels; {metadata.estimated_bytes / 1024**2:.2f} MiB")
        self.field.addItems(metadata.field_names)

    def _refresh_slice(self, name: str) -> None:
        array = self.project.m9_state.parameter_field_arrays.get(name)
        if array is None:
            self.slice_summary.setText("No field generated")
            return
        finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array.ravel()
        self.slice_summary.setText(f"{name}: min={finite.min() if finite.size else 'n/a'}, max={finite.max() if finite.size else 'n/a'}; X/Y/Z slices available")

    def _render_slice(self) -> None:
        metadata = self.project.m9_state.parameter_field_metadata
        plotter = getattr(self.parent(), "_plotter", None)
        if metadata is None or plotter is None:
            self._fail("Generate a field and use an available 3D view first")
            return
        if self._layer_manager is None:
            self._fail("An available 3D layer manager is required")
            return
        actor = None
        layer_id = ""
        try:
            from dfn_cave_studio.visualization.parameter_field_renderer import ParameterFieldRenderer

            renderer = ParameterFieldRenderer()
            axis = self.axis.currentText()
            field_name = self.field.currentText()
            slice_index, coordinate = renderer.slice_location(metadata, axis, self.slice_fraction.value())
            layer_id = renderer.layer_id(field_name, axis, slice_index)
            actor = renderer.render_slice(
                plotter,
                metadata,
                self.project.m9_state.parameter_field_arrays,
                field_name,
                axis,
                self.slice_fraction.value(),
                opacity=self.opacity.value(),
                actor_name=layer_id,
            )
            self._layer_manager.add_or_replace(
                layer_id,
                actor,
                field_name=field_name,
                axis=axis,
                slice_index=slice_index,
                coordinate=coordinate,
                opacity=self.opacity.value(),
            )
            if self.show_boreholes.isChecked():
                from dfn_cave_studio.services.m7_state import get_holdout

                renderer.render_borehole_roles(plotter, self.project.borehole_collection, get_holdout(self.project))
            plotter.render()
            self._refresh_layers_table(select_layer_id=layer_id)
        except Exception as exc:
            if actor is not None and layer_id and not self._layer_manager.contains(layer_id):
                remover = getattr(plotter, "remove_actor", None)
                if remover is not None:
                    remover(actor, render=False)
            self._fail(str(exc))

    def _current_layer_id(self) -> str | None:
        metadata = self.project.m9_state.parameter_field_metadata
        if metadata is None or not self.field.currentText():
            return None
        from dfn_cave_studio.visualization.parameter_field_renderer import ParameterFieldRenderer

        slice_index, _ = ParameterFieldRenderer.slice_location(
            metadata,
            self.axis.currentText(),
            self.slice_fraction.value(),
        )
        return ParameterFieldRenderer.layer_id(self.field.currentText(), self.axis.currentText(), slice_index)

    def _selected_layer_id(self) -> str | None:
        row = self.layers_table.currentRow()
        if row < 0:
            return None
        item = self.layers_table.item(row, 1)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _refresh_layers_table(self, select_layer_id: str | None = None) -> None:
        self.layers_table.blockSignals(True)
        try:
            self.layers_table.setRowCount(0)
            records = self._layer_manager.list_layers() if self._layer_manager is not None else []
            selected_row = -1
            for row, record in enumerate(records):
                self.layers_table.insertRow(row)
                visible = QTableWidgetItem()
                visible.setFlags(visible.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                visible.setCheckState(Qt.CheckState.Checked if record.visible else Qt.CheckState.Unchecked)
                name = QTableWidgetItem(record.layer_id)
                name.setData(Qt.ItemDataRole.UserRole, record.layer_id)
                values = [
                    visible,
                    name,
                    QTableWidgetItem(record.field_name),
                    QTableWidgetItem(record.axis.upper()),
                    QTableWidgetItem(str(record.slice_index)),
                    QTableWidgetItem(f"{record.coordinate:.6g}"),
                    QTableWidgetItem(f"{record.opacity:.2f}"),
                ]
                for column, item in enumerate(values):
                    if column != 6:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.layers_table.setItem(row, column, item)
                if record.layer_id == select_layer_id:
                    selected_row = row
            if selected_row >= 0:
                self.layers_table.selectRow(selected_row)
        finally:
            self.layers_table.blockSignals(False)

    def _layer_item_changed(self, item: QTableWidgetItem) -> None:
        if self._layer_manager is None:
            return
        name_item = self.layers_table.item(item.row(), 1)
        layer_id = name_item.data(Qt.ItemDataRole.UserRole) if name_item is not None else None
        if not layer_id:
            return
        try:
            if item.column() == 0:
                self._layer_manager.set_visible(layer_id, item.checkState() == Qt.CheckState.Checked)
            elif item.column() == 6:
                self._layer_manager.set_opacity(layer_id, float(item.text()))
        except (TypeError, ValueError) as exc:
            self._fail(str(exc))
            self._refresh_layers_table(select_layer_id=layer_id)

    def _toggle_selected_layer(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is None or self._layer_manager is None:
            return
        record = self._layer_manager.get(layer_id)
        if record is not None:
            self._layer_manager.set_visible(layer_id, not record.visible)
            self._refresh_layers_table(select_layer_id=layer_id)

    def _remove_selected_layer(self) -> None:
        layer_id = self._selected_layer_id()
        if layer_id is None or self._layer_manager is None:
            return
        self._layer_manager.remove(layer_id)
        self._refresh_layers_table()

    def _clear_current_slice(self) -> None:
        layer_id = self._current_layer_id()
        if layer_id is None or self._layer_manager is None:
            return
        self._layer_manager.remove(layer_id)
        self._refresh_layers_table()

    def _clear_all_layers(self) -> None:
        if self._layer_manager is None:
            return
        self._layer_manager.clear_m9_layers()
        self._refresh_layers_table()

    def _update_current_opacity(self, opacity: float) -> None:
        layer_id = self._current_layer_id()
        if layer_id is None or self._layer_manager is None:
            return
        if self._layer_manager.set_opacity(layer_id, opacity):
            self._refresh_layers_table(select_layer_id=layer_id)

    def _set_overlay_visibility(self, category: str, visible: bool) -> None:
        """Toggle existing borehole/domain actors without changing scientific arrays."""
        plotter = getattr(self.parent(), "_plotter", None)
        renderer = getattr(plotter, "renderer", None)
        actors = getattr(renderer, "actors", {})
        prefixes = ("BH-", "trajectory:", "Obs-", "m9-borehole:") if category == "borehole" else (
            "voxel_analysis_domain",
            "dfn_generation_domain",
        )
        for name, actor in actors.items():
            if str(name).startswith(prefixes) and hasattr(actor, "SetVisibility"):
                actor.SetVisibility(visible)
        if plotter is not None and hasattr(plotter, "render"):
            plotter.render()

    def _export(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Export M9 parameter package")
        if not directory:
            return
        try:
            self.service.export(directory)
        except Exception as exc:
            self._fail(str(exc))

    def _screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save parameter-field screenshot", "parameter_field.png", "PNG (*.png)")
        if not path:
            return
        try:
            from dfn_cave_studio.visualization.parameter_field_renderer import ParameterFieldRenderer

            ParameterFieldRenderer.screenshot(getattr(self.parent(), "_plotter", None), path)
        except Exception as exc:
            self._fail(str(exc))

    def accept(self) -> None:
        if self._worker is not None:
            self._fail("Wait for generation to finish or cancel it")
            return
        if self.committed_changes:
            self.workflow.complete_step("parameter_field")
            self.workflow.mark_ready("validation")
        super().accept()


class M9ValidationDialog(_M9Dialog):
    """Run independent held-out validation and report real metrics."""

    def __init__(self, project, workflow, parent=None) -> None:
        super().__init__(project, workflow, parent)
        self.setWindowTitle("M9 Validation")
        layout = QVBoxLayout(self)
        self.run_button = QPushButton("Run held-out validation")
        self.run_button.clicked.connect(self._run)
        layout.addWidget(self.run_button)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        layout.addWidget(self._buttons())
        self._refresh()

    def _run(self) -> None:
        try:
            self.run_button.setEnabled(False)
            self.progress.setVisible(True)
            self._worker = M9Worker(
                lambda progress, cancelled: self.service.validate(progress=progress, cancelled=cancelled)
            )
            self._worker.signals.progress.connect(lambda current, total: self.progress.setRange(0, total))
            self._worker.signals.progress.connect(lambda current, total: self.progress.setValue(current))
            self._worker.signals.finished.connect(self._validation_done)
            self._worker.signals.cancelled.connect(self._validation_cancelled)
            self._worker.signals.failed.connect(self._validation_failed)
            QThreadPool.globalInstance().start(self._worker)
        except Exception as exc:
            self._fail(str(exc))

    def _validation_done(self, _result) -> None:
        self._worker = None
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)
        self.committed_changes = True
        self._refresh()

    def _validation_cancelled(self) -> None:
        self._worker = None
        self.run_button.setEnabled(True)
        self.progress.setVisible(False)

    def _validation_failed(self, message: str) -> None:
        self._validation_cancelled()
        self._fail(message)

    def _refresh(self) -> None:
        summary = self.project.m9_state.validation_summary
        self.summary.setText(f"{summary.state.value}: valid={summary.valid_interval_count}, no data={summary.no_data_interval_count}, MAE={summary.mae}, RMSE={summary.rmse}, bias={summary.bias}")

    def accept(self) -> None:
        if self._worker is not None:
            self._fail("Wait for validation to finish or cancel the dialog")
            return
        if self.committed_changes:
            self.workflow.complete_step("validation")
        super().accept()
