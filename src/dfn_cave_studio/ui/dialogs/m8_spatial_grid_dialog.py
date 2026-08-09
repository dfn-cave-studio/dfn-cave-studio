"""M8 model-boundary, voxel-grid, and lightweight-preview dialog."""

from __future__ import annotations

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.services.spatial_domain_service import SpatialDomainService
from dfn_cave_studio.ui.qt_adapter import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class M8SpatialGridDialog(QDialog):
    """Define and validate the two M8 domains before allocating any grid."""

    def __init__(self, project, plotter=None, mode: str = "voxel", parent=None):
        super().__init__(parent)
        if mode not in {"bounds", "voxel"}:
            raise ValueError("mode must be 'bounds' or 'voxel'")
        self._project = project
        self._plotter = plotter
        self._dialog_mode = mode
        self._accepted_config = None
        self.setWindowTitle("M8 Model Boundary" if mode == "bounds" else "M8 Voxel Grid Preview and Confirmation")
        self.resize(720, 680)
        self._build_ui()
        self._load_project()
        self._bounds_group.setEnabled(self._dialog_mode == "bounds")
        self._voxel_group.setEnabled(self._dialog_mode == "voxel")
        self._coverage_button.setEnabled(self._dialog_mode == "bounds")
        self._preview_button.setEnabled(self._dialog_mode == "voxel" and self._plotter is not None)
        self._update_summary()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._bounds_group = QGroupBox("Voxel Analysis Domain")
        mode_layout = QVBoxLayout(self._bounds_group)
        mode_form = QFormLayout()
        self._mode = QComboBox()
        self._mode.addItem("Automatic from full trajectories and observations", "auto")
        self._mode.addItem("Manual", "manual")
        self._mode.currentIndexChanged.connect(self._mode_changed)
        mode_form.addRow("Boundary mode:", self._mode)
        self._margin = self._double(0.0, 1_000_000.0, 10.0)
        mode_form.addRow("Automatic outward margin (m):", self._margin)
        mode_layout.addLayout(mode_form)
        bounds_grid = QGridLayout()
        self._bounds: dict[str, QDoubleSpinBox] = {}
        for row, axis in enumerate(("x", "y", "z")):
            minimum = self._double(-1e9, 1e9, 10.0)
            maximum = self._double(-1e9, 1e9, 10.0)
            self._bounds[f"{axis}_min"] = minimum
            self._bounds[f"{axis}_max"] = maximum
            bounds_grid.addWidget(QLabel(axis.upper()), row, 0)
            bounds_grid.addWidget(QLabel("min"), row, 1)
            bounds_grid.addWidget(minimum, row, 2)
            bounds_grid.addWidget(QLabel("max"), row, 3)
            bounds_grid.addWidget(maximum, row, 4)
        mode_layout.addLayout(bounds_grid)
        self._clip = QCheckBox("Keep insufficient manual bounds and explicitly allow clipping")
        mode_layout.addWidget(self._clip)
        layout.addWidget(self._bounds_group)

        self._voxel_group = QGroupBox("Voxel and DFN Generation Domain")
        form = QFormLayout(self._voxel_group)
        self._dx = self._double(0.001, 1e6, 5.0)
        self._dy = self._double(0.001, 1e6, 5.0)
        self._dz = self._double(0.001, 1e6, 5.0)
        form.addRow("dx (m):", self._dx)
        form.addRow("dy (m):", self._dy)
        form.addRow("dz (m):", self._dz)
        self._layers = QSpinBox()
        self._layers.setRange(0, 10_000)
        self._layers.setValue(2)
        form.addRow("Voxel buffer layers:", self._layers)
        self._radius = self._double(0.0, 1e6, 0.0)
        form.addRow("Maximum fracture radius (m):", self._radius)
        self._wireframe = QCheckBox("Show sampled grid wireframe")
        self._wireframe.setChecked(True)
        form.addRow(self._wireframe)
        self._opacity = self._double(0.0, 1.0, 0.35)
        self._opacity.setSingleStep(0.05)
        form.addRow("Grid-line opacity:", self._opacity)
        self._slice_axis = QComboBox()
        self._slice_axis.addItems(["x", "y", "z"])
        self._slice_axis.setCurrentText("z")
        form.addRow("Preview slice axis:", self._slice_axis)
        self._slice_percent = QSpinBox()
        self._slice_percent.setRange(0, 100)
        self._slice_percent.setValue(50)
        form.addRow("Preview slice position (%):", self._slice_percent)
        layout.addWidget(self._voxel_group)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        actions = QHBoxLayout()
        self._coverage_button = QPushButton("Check Coverage")
        self._coverage_button.clicked.connect(self._check_coverage)
        actions.addWidget(self._coverage_button)
        self._preview_button = QPushButton("3D Preview")
        self._preview_button.clicked.connect(self._preview)
        self._preview_button.setEnabled(self._plotter is not None)
        actions.addWidget(self._preview_button)
        layout.addLayout(actions)
        for widget in [*self._bounds.values(), self._margin, self._dx, self._dy, self._dz, self._layers, self._radius]:
            widget.valueChanged.connect(self._update_summary)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _double(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setValue(value)
        return spin

    def _load_project(self) -> None:
        config = self._project.spatial_grid_config
        bounds = config.analysis_domain if config is not None else self._project.model_bounds
        for key, spin in self._bounds.items():
            spin.setValue(getattr(bounds, key))
        voxel = self._project.voxel_config
        self._dx.setValue(voxel.cell_size_x)
        self._dy.setValue(voxel.cell_size_y)
        self._dz.setValue(voxel.cell_size_z)
        if config is not None:
            self._mode.setCurrentIndex(0 if config.boundary_mode == "auto" else 1)
            self._margin.setValue(config.outward_margin)
            self._layers.setValue(config.buffer_layers)
            self._radius.setValue(config.maximum_fracture_radius)
            self._clip.setChecked(config.clipping_acknowledged)
            self._opacity.setValue(config.preview_opacity)
            self._slice_axis.setCurrentText(config.preview_slice_axis or "z")
            self._slice_percent.setValue(round(config.preview_slice_fraction * 100))
        self._mode_changed()

    def _mode_changed(self, *_args) -> None:
        automatic = self._mode.currentData() == "auto"
        for spin in self._bounds.values():
            spin.setEnabled(not automatic)
        if automatic and len(self._project.borehole_collection):
            try:
                bounds = SpatialDomainService.automatic_bounds(self._project.borehole_collection, self._margin.value())
                for key, spin in self._bounds.items():
                    spin.setValue(getattr(bounds, key))
            except ValueError:
                pass
        self._update_summary()

    def _analysis_bounds(self) -> ModelBounds:
        if self._dialog_mode == "voxel":
            return self._project.model_bounds
        if self._mode.currentData() == "auto" and len(self._project.borehole_collection):
            return SpatialDomainService.automatic_bounds(self._project.borehole_collection, self._margin.value())
        return ModelBounds(**{key: spin.value() for key, spin in self._bounds.items()})

    def _voxel(self) -> VoxelConfig:
        return VoxelConfig(cell_size_x=self._dx.value(), cell_size_y=self._dy.value(), cell_size_z=self._dz.value())

    def _update_summary(self, *_args) -> None:
        try:
            bounds = self._analysis_bounds()
            voxel = self._voxel()
            active_voxels = SpatialDomainService.box_mask_active_voxels(
                bounds, voxel, getattr(self._project, "rock_mask", None)
            )
            summary = SpatialDomainService.grid_summary(bounds, voxel, active_voxels=active_voxels)
            generation = SpatialDomainService.generation_domain(
                bounds, voxel, self._radius.value(), self._layers.value()
            )
            self._summary.setText(
                f"Grid: {summary.nx} × {summary.ny} × {summary.nz} = {summary.total_voxels:,} voxels; "
                f"active voxels {summary.active_voxels if summary.active_voxels is not None else 'mask unavailable'}; "
                f"estimated principal fields {summary.estimated_bytes / 1024**2:.2f} MiB. "
                f"Cell state distinguishes no-data, true zero, and outside-model. "
                f"Generation domain: X[{generation.x_min:.2f}, {generation.x_max:.2f}], "
                f"Y[{generation.y_min:.2f}, {generation.y_max:.2f}], "
                f"Z[{generation.z_min:.2f}, {generation.z_max:.2f}]."
                + (f" WARNING: {summary.warning}" if summary.warning else "")
            )
        except ValueError as error:
            self._summary.setText(str(error))

    def _check_coverage(self):
        try:
            report = SpatialDomainService.check_bounds(self._analysis_bounds(), self._project.borehole_collection)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid boundary", str(error))
            return None
        message = (
            f"Outside boreholes: {report.outside_borehole_count}\n"
            f"Outside trajectory points: {report.outside_trajectory_point_count}\n"
            f"Outside observation points: {report.outside_observation_point_count}\n"
            f"Maximum exceedance: {report.maximum_exceedance}\n"
            f"Affected holes: {', '.join(report.affected_holes) or 'none'}"
        )
        QMessageBox.information(self, "Boundary Coverage", message)
        return report

    def _preview(self) -> None:
        from dfn_cave_studio.visualization.m8_spatial_preview import M8SpatialPreviewRenderer

        bounds = self._analysis_bounds()
        voxel = self._voxel()
        generation = SpatialDomainService.generation_domain(bounds, voxel, self._radius.value(), self._layers.value())
        M8SpatialPreviewRenderer.render(
            self._plotter,
            bounds,
            generation,
            voxel,
            self._project.borehole_collection,
            opacity=self._opacity.value(),
            show_sampled_wireframe=self._wireframe.isChecked(),
            slice_axis=self._slice_axis.currentText(),
            slice_fraction=self._slice_percent.value() / 100.0,
        )

    def _accept(self) -> None:
        try:
            bounds = self._analysis_bounds()
            report = SpatialDomainService.check_bounds(bounds, self._project.borehole_collection)
            if report.has_violations and not self._clip.isChecked():
                answer = QMessageBox.question(
                    self,
                    "Boundary is insufficient",
                    "Some data lies outside. Expand to the recommended boundary?\n"
                    "Choose No to return and explicitly enable clipping.",
                )
                if answer == QMessageBox.StandardButton.Yes:
                    bounds = report.recommended_domain
                    for key, spin in self._bounds.items():
                        spin.setValue(getattr(bounds, key))
                else:
                    return
            self._accepted_config = SpatialDomainService.build_config(
                bounds,
                self._voxel(),
                self._mode.currentData(),
                self._margin.value(),
                self._layers.value(),
                self._radius.value(),
                clipping_acknowledged=self._clip.isChecked(),
            )
            self._accepted_config.preview_opacity = self._opacity.value()
            self._accepted_config.preview_slice_axis = self._slice_axis.currentText()
            self._accepted_config.preview_slice_fraction = self._slice_percent.value() / 100.0
        except ValueError as error:
            QMessageBox.warning(self, "Invalid spatial settings", str(error))
            return
        self.accept()

    def get_config(self):
        """Return the accepted persisted spatial configuration."""
        return self._accepted_config

    def get_voxel_config(self) -> VoxelConfig:
        """Return accepted voxel spacing."""
        return self._voxel()
