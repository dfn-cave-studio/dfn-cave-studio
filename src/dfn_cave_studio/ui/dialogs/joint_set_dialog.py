"""Joint set manager dialog for configuring fracture population parameters."""

from typing import List, Optional

from dfn_cave_studio.ui.qt_adapter import (
    Qt, QColor, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QDoubleSpinBox, QComboBox, QGroupBox, QSignalBlocker, QSpinBox,
    QDialogButtonBox, QListWidget, QListWidgetItem,
    QSplitter, QTabWidget, QWidget, QColorDialog, QLineEdit,
    QTableWidget, QTableWidgetItem,
)
from dfn_cave_studio.models.fracture_set import (
    JointSetConfig, OrientationDistribution, SizeDistribution,
)
from dfn_cave_studio.models.enums import SizeDistributionType
from dfn_cave_studio.services.joint_set_service import joint_set_color


class JointSetManagerDialog(QDialog):
    """Dialog for creating and editing joint sets with live statistics."""

    def __init__(self, joint_sets: Optional[List[JointSetConfig]] = None,
                 model_volume: float = 1000000.0, parent=None,
                 borehole_collection=None, project=None):
        super().__init__(parent)
        self.setWindowTitle("Joint Set Manager")
        self.resize(1000, 760)
        self._volume = model_volume
        self._borehole_collection = borehole_collection
        self._project = project
        self._phase2a_service = None
        self._pending_global_fit = None
        if project is not None:
            from dfn_cave_studio.services.borehole_fracture_service import BoreholeFractureService

            self._phase2a_service = BoreholeFractureService(project)
        self._sets: List[JointSetConfig] = [item.model_copy(deep=True) for item in (joint_sets or [])]
        self._current_idx = 0
        self._init_ui()

    def _default_set(self, idx: int) -> JointSetConfig:
        return JointSetConfig(
            set_id=idx + 1, name=f"Joint Set {idx + 1}",
            color=joint_set_color(idx + 1),
            orientation=OrientationDistribution(mean_dip_direction=45, mean_dip=60, kappa=30),
            size=SizeDistribution(distribution_type=SizeDistributionType.LOGNORMAL,
                                  lognormal_mu=1.0, lognormal_sigma=0.5,
                                  min_radius=0.5, max_radius=10.0),
            target_p32=0.5,
        )

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: set list
        left = QWidget()
        ll = QVBoxLayout(left)
        self._list = QListWidget()
        self._refresh_list()
        self._list.currentRowChanged.connect(self._on_set_selected)
        ll.addWidget(QLabel("Confirmed Global Joint Sets:"))
        ll.addWidget(self._list)
        self._data_status = QLabel()
        self._data_status.setWordWrap(True)
        ll.addWidget(self._data_status)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(self._add_set)
        del_btn = QPushButton("- Remove")
        del_btn.clicked.connect(self._remove_set)
        btn_row.addWidget(add_btn); btn_row.addWidget(del_btn)
        ll.addLayout(btn_row)
        splitter.addWidget(left)

        # Right: property tabs
        right = QTabWidget()
        self._name_le = QLineEdit()
        self._color_btn = QPushButton()
        self._color_btn.clicked.connect(self._pick_color)
        self._dd_spin = self._make_double(0, 360, 0); self._dd_spin.setValue(45)
        self._dip_spin = self._make_double(0, 90, 0); self._dip_spin.setValue(60)
        self._kappa_spin = self._make_double(0.1, 999, 3); self._kappa_spin.setValue(30)
        self._dist_combo = QComboBox()
        for value in ("lognormal", "power_law", "fixed", "exponential", "truncated_power_law"):
            self._dist_combo.addItem(value, value)
        self._dist_combo.currentIndexChanged.connect(self._update_stats)
        self._mu_spin = self._make_double(0.1, 5, 1); self._mu_spin.setValue(1.0)
        self._sigma_spin = self._make_double(0.1, 3, 1); self._sigma_spin.setValue(0.5)
        self._D_spin = self._make_double(1.5, 5, 1); self._D_spin.setValue(3.0)
        self._min_r_spin = self._make_double(0.01, 100, 1); self._min_r_spin.setValue(0.5)
        self._max_r_spin = self._make_double(0.02, 200, 1); self._max_r_spin.setValue(10.0)
        self._p32_spin = self._make_double(0.01, 100, 2); self._p32_spin.setValue(0.5)
        self._tol_spin = self._make_double(0.01, 1, 2); self._tol_spin.setValue(0.05)
        self._opacity_spin = self._make_double(0, 1, 2); self._opacity_spin.setValue(1.0)

        for sp in [self._dd_spin, self._dip_spin, self._kappa_spin, self._mu_spin,
                    self._sigma_spin, self._D_spin, self._min_r_spin, self._max_r_spin,
                    self._p32_spin, self._tol_spin, self._opacity_spin]:
            sp.valueChanged.connect(self._update_stats)
        self._name_le.textChanged.connect(self._update_stats)

        # Basic tab
        basic_tab = QWidget()
        bf = QFormLayout(basic_tab)
        bf.addRow("Name:", self._name_le)
        bf.addRow("Color:", self._color_btn)
        bf.addRow("Opacity:", self._opacity_spin)
        right.addTab(basic_tab, "Basic")

        # Orientation tab
        orient_tab = QWidget()
        of = QFormLayout(orient_tab)
        self._orientation_form = of
        of.addRow("Mean Dip Direction (°):", self._dd_spin)
        of.addRow("Mean Dip (°):", self._dip_spin)
        of.addRow("Kappa (concentration):", self._kappa_spin)
        self._kappa_status_label = QLabel()
        self._kappa_status_label.setWordWrap(True)
        of.addRow("Kappa source:", self._kappa_status_label)
        self._import_from_obs_btn = QPushButton("从钻孔裂隙观测建立/更新裂隙组")
        self._import_from_obs_btn.setToolTip(
            "Group fracture observations by set_id, compute Fisher statistics, "
            "and populate orientation parameters from borehole data."
        )
        self._import_from_obs_btn.clicked.connect(self._on_import_from_observations)
        if self._complete_borehole_orientation_count() == 0:
            self._import_from_obs_btn.setEnabled(False)
            self._import_from_obs_btn.setToolTip(
                "No complete legacy borehole orientations are available. Use the Phase 2A P/Z fitting entry for "
                "representative orientations."
            )
        of.addRow(self._import_from_obs_btn)
        right.addTab(orient_tab, "Orientation")

        # Size tab
        size_tab = QWidget()
        sf = QFormLayout(size_tab)
        self._size_form = sf
        self._size_status_label = QLabel()
        self._size_status_label.setWordWrap(True)
        self._size_formula_label = QLabel(
            "Lognormal definition: ln(R / 1 m) ~ Normal(μ, σ²). μ and σ are dimensionless. "
            "The ordinary Lognormal is untruncated; min/max in m apply to explicitly bounded distributions."
        )
        self._size_formula_label.setWordWrap(True)
        sf.addRow("Size status:", self._size_status_label)
        sf.addRow(self._size_formula_label)
        sf.addRow("Distribution:", self._dist_combo)
        sf.addRow("Lognormal μ:", self._mu_spin)
        sf.addRow("Lognormal σ:", self._sigma_spin)
        sf.addRow("Power-law D:", self._D_spin)
        sf.addRow("Min Radius (m):", self._min_r_spin)
        sf.addRow("Max Radius (m):", self._max_r_spin)
        right.addTab(size_tab, "Size")

        # Intensity tab
        int_tab = QWidget()
        inf = QFormLayout(int_tab)
        self._intensity_form = inf
        self._intensity_status_label = QLabel()
        self._intensity_status_label.setWordWrap(True)
        inf.addRow("Intensity source:", self._intensity_status_label)
        inf.addRow("Target P32 (m²/m³):", self._p32_spin)
        inf.addRow("Tolerance:", self._tol_spin)
        self._stats_lbl = QLabel()
        inf.addRow("Statistics:", self._stats_lbl)
        right.addTab(int_tab, "Intensity")

        splitter.addWidget(right)
        splitter.setSizes([250, 550])
        layout.addWidget(splitter)
        self._build_imported_representatives_ui(layout)

        if self._sets:
            self._refresh_stats_label(self._sets[0])
        else:
            self._stats_lbl.setText("No confirmed global joint set selected.")

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._on_set_selected(0)
        self._refresh_data_status()
        self._refresh_imported_representatives()

    def _make_double(self, lo, hi, decimals) -> QDoubleSpinBox:
        sb = QDoubleSpinBox(); sb.setRange(lo, hi); sb.setDecimals(decimals); return sb

    def _refresh_list(self) -> None:
        self._list.clear()
        for s in self._sets:
            item = QListWidgetItem(f"  {s.name}")
            item.setForeground(QColor(s.color))
            self._list.addItem(item)

    def _complete_borehole_orientation_count(self) -> int:
        return sum(
            1
            for borehole in (self._borehole_collection or [])
            for observation in borehole.fracture_observations
            if observation.dip_direction is not None and observation.dip is not None
        )

    def _refresh_data_status(self) -> None:
        complete = self._complete_borehole_orientation_count()
        if complete:
            text = f"{complete} complete legacy borehole orientation observation(s) are available."
        elif self._sets:
            text = (
                "No complete legacy borehole orientations are available. Existing confirmed project joint sets "
                "can still be viewed or edited. To fit P/Z representative orientations, use DFN > Generate "
                "Borehole Fracture Realizations."
            )
        else:
            text = (
                "No confirmed joint sets or complete legacy borehole orientations are available. No direction "
                "has been invented. Use the Phase 2A P/Z fitting entry or add a set manually."
            )
        self._data_status.setText(text)

    def _build_imported_representatives_ui(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Imported observation candidates and authoritative mapping")
        group_layout = QVBoxLayout(group)
        self._observation_tabs = QTabWidget()
        self._representative_table = QTableWidget(0, 8)
        self._representative_table.setHorizontalHeaderLabels(
            ["Point", "Source", "Local set", "Dip", "Dip direction", "Joint count", "Spacing (m)", "Role"]
        )
        self._random_table = QTableWidget(0, 6)
        self._random_table.setHorizontalHeaderLabels(
            ["Point", "Source", "Local component", "Spacing (m)", "Joint count", "Role"]
        )
        self._observation_tabs.addTab(self._representative_table, "Imported Local Representatives")
        self._observation_tabs.addTab(self._random_table, "Random Background")
        group_layout.addWidget(self._observation_tabs)

        controls = QHBoxLayout()
        self._global_k_spin = QSpinBox()
        self._global_k_spin.setRange(1, 50)
        self._global_seed_spin = QSpinBox()
        self._global_seed_spin.setRange(-2_147_483_648, 2_147_483_647)
        if self._project is not None:
            config = self._project.borehole_fracture_state.config
            self._global_k_spin.setValue(config.number_of_sets)
            self._global_seed_spin.setValue(config.master_seed)
        self._fit_imported_button = QPushButton("Fit Global Joint Sets from Imported Representatives")
        self._fit_imported_button.clicked.connect(self._fit_imported_representatives)
        controls.addWidget(QLabel("Global K:"))
        controls.addWidget(self._global_k_spin)
        controls.addWidget(QLabel("Seed:"))
        controls.addWidget(self._global_seed_spin)
        controls.addWidget(self._fit_imported_button)
        group_layout.addLayout(controls)
        self._k_explanation = QLabel(
            "K is the number of dominant joint sets for the whole study area. It is not a point-local local_set "
            "count, fracture count, or Poisson parameter. P representatives use joint_num weights; Z uses weight 1."
        )
        self._k_explanation.setWordWrap(True)
        group_layout.addWidget(self._k_explanation)
        self._fit_status = QLabel()
        self._fit_status.setWordWrap(True)
        group_layout.addWidget(self._fit_status)
        self._count_summary = QLabel()
        self._count_summary.setWordWrap(True)
        group_layout.addWidget(self._count_summary)
        self._mapping_table = QTableWidget(0, 6)
        self._mapping_table.setHorizontalHeaderLabels(
            ["Observation", "Point", "Source", "Local set", "Global set", "Weight"]
        )
        group_layout.addWidget(self._mapping_table)
        layout.addWidget(group)

    def _refresh_imported_representatives(self) -> None:
        if self._phase2a_service is None:
            self._fit_imported_button.setEnabled(False)
            self._fit_status.setText("No project observation repository is attached to this dialog.")
            return
        observations = self._phase2a_service.observations.orientation_points()
        candidates = [
            item
            for item in observations
            if item.component_type == "LOCAL_DOMINANT_SET"
            and item.orientation_status == "COMPLETE"
            and item.dip is not None
            and item.dip_direction is not None
        ]
        random_rows = [item for item in observations if item.component_type == "RANDOM_BACKGROUND"]
        self._count_summary.setText(
            f"Imported Local Representatives: {len(candidates)} | Random Background: {len(random_rows)} | "
            f"Confirmed Global Joint Sets: {len(self._sets)} | Selected global K: {self._global_k_spin.value()}"
        )
        self._representative_table.setRowCount(len(candidates))
        for row, item in enumerate(candidates):
            density_role = (
                "P representative — supplies spacing/density"
                if item.source_kind == "POINT_CLOUD"
                else "Z representative — direction only; does not provide density"
            )
            values = (
                item.point_id,
                item.source_kind,
                item.local_set_id,
                item.dip,
                item.dip_direction,
                item.joint_num if item.joint_num is not None else "—",
                item.joint_spacing_m if item.joint_spacing_m is not None else "—",
                density_role,
            )
            for column, value in enumerate(values):
                self._representative_table.setItem(row, column, QTableWidgetItem(str(value)))
        self._random_table.setRowCount(len(random_rows))
        for row, item in enumerate(random_rows):
            values = (
                item.point_id,
                item.source_kind,
                item.local_set_id,
                item.joint_spacing_m if item.joint_spacing_m is not None else "—",
                item.joint_num if item.joint_num is not None else "—",
                "Random background — excluded from K and has no ordinary global_set_id",
            )
            for column, value in enumerate(values):
                self._random_table.setItem(row, column, QTableWidgetItem(str(value)))

        self._fit_imported_button.setEnabled(bool(candidates))
        if self._sets:
            self._pending_global_fit = self._phase2a_service.authoritative_confirmed_fit(self._sets)
            self._fit_status.setText(
                f"Using {len(self._sets)} confirmed global joint set(s). Imported representatives are mapped by "
                "axial orientation; click Fit only to explicitly refit."
            )
        elif candidates:
            self._fit_status.setText(
                f"{len(candidates)} complete P/Z representative(s) are available. Choose global K and fit before "
                "confirming."
            )
        else:
            self._fit_status.setText(
                "No complete imported direction candidates are available. Missing-direction and RANDOM records "
                "are not converted into ordinary global joint sets."
            )
        self._refresh_authoritative_mapping()

    def _fit_imported_representatives(self) -> None:
        if self._phase2a_service is None:
            return
        from dfn_cave_studio.ui.qt_adapter import QMessageBox

        try:
            fit = self._phase2a_service.fit_global_sets(
                self._global_k_spin.value(),
                self._global_seed_spin.value(),
            )
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Cannot fit global joint sets", str(exc))
            return
        existing = {item.set_id: item for item in self._sets}
        fitted_sets: list[JointSetConfig] = []
        for model in fit.sets:
            joint_set = existing.get(model.global_set_id)
            is_new = joint_set is None
            if joint_set is None:
                joint_set = self._default_set(model.global_set_id - 1)
            else:
                joint_set = joint_set.model_copy(deep=True)
            joint_set.set_id = model.global_set_id
            joint_set.name = f"Joint Set {model.global_set_id}"
            joint_set.color = joint_set_color(model.global_set_id)
            joint_set.orientation = OrientationDistribution(
                mean_dip_direction=model.mean_dip_direction,
                mean_dip=model.mean_dip,
                kappa=model.kappa,
            )
            joint_set.provenance["orientation"] = "P/Z representative axial fit"
            joint_set.provenance["kappa_status"] = model.kappa_status
            joint_set.provenance["kappa_meaning"] = (
                "one representative direction cannot estimate dispersion"
                if model.kappa_status == "UNRESOLVED"
                else "dispersion among site representative means; not raw within-set dispersion"
            )
            if is_new:
                joint_set.provenance.update(
                    {
                        "size_status": "UNRESOLVED",
                        "size_meaning": "no trace-length or fracture-size observations were fitted",
                        "p32_status": "DERIVED_LATER_BY_M9",
                        "p32_meaning": "inactive placeholder; Phase 2A spacing is not a P32 estimate",
                    }
                )
            fitted_sets.append(joint_set)
        self._sets = fitted_sets
        self._pending_global_fit = fit
        self._refresh_list()
        if self._sets:
            self._list.setCurrentRow(0)
            # Clearing and repopulating QListWidget can leave currentRow() at
            # zero without emitting currentRowChanged.  Synchronize the editor
            # explicitly so accepting cannot write stale form defaults back
            # over the fitted orientation.
            self._on_set_selected(0)
        self._refresh_data_status()
        self._fit_status.setText(
            f"Fitted {len(fit.sets)} global joint set(s). Review the local-to-global mapping, then click OK to confirm."
        )
        self._count_summary.setText(
            f"Imported Local Representatives: {self._representative_table.rowCount()} | "
            f"Random Background: {self._random_table.rowCount()} | Confirmed Global Joint Sets pending: "
            f"{len(fit.sets)} (strictly selected K={self._global_k_spin.value()})"
        )
        self._refresh_authoritative_mapping()

    def _refresh_authoritative_mapping(self) -> None:
        mappings = [] if self._pending_global_fit is None else self._pending_global_fit.mappings
        self._mapping_table.setRowCount(len(mappings))
        for row, mapping in enumerate(mappings):
            values = (
                mapping.observation_id,
                mapping.point_key,
                mapping.source_kind,
                mapping.local_set_id,
                mapping.global_set_id,
                mapping.sample_weight,
            )
            for column, value in enumerate(values):
                self._mapping_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _pending_fit_matches_sets(self) -> bool:
        if self._pending_global_fit is None or len(self._pending_global_fit.sets) != len(self._sets):
            return False
        models = {item.global_set_id: item for item in self._pending_global_fit.sets}
        for joint_set in self._sets:
            model = models.get(joint_set.set_id)
            if model is None:
                return False
            values = (
                (joint_set.orientation.mean_dip_direction, model.mean_dip_direction),
                (joint_set.orientation.mean_dip, model.mean_dip),
                (joint_set.orientation.kappa, model.kappa),
            )
            if any(abs(left - right) > 1e-10 for left, right in values):
                return False
        return True

    def _on_set_selected(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._sets):
            return
        self._current_idx = idx
        s = self._sets[idx]
        editors = [
            self._name_le,
            self._dd_spin,
            self._dip_spin,
            self._kappa_spin,
            self._mu_spin,
            self._sigma_spin,
            self._D_spin,
            self._min_r_spin,
            self._max_r_spin,
            self._p32_spin,
            self._tol_spin,
            self._opacity_spin,
            self._dist_combo,
        ]
        blockers = [QSignalBlocker(editor) for editor in editors]
        try:
            self._name_le.setText(s.name)
            self._color_btn.setStyleSheet(f"background-color: {s.color}; min-width: 60px;")
            self._dd_spin.setValue(s.orientation.mean_dip_direction)
            self._dip_spin.setValue(s.orientation.mean_dip)
            self._kappa_spin.setValue(s.orientation.kappa)
            self._mu_spin.setValue(s.size.lognormal_mu)
            self._sigma_spin.setValue(s.size.lognormal_sigma)
            self._D_spin.setValue(s.size.power_law_exponent)
            self._min_r_spin.setValue(s.size.min_radius)
            self._max_r_spin.setValue(s.size.max_radius)
            self._p32_spin.setValue(s.target_p32)
            self._tol_spin.setValue(s.p32_tolerance)
            self._opacity_spin.setValue(s.opacity)
            idx_lookup = {
                "lognormal": 0,
                "power_law": 1,
                "fixed": 2,
                "exponential": 3,
                "truncated_power_law": 4,
            }
            self._dist_combo.setCurrentIndex(idx_lookup.get(s.size.distribution_type.value, 0))
        finally:
            blockers.clear()
        self._update_parameter_status_ui(s)
        self._refresh_stats_label(s)

    def _update_parameter_status_ui(self, joint_set: JointSetConfig) -> None:
        """Show provenance without presenting inactive placeholders as fitted parameters."""
        kappa_status = str(joint_set.provenance.get("kappa_status", "LEGACY_OR_MANUAL"))
        kappa_text = {
            "UNRESOLVED": "UNRESOLVED — one representative direction cannot estimate dispersion; fixed mean is used.",
            "SITE_MEAN_DISPERSION": "SITE_MEAN_DISPERSION — dispersion among site means, not raw within-set Kappa.",
            "MEASURED_WITHIN_SET": "Measured from original within-set orientation observations.",
            "MANUAL": "Manually confirmed Kappa.",
            "ASSUMED": "Assumed Kappa.",
        }.get(kappa_status, "Legacy or manually supplied Kappa; provenance was not classified.")
        self._kappa_status_label.setText(kappa_text)
        self._orientation_form.setRowVisible(self._kappa_spin, kappa_status != "UNRESOLVED")

        size_unresolved = joint_set.provenance.get("size_status") == "UNRESOLVED"
        self._size_status_label.setText(
            "UNRESOLVED — no trace-length or size observations; inactive until M9 size modelling."
            if size_unresolved
            else str(joint_set.provenance.get("size_status", "Legacy or user-defined size model"))
        )
        for editor in (
            self._dist_combo,
            self._mu_spin,
            self._sigma_spin,
            self._D_spin,
            self._min_r_spin,
            self._max_r_spin,
        ):
            self._size_form.setRowVisible(editor, not size_unresolved)

        p32_derived_later = joint_set.provenance.get("p32_status") == "DERIVED_LATER_BY_M9"
        self._intensity_status_label.setText(
            "Derived later by M9 — P-site spacing controls Phase 2A group probability only; borehole spacing "
            "controls Poisson total count."
            if p32_derived_later
            else str(joint_set.provenance.get("p32_status", "Explicit GLOBAL_CONSTANT / legacy target"))
        )
        self._intensity_form.setRowVisible(self._p32_spin, not p32_derived_later)
        self._intensity_form.setRowVisible(self._tol_spin, not p32_derived_later)

    def _add_set(self) -> None:
        next_id = max((item.set_id for item in self._sets), default=0) + 1
        s = self._default_set(next_id - 1)
        self._sets.append(s)
        self._refresh_list()
        self._list.setCurrentRow(len(self._sets) - 1)
        self._refresh_data_status()

    def _remove_set(self) -> None:
        if len(self._sets) <= 1:
            return
        del self._sets[self._current_idx]
        self._current_idx = max(0, self._current_idx - 1)
        self._refresh_list()
        self._list.setCurrentRow(self._current_idx)
        self._refresh_data_status()

    def _pick_color(self) -> None:
        s = self._sets[self._current_idx]
        color = QColorDialog.getColor()
        if color.isValid():
            s.color = color.name()
            self._color_btn.setStyleSheet(f"background-color: {s.color}; min-width: 60px;")

    def _update_stats(self) -> None:
        if not self._sets or self._current_idx >= len(self._sets):
            return
        s = self._sets[self._current_idx]
        # Live-update the set from UI
        try:
            s.name = self._name_le.text()
            s.orientation.mean_dip_direction = self._dd_spin.value()
            s.orientation.mean_dip = self._dip_spin.value()
            if s.provenance.get("kappa_status") != "UNRESOLVED":
                s.orientation.kappa = self._kappa_spin.value()
            if s.provenance.get("size_status") != "UNRESOLVED":
                s.size.lognormal_mu = self._mu_spin.value()
                s.size.lognormal_sigma = self._sigma_spin.value()
                s.size.power_law_exponent = self._D_spin.value()
                s.size.min_radius = self._min_r_spin.value()
                s.size.max_radius = self._max_r_spin.value()
            if s.provenance.get("p32_status") != "DERIVED_LATER_BY_M9":
                s.target_p32 = self._p32_spin.value()
                s.p32_tolerance = self._tol_spin.value()
            s.opacity = self._opacity_spin.value()
            dist_map = {"lognormal": "lognormal", "power_law": "power_law", "fixed": "fixed",
                        "exponential": "exponential", "truncated_power_law": "truncated_power_law"}
            if s.provenance.get("size_status") != "UNRESOLVED":
                s.size.distribution_type = SizeDistributionType(
                    dist_map.get(self._dist_combo.currentData(), "lognormal")
                )

            self._refresh_stats_label(s)
        except Exception:
            self._stats_lbl.setText("(invalid parameters)")

    def _refresh_stats_label(self, joint_set: JointSetConfig) -> None:
        """Refresh derived display text without writing editor values to the model."""
        if (
            joint_set.provenance.get("size_status") == "UNRESOLVED"
            or joint_set.provenance.get("p32_status") == "DERIVED_LATER_BY_M9"
        ):
            self._stats_lbl.setText("Expected fracture count is not computed from unresolved Size/P32 placeholders.")
            return
        mean_r = joint_set.size.mean_radius
        mean_area = joint_set.expected_mean_area()
        n_est = joint_set.expected_fracture_count(self._volume)
        self._stats_lbl.setText(
            f"E[R] = {mean_r:.2f} m\n"
            f"E[πR²] = {mean_area:.2f} m²\n"
            f"Expected fractures: ~{n_est:,}"
        )

    def _on_import_from_observations(self) -> None:
        """Import orientation from borehole fracture observations.

        Groups observations by set_id, computes Fisher statistics, and
        creates or updates joint sets with computed orientation parameters.
        Size distribution and P32 remain at user-defined values.
        """
        if self._borehole_collection is None:
            return

        from dfn_cave_studio.borehole.orientation_statistics import OrientationStatisticsCalculator
        calc = OrientationStatisticsCalculator()
        stats = calc.compute_by_set(self._borehole_collection)

        if not stats:
            from dfn_cave_studio.ui.qt_adapter import QMessageBox
            QMessageBox.warning(
                self, "No Data",
                "No fracture observations with valid set_id found.\n\n"
                "Ensure fracture observations have set_id assigned (1, 2, 3, ...) "
                "before importing."
            )
            return

        # Build preview message
        lines = ["Computed Fisher statistics from borehole observations:\n"]
        for set_id in sorted(stats.keys()):
            orient = stats[set_id]
            n_obs = sum(
                1 for bh in self._borehole_collection
                for obs in bh.fracture_observations
                if obs.set_id == set_id
            )
            lines.append(
                f"  Set {set_id}: dd={orient.mean_dip_direction:.1f}°, "
                f"dip={orient.mean_dip:.1f}°, kappa={orient.kappa:.1f} "
                f"({n_obs} observations)"
            )

        lines.append("\nOrientation will be set from borehole observations.")
        lines.append("Size distribution and P32 remain at current values.\n")
        lines.append("Proceed with updating joint sets?")

        from dfn_cave_studio.ui.qt_adapter import QMessageBox
        reply = QMessageBox.question(
            self, "Import from Observations",
            "\n".join(lines),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Update or create joint sets for each set_id
        existing_ids = {s.set_id for s in self._sets}
        for set_id in sorted(stats.keys()):
            orient = stats[set_id]
            if set_id in existing_ids:
                # Update existing set
                for s in self._sets:
                    if s.set_id == set_id:
                        s.orientation = orient
                        s.provenance["orientation"] = "borehole"
                        break
            else:
                # Create new set
                name = f"Joint Set {set_id}"
                js = JointSetConfig(
                    set_id=set_id,
                    name=name,
                    color=joint_set_color(set_id),
                    orientation=orient,
                    size=SizeDistribution(
                        distribution_type=SizeDistributionType.FIXED,
                        min_radius=0.5, max_radius=5.0,
                    ),
                    target_p32=1.0,
                    provenance={"orientation": "borehole", "size": "user", "p32": "user"},
                )
                self._sets.append(js)

        # Refresh UI
        self._refresh_list()
        self._list.setCurrentRow(0)
        self._on_set_selected(0)
        self._refresh_data_status()

    def _on_accept(self) -> None:
        if self._phase2a_service is not None and not self._sets and self._representative_table.rowCount() > 0:
            self._fit_status.setText(
                "Imported P/Z direction candidates are available. Fit and review the global joint sets before "
                "confirming."
            )
            return
        if self._phase2a_service is not None and self._sets and not self._pending_fit_matches_sets():
            self._pending_global_fit = self._phase2a_service.use_confirmed_project_sets(self._sets)
        self.accept()

    def get_joint_sets(self) -> List[JointSetConfig]:
        return self._sets

    def get_global_fit(self):
        """Return the pending authoritative local-to-global mapping, if any."""
        return self._pending_global_fit
