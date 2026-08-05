"""Joint set manager dialog for configuring fracture population parameters."""

import math
from typing import List, Optional

from dfn_cave_studio.ui.qt_adapter import (
    Qt, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QComboBox,
    QDialogButtonBox, QGroupBox, QListWidget, QListWidgetItem,
    QSplitter, QTabWidget, QWidget, QColorDialog, QLineEdit,
)
from dfn_cave_studio.models.fracture_set import (
    JointSetConfig, OrientationDistribution, SizeDistribution,
)
from dfn_cave_studio.models.enums import SizeDistributionType


SET_COLORS = ["#1976d2", "#388e3c", "#f57c00", "#d32f2f", "#7b1fa2",
              "#0288d1", "#689f38", "#fbc02d", "#e64a19", "#5c6bc0"]


class JointSetManagerDialog(QDialog):
    """Dialog for creating and editing joint sets with live statistics."""

    def __init__(self, joint_sets: Optional[List[JointSetConfig]] = None,
                 model_volume: float = 1000000.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Joint Set Manager")
        self.resize(800, 550)
        self._volume = model_volume
        self._sets: List[JointSetConfig] = list(joint_sets) if joint_sets else []
        if not self._sets:
            self._sets.append(self._default_set(0))
        self._current_idx = 0
        self._init_ui()

    def _default_set(self, idx: int) -> JointSetConfig:
        return JointSetConfig(
            set_id=idx + 1, name=f"Joint Set {idx + 1}",
            color=SET_COLORS[idx % len(SET_COLORS)],
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
        ll.addWidget(QLabel("Fracture Sets:"))
        ll.addWidget(self._list)
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
        self._kappa_spin = self._make_double(0.1, 200, 1); self._kappa_spin.setValue(30)
        self._dist_combo = QComboBox()
        self._dist_combo.addItems(["lognormal", "power_law", "fixed", "exponential", "truncated_power_law"])
        self._dist_combo.currentTextChanged.connect(self._update_stats)
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
                    self._p32_spin, self._tol_spin]:
            sp.valueChanged.connect(self._update_stats)

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
        of.addRow("Mean Dip Direction (°):", self._dd_spin)
        of.addRow("Mean Dip (°):", self._dip_spin)
        of.addRow("Kappa (concentration):", self._kappa_spin)
        right.addTab(orient_tab, "Orientation")

        # Size tab
        size_tab = QWidget()
        sf = QFormLayout(size_tab)
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
        inf.addRow("Target P32 (m²/m³):", self._p32_spin)
        inf.addRow("Tolerance:", self._tol_spin)
        self._stats_lbl = QLabel()
        inf.addRow("Statistics:", self._stats_lbl)
        right.addTab(int_tab, "Intensity")

        splitter.addWidget(right)
        splitter.setSizes([250, 550])
        layout.addWidget(splitter)

        self._update_stats()

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._on_set_selected(0)

    def _make_double(self, lo, hi, decimals) -> QDoubleSpinBox:
        sb = QDoubleSpinBox(); sb.setRange(lo, hi); sb.setDecimals(decimals); return sb

    def _refresh_list(self) -> None:
        self._list.clear()
        for s in self._sets:
            item = QListWidgetItem(f"  {s.name}")
            item.setForeground(Qt.GlobalColor(int(s.color[1:], 16))  # approximate
                              if hasattr(Qt, 'GlobalColor') else None)
            self._list.addItem(item)

    def _on_set_selected(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._sets):
            return
        self._current_idx = idx
        s = self._sets[idx]
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
        idx_lookup = {"lognormal": 0, "power_law": 1, "fixed": 2, "exponential": 3, "truncated_power_law": 4}
        self._dist_combo.setCurrentIndex(idx_lookup.get(s.size.distribution_type.value, 0))
        self._update_stats()

    def _add_set(self) -> None:
        s = self._default_set(len(self._sets))
        self._sets.append(s)
        self._refresh_list()
        self._list.setCurrentRow(len(self._sets) - 1)

    def _remove_set(self) -> None:
        if len(self._sets) <= 1:
            return
        del self._sets[self._current_idx]
        self._current_idx = max(0, self._current_idx - 1)
        self._refresh_list()
        self._list.setCurrentRow(self._current_idx)

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
            s.orientation.kappa = self._kappa_spin.value()
            s.size.lognormal_mu = self._mu_spin.value()
            s.size.lognormal_sigma = self._sigma_spin.value()
            s.size.power_law_exponent = self._D_spin.value()
            s.size.min_radius = self._min_r_spin.value()
            s.size.max_radius = self._max_r_spin.value()
            s.target_p32 = self._p32_spin.value()
            s.p32_tolerance = self._tol_spin.value()
            s.opacity = self._opacity_spin.value()
            dist_map = {"lognormal": "lognormal", "power_law": "power_law", "fixed": "fixed",
                        "exponential": "exponential", "truncated_power_law": "truncated_power_law"}
            s.size.distribution_type = SizeDistributionType(dist_map.get(self._dist_combo.currentText(), "lognormal"))

            mean_r = s.size.mean_radius
            mean_area = s.expected_mean_area()
            n_est = s.expected_fracture_count(self._volume)
            self._stats_lbl.setText(
                f"E[R] = {mean_r:.2f} m\n"
                f"E[πR²] = {mean_area:.2f} m²\n"
                f"Expected fractures: ~{n_est:,}"
            )
        except Exception:
            self._stats_lbl.setText("(invalid parameters)")

    def _on_accept(self) -> None:
        self._update_stats()
        for i, s in enumerate(self._sets):
            s.set_id = i + 1
        self.accept()

    def get_joint_sets(self) -> List[JointSetConfig]:
        return self._sets
