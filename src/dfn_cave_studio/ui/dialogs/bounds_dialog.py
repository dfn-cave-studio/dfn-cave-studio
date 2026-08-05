"""Model bounds and voxel configuration dialog."""

from typing import Optional, Tuple

from dfn_cave_studio.ui.qt_adapter import (
    Qt, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QDialogButtonBox, QGroupBox,
)
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig


class ModelBoundsDialog(QDialog):
    """Dialog for setting model boundaries and voxel configuration."""

    def __init__(self, bounds: Optional[ModelBounds] = None,
                 voxel: Optional[VoxelConfig] = None,
                 seed: int = 42, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Bounds & Voxel Settings")
        self._bounds = bounds or ModelBounds(x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=100)
        self._voxel = voxel or VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)
        self._seed = seed
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Bounds group
        bounds_group = QGroupBox("Model Bounds")
        bf = QFormLayout(bounds_group)
        self._x_min = self._make_double(self._bounds.x_min)
        self._x_max = self._make_double(self._bounds.x_max)
        self._y_min = self._make_double(self._bounds.y_min)
        self._y_max = self._make_double(self._bounds.y_max)
        self._z_min = self._make_double(self._bounds.z_min)
        self._z_max = self._make_double(self._bounds.z_max)
        row1 = QHBoxLayout(); row1.addWidget(QLabel("X Min:")); row1.addWidget(self._x_min)
        row1.addWidget(QLabel("X Max:")); row1.addWidget(self._x_max)
        row2 = QHBoxLayout(); row2.addWidget(QLabel("Y Min:")); row2.addWidget(self._y_min)
        row2.addWidget(QLabel("Y Max:")); row2.addWidget(self._y_max)
        row3 = QHBoxLayout(); row3.addWidget(QLabel("Z Min:")); row3.addWidget(self._z_min)
        row3.addWidget(QLabel("Z Max:")); row3.addWidget(self._z_max)
        bf.addRow(row1); bf.addRow(row2); bf.addRow(row3)
        self._volume_lbl = QLabel()
        bf.addRow("Volume:", self._volume_lbl)
        layout.addWidget(bounds_group)

        # Voxel group
        voxel_group = QGroupBox("Voxel Settings")
        vf = QFormLayout(voxel_group)
        self._dx = self._make_double(self._voxel.cell_size_x, 0.1, 100.0, 2)
        self._dy = self._make_double(self._voxel.cell_size_y, 0.1, 100.0, 2)
        self._dz = self._make_double(self._voxel.cell_size_z, 0.1, 100.0, 2)
        vr = QHBoxLayout()
        vr.addWidget(QLabel("Cell Size X (m):")); vr.addWidget(self._dx)
        vr.addWidget(QLabel("Y (m):")); vr.addWidget(self._dy)
        vr.addWidget(QLabel("Z (m):")); vr.addWidget(self._dz)
        vf.addRow(vr)
        self._grid_lbl = QLabel()
        vf.addRow("Grid Size:", self._grid_lbl)
        self._mem_lbl = QLabel()
        vf.addRow("Memory Estimate:", self._mem_lbl)
        self._danger_lbl = QLabel()
        self._danger_lbl.setStyleSheet("color: red; font-weight: bold;")
        vf.addRow(self._danger_lbl)
        layout.addWidget(voxel_group)

        # Seed
        seed_group = QGroupBox("Random Seed")
        sf = QFormLayout(seed_group)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2147483647)
        self._seed_spin.setValue(self._seed)
        sf.addRow("Master Seed:", self._seed_spin)
        layout.addWidget(seed_group)

        # Connect signals
        for sp in [self._x_min, self._x_max, self._y_min, self._y_max,
                    self._z_min, self._z_max, self._dx, self._dy, self._dz]:
            sp.valueChanged.connect(self._update_info)

        self._update_info()

        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _make_double(self, val: float, lo: float = -1e6, hi: float = 1e6, decimals: int = 1) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(decimals)
        sb.setValue(val)
        return sb

    def _update_info(self) -> None:
        w = self._x_max.value() - self._x_min.value()
        d = self._y_max.value() - self._y_min.value()
        h = self._z_max.value() - self._z_min.value()
        vol = w * d * h
        self._volume_lbl.setText(f"{vol:.0f} m³ ({w:.1f}×{d:.1f}×{h:.1f} m)")

        import math
        nx = max(1, math.ceil(w / max(0.01, self._dx.value())))
        ny = max(1, math.ceil(d / max(0.01, self._dy.value())))
        nz = max(1, math.ceil(h / max(0.01, self._dz.value())))
        total = nx * ny * nz
        self._grid_lbl.setText(f"{nx} × {ny} × {nz} = {total:,} voxels")
        mem_mb = total * 4 * 5 / (1024 * 1024)
        self._mem_lbl.setText(f"~{mem_mb:.0f} MB (5 attributes, int32)")

        if total > 100_000_000:
            self._danger_lbl.setText("⚠ DANGER: >100M voxels — extreme memory usage!")
        elif total > 10_000_000:
            self._danger_lbl.setText("⚠ WARNING: >10M voxels — high memory usage")
        else:
            self._danger_lbl.setText("")

    def get_bounds(self) -> ModelBounds:
        return ModelBounds(
            x_min=self._x_min.value(), x_max=self._x_max.value(),
            y_min=self._y_min.value(), y_max=self._y_max.value(),
            z_min=self._z_min.value(), z_max=self._z_max.value(),
        )

    def get_voxel_config(self) -> VoxelConfig:
        return VoxelConfig(
            cell_size_x=self._dx.value(),
            cell_size_y=self._dy.value(),
            cell_size_z=self._dz.value(),
        )

    def get_seed(self) -> int:
        return self._seed_spin.value()
