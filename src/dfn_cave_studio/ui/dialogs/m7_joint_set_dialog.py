"""M7 joint set identification dialog — wired to M7 services."""

from dfn_cave_studio.ui.qt_adapter import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialogButtonBox,
    QGroupBox,
    QMessageBox,
)
from dfn_cave_studio.services.joint_set_service import JointSetService
from dfn_cave_studio.services.m7_state import get_holdout

SET_COLORS = ["#1976d2", "#388e3c", "#f57c00", "#d32f2f", "#7b1fa2"]


class M7JointSetDialog(QDialog):
    """Joint set identification using M7 services."""

    def __init__(self, project, workflow, parent=None):
        super().__init__(parent)
        self._project = project
        self._workflow = workflow
        self._service = JointSetService(random_seed=project.config.master_seed)
        self._committed_changes = False
        self.setWindowTitle("Joint Set Identification")
        self.resize(750, 500)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Mode selection
        mode_group = QGroupBox("Identification Mode")
        mf = QFormLayout(mode_group)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Mode A: Use imported set_id from CSV", "imported")
        self._mode_combo.addItem("Mode B: Auto-identify (spherical K-Means)", "automatic")
        mf.addRow("Mode:", self._mode_combo)

        self._n_clusters_spin = QSpinBox()
        self._n_clusters_spin.setRange(1, 10)
        self._n_clusters_spin.setValue(3)
        mf.addRow("Number of sets:", self._n_clusters_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(1, 99999)
        self._seed_spin.setValue(self._project.config.master_seed)
        mf.addRow("Random seed:", self._seed_spin)
        layout.addWidget(mode_group)

        # Run button
        self._identify_btn = QPushButton("Identify Joint Sets")
        self._identify_btn.clicked.connect(self._on_identify)
        layout.addWidget(self._identify_btn)

        # Results table
        self._result_table = QTableWidget(0, 7)
        self._result_table.setHorizontalHeaderLabels(
            ["Set ID", "Name", "Dip Dir (°)", "Dip (°)", "Kappa", "Count", "Source"]
        )
        self._result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self._result_table)

        # Stats
        self._stats_label = QLabel("")
        layout.addWidget(self._stats_label)

        # Note about axial equivalence
        note = QLabel(
            "<i>Auto-identification uses unit normal vectors on the sphere.<br>"
            "n and -n are treated as the same fracture plane (axial equivalence).<br>"
            "Validation boreholes are excluded from clustering and shown separately.</i>"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    def _on_identify(self):
        collection = self._project.borehole_collection
        if collection is None:
            QMessageBox.warning(self, "No Data", "Import borehole data first.")
            return

        # Get calibration/validation split from holdout via m7_state
        holdout = get_holdout(self._project)
        if holdout is None or not holdout.is_locked:
            QMessageBox.warning(
                self,
                "Holdout Not Locked",
                "Please lock the validation holdout split before identifying joint sets.\n\n"
                "Open 'Validation Borehole Holdout', select validation boreholes, and lock the split.",
            )
            return

        cal_holes = set(holdout.calibration_holes)
        val_holes = set(holdout.validation_holes)
        hole_ids = [bh.borehole_id for bh in collection]

        cal_holes.intersection_update(hole_ids)
        val_holes.intersection_update(hole_ids)
        if not cal_holes:
            QMessageBox.warning(
                self,
                "No Calibration Boreholes",
                "At least one calibration borehole is required.",
            )
            return

        mode = self._mode_combo.currentData()
        seed = self._seed_spin.value()

        if mode == "imported":
            result = self._service.identify_from_imported(collection, cal_holes, val_holes)
        else:
            result = self._service.identify_auto(
                collection,
                cal_holes,
                val_holes,
                n_clusters=self._n_clusters_spin.value(),
                random_seed=seed,
            )

        self._populate_results(result)

    def _populate_results(self, result):
        self._result_table.setRowCount(len(result.sets))
        total_cal = 0
        for i, (set_id, js) in enumerate(sorted(result.sets.items())):
            n_assigned = len([a for a in result.assignments.values() if a == set_id])
            total_cal += n_assigned
            self._result_table.setItem(i, 0, QTableWidgetItem(str(set_id)))
            self._result_table.setItem(i, 1, QTableWidgetItem(js.name))
            self._result_table.setItem(i, 2, QTableWidgetItem(f"{js.orientation.mean_dip_direction:.1f}"))
            self._result_table.setItem(i, 3, QTableWidgetItem(f"{js.orientation.mean_dip:.1f}"))
            self._result_table.setItem(i, 4, QTableWidgetItem(f"{js.orientation.kappa:.1f}"))
            self._result_table.setItem(i, 5, QTableWidgetItem(str(n_assigned)))
            self._result_table.setItem(i, 6, QTableWidgetItem(js.provenance.get("orientation", result.mode)))
        self._stats_label.setText(
            f"Mode: {result.mode}  |  "
            f"Calibration fractures used: {result.calibration_count}  |  "
            f"Validation fractures excluded: {result.validation_count}  |  "
            f"Sets count sum: {total_cal}"
        )

    def _on_accept(self):
        # Update project joint sets
        all_sets = self._service.get_all_joint_sets()
        if all_sets:
            self._project.joint_sets = all_sets
            self._workflow.complete_step("joint_sets")
            self._committed_changes = True
        self.accept()

    @property
    def committed_changes(self) -> bool:
        """Whether OK wrote identified joint sets to the project."""
        return self._committed_changes
