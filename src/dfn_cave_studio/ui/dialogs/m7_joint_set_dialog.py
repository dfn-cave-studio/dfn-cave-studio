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
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.m7_state import get_holdout
from dfn_cave_studio.ui.i18n import language_manager, tr

class M7JointSetDialog(QDialog):
    """Joint set identification using M7 services."""

    def __init__(self, project, workflow, parent=None):
        super().__init__(parent)
        self._project = project
        self._workflow = workflow
        self._service = JointSetService(random_seed=project.config.master_seed)
        self._result = None
        self._committed_changes = False
        self.setWindowTitle("Joint Set Identification")
        self.resize(750, 500)
        self._init_ui()
        language_manager().language_changed.connect(self._retranslate_dynamic_content)

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
        self._result_table = QTableWidget(0, 12)
        self._result_table.setHorizontalHeaderLabels(
            [
                "Set ID",
                "Name",
                "Dip Dir (°)",
                "Dip (°)",
                "Kappa",
                "Calibration Count",
                "Calibration Full",
                "Calibration Dip-only",
                "Source",
                "Validation Count",
                "Validation Full",
                "Validation Dip-only",
            ]
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

        try:
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
        except ValueError as error:
            QMessageBox.warning(self, "Joint Set Identification", str(error))
            return

        self._result = result
        self._populate_results(result)

    def _populate_results(self, result):
        visible_set_ids = sorted(set(result.sets) | set(result.set_counts))
        self._result_table.setRowCount(len(visible_set_ids))
        total_cal = 0
        for i, set_id in enumerate(visible_set_ids):
            js = result.sets.get(set_id)
            counts = result.set_counts.get(set_id, {})
            validation_counts = result.validation_set_counts.get(set_id, {})
            n_assigned = counts.get("total", len([a for a in result.assignments.values() if a == set_id]))
            full_count = counts.get("full_orientation", n_assigned)
            dip_only_count = counts.get("dip_only", 0)
            validation_count = validation_counts.get("total", 0)
            validation_full = validation_counts.get("full_orientation", 0)
            validation_dip_only = validation_counts.get("dip_only", 0)
            total_cal += n_assigned
            self._result_table.setItem(i, 0, QTableWidgetItem(str(set_id)))
            fallback_name = tr("Joint Set {set_id}").format(set_id=set_id)
            self._result_table.setItem(i, 1, QTableWidgetItem(js.name if js is not None else fallback_name))
            self._result_table.setItem(
                i,
                2,
                QTableWidgetItem(f"{js.orientation.mean_dip_direction:.1f}" if js is not None else "—"),
            )
            self._result_table.setItem(i, 3, QTableWidgetItem(f"{js.orientation.mean_dip:.1f}" if js is not None else "—"))
            self._result_table.setItem(i, 4, QTableWidgetItem(f"{js.orientation.kappa:.1f}" if js is not None else "—"))
            self._result_table.setItem(i, 5, QTableWidgetItem(str(n_assigned)))
            self._result_table.setItem(i, 6, QTableWidgetItem(str(full_count)))
            self._result_table.setItem(i, 7, QTableWidgetItem(str(dip_only_count)))
            source = js.provenance.get("orientation", result.mode) if js is not None else "INSUFFICIENT_ORIENTATION_DATA"
            source_item = QTableWidgetItem(self._source_text(source))
            source_item.setData(0x0100, source)
            self._result_table.setItem(i, 8, source_item)
            self._result_table.setItem(i, 9, QTableWidgetItem(str(validation_count)))
            self._result_table.setItem(i, 10, QTableWidgetItem(str(validation_full)))
            self._result_table.setItem(i, 11, QTableWidgetItem(str(validation_dip_only)))
        self._stats_label.setText(self._stats_text(result, total_cal))

    @staticmethod
    def _source_text(source: str) -> str:
        """Translate source/status presentation without modifying the stable value."""
        return {
            "INSUFFICIENT_ORIENTATION_DATA": tr("Insufficient Orientation Data"),
            "automatic": tr("Automatic"),
            "imported": tr("Imported"),
        }.get(source, source)

    @staticmethod
    def _stats_text(result, total_cal: int) -> str:
        """Build the translated statistics summary from unchanged scientific counts."""
        dip_only = result.dip_only_count if result.mode == "automatic" else 0
        return tr(
            "Mode: {mode} | Calibration fractures used: {calibration} | "
            "Full-orientation records used: {full} | "
            "Dip-only records not eligible for spherical clustering: {dip_only} | "
            "Validation fractures excluded: {validation} | Sets count sum: {total}"
        ).format(
            mode=M7JointSetDialog._source_text(result.mode),
            calibration=result.calibration_count,
            full=result.full_orientation_count,
            dip_only=dip_only,
            validation=result.validation_count,
            total=total_cal,
        )

    def _retranslate_dynamic_content(self, _language: str) -> None:
        """Refresh dynamic result cells without rerunning joint-set identification."""
        if self._result is None:
            return
        self._populate_results(self._result)

    def _on_accept(self):
        """Commit fitted sets and their canonical Formal-record assignments."""
        all_sets = self._service.get_all_joint_sets()
        if not all_sets or self._result is None:
            self.accept()
            return
        repository = BoreholeRepository(self._project)
        if not repository.database.records:
            repository.migrate_m7()
        assignments = {**self._result.assignments, **self._result.validation_assignments}
        try:
            repository.apply_joint_set_assignments(assignments)
        except (KeyError, TypeError, ValueError) as error:
            QMessageBox.critical(self, tr("Joint Set Commit Failed"), str(error))
            return
        self._project.joint_sets = all_sets
        self._workflow.complete_step("joint_sets")
        self._committed_changes = True
        self.accept()

    @property
    def committed_changes(self) -> bool:
        """Whether OK wrote identified joint sets to the project."""
        return self._committed_changes
