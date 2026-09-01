"""M7 validation borehole holdout dialog."""

from dfn_cave_studio.ui.qt_adapter import (
    Qt,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QDialogButtonBox,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QMessageBox,
    QComboBox,
)
from dfn_cave_studio.services.holdout_service import HoldoutService, HoldoutRole
from dfn_cave_studio.services.m7_state import get_holdout, set_holdout


class M7HoldoutDialog(QDialog):
    """Dialog for calibration/validation borehole split."""

    def __init__(self, project, workflow, parent=None):
        super().__init__(parent)
        self._project = project
        self._workflow = workflow
        saved = get_holdout(project)
        self._service = HoldoutService.from_dict(saved.to_dict()) if saved is not None else HoldoutService()
        self._original_state = saved.to_dict() if saved is not None else None
        self._committed_changes = False
        self._saved_method = self._service.config.method
        self.setWindowTitle("Validation Borehole Holdout")
        self.resize(600, 450)
        self._init_ui()
        self._load_boreholes()
        if self._hole_ids and not self._service.calibration_holes and not self._service.validation_holes:
            self._service.select_manual(self._hole_ids, [])
        self._refresh_from_service()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Method selection
        method_group = QGroupBox("Selection Method")
        mf = QFormLayout(method_group)
        self._method_combo = QComboBox()
        self._method_combo.addItem("Manual selection", "manual")
        self._method_combo.addItem("Random (fixed seed)", "random")
        self._method_combo.addItem("Stratified by domain", "stratified")
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        mf.addRow("Method:", self._method_combo)

        self._fraction_spin = QDoubleSpinBox()
        self._fraction_spin.setRange(0.05, 0.5)
        self._fraction_spin.setValue(0.25)
        self._fraction_spin.setSingleStep(0.05)
        self._fraction_spin.setDecimals(2)
        mf.addRow("Validation fraction:", self._fraction_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(1, 99999)
        self._seed_spin.setValue(42)
        mf.addRow("Random seed:", self._seed_spin)
        layout.addWidget(method_group)

        # Borehole list
        list_group = QGroupBox("Boreholes")
        ll = QVBoxLayout(list_group)
        self._bh_list = QListWidget()
        self._bh_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        ll.addWidget(self._bh_list)

        btn_row = QHBoxLayout()
        self._cal_btn = QPushButton("Mark as Calibration")
        self._cal_btn.clicked.connect(lambda: self._mark_selected(HoldoutRole.CALIBRATION))
        btn_row.addWidget(self._cal_btn)
        self._val_btn = QPushButton("Mark as Validation")
        self._val_btn.clicked.connect(lambda: self._mark_selected(HoldoutRole.VALIDATION))
        btn_row.addWidget(self._val_btn)
        ll.addLayout(btn_row)
        layout.addWidget(list_group)

        # Stats
        self._stats_label = QLabel("")
        layout.addWidget(self._stats_label)

        # Lock
        lock_row = QHBoxLayout()
        self._lock_btn = QPushButton("Lock Holdout Split")
        self._lock_btn.clicked.connect(self._on_lock)
        lock_row.addWidget(self._lock_btn)
        self._unlock_btn = QPushButton("Unlock")
        self._unlock_btn.clicked.connect(self._on_unlock)
        self._unlock_btn.setEnabled(False)
        lock_row.addWidget(self._unlock_btn)
        layout.addLayout(lock_row)

        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    def _load_boreholes(self):
        self._bh_list.clear()
        coll = self._project.borehole_collection
        if coll is None:
            return
        self._hole_ids = [bh.borehole_id for bh in coll]
        for bh in coll:
            item = QListWidgetItem(f"{bh.borehole_id} ({bh.collar.final_depth:.0f}m)")
            item.setData(Qt.ItemDataRole.UserRole, bh.borehole_id)
            self._bh_list.addItem(item)

    def _on_method_changed(self, idx):
        method = self._method_combo.itemData(idx)
        if method == "manual":
            self._fraction_spin.setEnabled(False)
            self._seed_spin.setEnabled(False)
        else:
            self._fraction_spin.setEnabled(True)
            self._seed_spin.setEnabled(True)
            if method == "random":
                self._apply_random()
            elif method == "stratified":
                self._apply_stratified()

    def _mark_selected(self, role: HoldoutRole):
        for item in self._bh_list.selectedItems():
            hid = item.data(Qt.ItemDataRole.UserRole)
            if role == HoldoutRole.VALIDATION:
                item.setText(f"🔴 VAL: {hid}")
            else:
                item.setText(f"🟢 CAL: {hid}")
        self._update_stats()
        # Build manual holdout from current list state
        vals = []
        for i in range(self._bh_list.count()):
            item = self._bh_list.item(i)
            if "VAL" in item.text():
                vals.append(item.data(Qt.ItemDataRole.UserRole))
        self._service.select_manual(self._hole_ids, vals)
        # Save fraction to config even in manual mode
        self._service.update_config(validation_fraction=self._fraction_spin.value())

    def _apply_random(self):
        try:
            self._service.select_random(
                self._hole_ids,
                self._fraction_spin.value(),
                random_seed=self._seed_spin.value(),
            )
            self._refresh_list_from_service()
        except (RuntimeError, ValueError) as e:
            QMessageBox.warning(self, "Error", str(e))

    def _apply_stratified(self):
        QMessageBox.information(
            self,
            "Stratified",
            "Stratified holdout requires domain assignments. Use the Domain Editor first.\n\n"
            "Using random selection for now.",
        )
        self._apply_random()

    def _refresh_from_service(self):
        """Restore UI from saved holdout state."""
        if self._service is None:
            return
        self._refresh_list_from_service()
        cfg = self._service.config
        # Restore fraction/seed
        if cfg.random_seed is not None:
            self._seed_spin.setValue(cfg.random_seed)
        self._fraction_spin.setValue(cfg.validation_fraction)
        # Restore method without re-running a selection algorithm.
        self._method_combo.blockSignals(True)
        restored_index = self._method_combo.findData(cfg.method)
        self._method_combo.setCurrentIndex(restored_index if restored_index >= 0 else 0)
        self._method_combo.blockSignals(False)
        is_manual = self._method_combo.currentData() == "manual"
        self._fraction_spin.setEnabled(not is_manual)
        self._seed_spin.setEnabled(not is_manual)
        # Restore locked state
        if self._service.is_locked:
            self._set_locked_ui(True)
            self._stats_label.setText(self._stats_label.text() + "  [LOCKED]")

    def _refresh_list_from_service(self):
        for i in range(self._bh_list.count()):
            item = self._bh_list.item(i)
            hid = item.data(Qt.ItemDataRole.UserRole)
            if self._service.is_validation(hid):
                item.setText(f"🔴 VAL: {hid}")
            else:
                item.setText(f"🟢 CAL: {hid}")
        self._update_stats()

    def _update_stats(self):
        cal = sum(1 for i in range(self._bh_list.count()) if "CAL" in self._bh_list.item(i).text())
        val = sum(1 for i in range(self._bh_list.count()) if "VAL" in self._bh_list.item(i).text())
        total = self._bh_list.count()
        self._stats_label.setText(f"Calibration: {cal}  |  Validation: {val}  |  Total: {total}")

    def _on_lock(self):
        try:
            self._service.lock()
            self._set_locked_ui(True)
            self._stats_label.setText(self._stats_label.text() + "  [LOCKED]")
        except (RuntimeError, ValueError) as e:
            QMessageBox.warning(self, "Lock Error", str(e))

    def _on_unlock(self):
        self._service.unlock()
        self._set_locked_ui(False)
        self._stats_label.setText(self._stats_label.text().replace("  [LOCKED]", ""))

    def _set_locked_ui(self, locked: bool) -> None:
        """Prevent holdout editing until the pending split is unlocked."""
        self._lock_btn.setEnabled(not locked)
        self._unlock_btn.setEnabled(locked)
        self._method_combo.setEnabled(not locked)
        self._bh_list.setEnabled(not locked)
        self._cal_btn.setEnabled(not locked)
        self._val_btn.setEnabled(not locked)
        if locked:
            self._fraction_spin.setEnabled(False)
            self._seed_spin.setEnabled(False)
        else:
            self._on_method_changed(self._method_combo.currentIndex())

    def _on_accept(self):
        """Commit the pending holdout and workflow state atomically."""
        new_state = self._service.to_dict()
        self._committed_changes = new_state != self._original_state
        if self._committed_changes:
            set_holdout(self._project, self._service)
            if self._service.is_locked:
                self._workflow.complete_step("holdout")
                self._workflow.mark_ready("domains")
            else:
                self._workflow.invalidate_from("holdout")
        self.accept()

    @property
    def committed_changes(self) -> bool:
        """Whether OK committed a different holdout configuration."""
        return self._committed_changes
