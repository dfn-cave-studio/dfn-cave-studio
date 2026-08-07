"""M7 structural domain editor dialog."""

import copy

from dfn_cave_studio.ui.qt_adapter import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QDialogButtonBox,
    QListWidget,
    QMessageBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)
from dfn_cave_studio.models.structural_domain import StructuralDomain, StructuralDomainCollection
from dfn_cave_studio.models.data_management import DomainInterval
from dfn_cave_studio.services.m7_state import (
    get_domain_intervals,
    set_domain_intervals,
    get_raw_domain_intervals,
)

DOMAIN_COLORS = ["#66bb6a", "#42a5f5", "#ffa726", "#ef5350", "#ab47bc", "#26c6da", "#d4e157", "#8d6e63"]


class M7DomainDialog(QDialog):
    """Editor for structural domains and borehole interval assignments."""

    def __init__(self, project, workflow, parent=None):
        super().__init__(parent)
        self._project = project
        self._workflow = workflow
        self._domains: StructuralDomainCollection = copy.deepcopy(project.structural_domains)
        self._original_domains = project.structural_domains.model_dump()
        self._original_intervals = [interval.model_dump() for interval in get_domain_intervals(project)]
        self._committed_changes = False
        self._intervals: list = []
        self.setWindowTitle("Structural Domain Editor")
        self.resize(700, 500)
        self._init_ui()
        self._refresh_domain_list()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # Left: domain list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(QLabel("Domains:"))
        self._domain_list = QListWidget()
        self._domain_list.currentRowChanged.connect(self._on_domain_selected)
        ll.addWidget(self._domain_list)

        btn_row = QHBoxLayout()
        self._add_domain_btn = QPushButton("+ Add")
        self._add_domain_btn.clicked.connect(self._add_domain)
        btn_row.addWidget(self._add_domain_btn)
        self._remove_domain_btn = QPushButton("- Remove")
        self._remove_domain_btn.clicked.connect(self._remove_domain)
        btn_row.addWidget(self._remove_domain_btn)
        ll.addLayout(btn_row)
        layout.addWidget(left, 1)

        # Right: domain properties
        right = QWidget()
        rl = QFormLayout(right)

        self._name_edit = QLineEdit()
        rl.addRow("Name:", self._name_edit)

        self._id_spin = QSpinBox()
        self._id_spin.setRange(0, 999)
        rl.addRow("Domain ID:", self._id_spin)

        self._color_combo = QComboBox()
        for c in DOMAIN_COLORS:
            self._color_combo.addItem(c, c)
        rl.addRow("Color:", self._color_combo)

        self._note_edit = QLineEdit()
        rl.addRow("Notes:", self._note_edit)

        rl.addRow(QLabel(""))
        rl.addRow(QLabel("<i>Assign domain to boreholes:</i>"))

        self._bh_combo = QComboBox()
        coll = self._project.borehole_collection
        if coll:
            for bh in coll:
                self._bh_combo.addItem(bh.borehole_id, bh.borehole_id)
        rl.addRow("Borehole:", self._bh_combo)

        interval_row = QHBoxLayout()
        self._from_spin = QDoubleSpinBox()
        self._from_spin.setRange(0, 10000)
        self._from_spin.setDecimals(1)
        interval_row.addWidget(QLabel("From:"))
        interval_row.addWidget(self._from_spin)
        self._to_spin = QDoubleSpinBox()
        self._to_spin.setRange(0, 10000)
        self._to_spin.setDecimals(1)
        self._to_spin.setValue(200)
        interval_row.addWidget(QLabel("To:"))
        interval_row.addWidget(self._to_spin)
        rl.addRow(interval_row)

        self._assign_interval_btn = QPushButton("Assign Interval")
        self._assign_interval_btn.clicked.connect(self._assign_interval)
        rl.addRow(self._assign_interval_btn)

        self._delete_interval_btn = QPushButton("Delete Selected Interval")
        self._delete_interval_btn.clicked.connect(self._delete_selected_interval)
        rl.addRow(self._delete_interval_btn)

        rl.addRow(QLabel("Assigned intervals:"))
        self._interval_table = QTableWidget(0, 4)
        self._interval_table.setHorizontalHeaderLabels(["Hole ID", "From", "To", "Domain"])
        self._interval_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rl.addRow(self._interval_table)

        layout.addWidget(right, 2)

        # Note
        note = QLabel(
            "<i>Current scope: borehole-interval constraints only.<br>"
            "3D domain volumes will be interpolated/imported in a later phase.</i>"
        )
        note.setWordWrap(True)
        rl.addRow(note)

        self._button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._button_box.accepted.connect(self._on_accept)
        self._button_box.rejected.connect(self.reject)
        rl.addRow(self._button_box)

        # NOW load data — after all widgets (including _interval_table) exist
        self._load_imported_domains()

    def _refresh_domain_list(self):
        self._domain_list.clear()
        for d in self._domains.domains:
            self._domain_list.addItem(f"{d.name} (ID={d.domain_id})")

    def _on_domain_selected(self, idx):
        if idx < 0 or idx >= len(self._domains.domains):
            return
        d = self._domains.domains[idx]
        self._name_edit.setText(d.name)
        self._id_spin.setValue(d.domain_id)
        idx_c = next((i for i in range(self._color_combo.count()) if self._color_combo.itemData(i) == d.color), -1)
        if idx_c >= 0:
            self._color_combo.setCurrentIndex(idx_c)
        self._note_edit.setText(d.description)
        # Filter table: Global (id=0) shows all, others filter by domain_id
        filter_id = None if d.domain_id == 0 else d.domain_id
        self._refresh_interval_table(filter_id)

    def _add_domain(self):
        new_id = max([d.domain_id for d in self._domains.domains], default=0) + 1
        domain = StructuralDomain(
            domain_id=new_id,
            name=f"Domain {new_id}",
            color=DOMAIN_COLORS[(new_id - 1) % len(DOMAIN_COLORS)],
        )
        self._domains.domains.append(domain)
        self._refresh_domain_list()

    def _remove_domain(self):
        idx = self._domain_list.currentRow()
        if idx <= 0:
            return
        del self._domains.domains[idx]
        self._refresh_domain_list()

    def _assign_interval(self):
        domain_idx = self._domain_list.currentRow()
        if domain_idx < 0:
            return
        domain = self._domains.domains[domain_idx]
        bh_id = self._bh_combo.currentData()
        from_depth = self._from_spin.value()
        to_depth = self._to_spin.value()
        if from_depth >= to_depth:
            QMessageBox.warning(self, "Invalid Interval", "From depth must be less than To depth.")
            return
        borehole = next(
            (bh for bh in self._project.borehole_collection or [] if bh.borehole_id == bh_id),
            None,
        )
        if borehole is not None and to_depth > borehole.collar.final_depth:
            QMessageBox.warning(
                self,
                "Invalid Interval",
                f"To depth {to_depth:g} exceeds {bh_id} depth " f"{borehole.collar.final_depth:g}.",
            )
            return
        candidate = DomainInterval(
            hole_id=bh_id,
            from_depth=from_depth,
            to_depth=to_depth,
            domain_id=domain.domain_id,
            domain_name=domain.name,
            assignment_method="manual",
        )
        if any(interval.hole_id == bh_id and interval.overlaps(candidate) for interval in self._intervals):
            QMessageBox.warning(
                self,
                "Overlapping Interval",
                f"{bh_id} already has an overlapping interval.",
            )
            return
        self._intervals.append(candidate)
        row = self._interval_table.rowCount()
        self._interval_table.insertRow(row)
        self._interval_table.setItem(row, 0, QTableWidgetItem(bh_id))
        self._interval_table.setItem(row, 1, QTableWidgetItem(str(candidate.from_depth)))
        self._interval_table.setItem(row, 2, QTableWidgetItem(str(candidate.to_depth)))
        self._interval_table.setItem(row, 3, QTableWidgetItem(domain.name))

    def _on_accept(self):
        idx = self._domain_list.currentRow()
        if idx >= 0 and idx < len(self._domains.domains):
            d = self._domains.domains[idx]
            d.name = self._name_edit.text()
            d.domain_id = self._id_spin.value()
            d.color = self._color_combo.currentData()
            d.description = self._note_edit.text()
        validation_error = self._validate_intervals()
        if validation_error:
            QMessageBox.critical(self, "Invalid Domain Intervals", validation_error)
            return
        new_intervals = [interval.model_dump() for interval in self._intervals]
        self._committed_changes = (
            self._domains.model_dump() != self._original_domains or new_intervals != self._original_intervals
        )
        if self._committed_changes:
            self._project.structural_domains = copy.deepcopy(self._domains)
            set_domain_intervals(self._project, copy.deepcopy(self._intervals))
            self._workflow.complete_step("domains")
            self._workflow.mark_ready("joint_sets")
        self.accept()

    def _validate_intervals(self) -> str:
        """Validate every pending interval before committing it."""
        max_depths = {
            borehole.borehole_id: borehole.collar.final_depth for borehole in self._project.borehole_collection or []
        }
        by_hole: dict[str, list[DomainInterval]] = {}
        for interval in self._intervals:
            if interval.from_depth >= interval.to_depth:
                return f"{interval.hole_id}: from_depth must be less than " "to_depth."
            max_depth = max_depths.get(interval.hole_id)
            if max_depth is None:
                return f"{interval.hole_id}: borehole does not exist."
            if interval.to_depth > max_depth:
                return (
                    f"{interval.hole_id}: interval ends at "
                    f"{interval.to_depth:g} m, beyond borehole depth "
                    f"{max_depth:g} m."
                )
            by_hole.setdefault(interval.hole_id, []).append(interval)
        for hole_id, intervals in by_hole.items():
            ordered = sorted(intervals, key=lambda item: item.from_depth)
            for previous, current in zip(ordered, ordered[1:]):
                if previous.overlaps(current):
                    return (
                        f"{hole_id}: intervals "
                        f"{previous.from_depth:g}–{previous.to_depth:g} m and "
                        f"{current.from_depth:g}–{current.to_depth:g} m overlap."
                    )
        return ""

    def _load_imported_domains(self):
        """Load domain intervals from project formal state.

        Uses get_domain_intervals() which returns list[DomainInterval].
        Falls back to raw_domain_intervals DataFrame for auto-creation
        on first use (migrated once).
        """
        # Try formal state first
        formal = get_domain_intervals(self._project)
        if formal:
            self._intervals = copy.deepcopy(formal)
            self._auto_create_domains_from_intervals()
            self._refresh_domain_list()
            self._refresh_interval_table(None)  # show all
            return

        # Fall back: convert raw DataFrame to formal list
        import pandas as pd

        raw = get_raw_domain_intervals(self._project)
        if raw is not None and not (isinstance(raw, pd.DataFrame) and raw.empty):
            df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
            self._intervals = []
            for _, row in df.iterrows():
                self._intervals.append(
                    DomainInterval(
                        hole_id=str(row["hole_id"]).strip(),
                        from_depth=float(row["from_depth"]),
                        to_depth=float(row["to_depth"]),
                        domain_id=int(row["domain_id"]),
                        domain_name=str(row.get("domain_name", "")),
                        assignment_method="imported",
                    )
                )
            self._auto_create_domains_from_intervals()
            self._refresh_domain_list()
            self._refresh_interval_table(None)

    def _auto_create_domains_from_intervals(self):
        """Auto-create StructuralDomain definitions from interval data."""
        existing_ids = {d.domain_id for d in self._domains.domains}
        for interval in self._intervals:
            if interval.domain_id not in existing_ids and interval.domain_id != 0:
                self._domains.domains.append(
                    StructuralDomain(
                        domain_id=interval.domain_id,
                        name=interval.domain_name or f"Domain {interval.domain_id}",
                        color=DOMAIN_COLORS[(interval.domain_id - 1) % len(DOMAIN_COLORS)],
                    )
                )
                existing_ids.add(interval.domain_id)

    def _refresh_interval_table(self, domain_id):
        """Populate interval table — all rows or filtered by domain_id."""
        self._interval_table.setRowCount(0)
        for interval in self._intervals:
            if domain_id is not None and interval.domain_id != domain_id:
                continue
            r = self._interval_table.rowCount()
            self._interval_table.insertRow(r)
            self._interval_table.setItem(r, 0, QTableWidgetItem(interval.hole_id))
            self._interval_table.setItem(r, 1, QTableWidgetItem(str(interval.from_depth)))
            self._interval_table.setItem(r, 2, QTableWidgetItem(str(interval.to_depth)))
            self._interval_table.setItem(r, 3, QTableWidgetItem(interval.domain_name or str(interval.domain_id)))

    def _delete_selected_interval(self):
        selected = self._interval_table.currentRow()
        if selected >= 0 and selected < len(self._intervals):
            domain_id = (
                self._domains.domains[self._domain_list.currentRow()].domain_id
                if self._domain_list.currentRow() >= 0
                else None
            )
            # Find the actual index in self._intervals
            visible = [
                i
                for i, iv in enumerate(self._intervals)
                if domain_id is None or domain_id == 0 or iv.domain_id == domain_id
            ]
            if selected < len(visible):
                del self._intervals[visible[selected]]
                filter_id = None if domain_id == 0 else domain_id
                self._refresh_interval_table(filter_id)

    def get_intervals(self):
        return self._intervals

    @property
    def committed_changes(self) -> bool:
        """Whether OK committed domain definitions or intervals."""
        return self._committed_changes
