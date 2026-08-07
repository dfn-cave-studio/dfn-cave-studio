"""M7 final validation tests: CSV mapping, XLSX import, StructuralDomain
persistence, MainWindow round-trip, Joint Set reproducibility.

All tests use pytest-qt with FakePlotter (no real VTK/OpenGL).
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.bounds import ModelBounds
from dfn_cave_studio.models.borehole import SurveyStation
from dfn_cave_studio.models.structural_domain import (
    StructuralDomain,
)
from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.services.joint_set_service import JointSetService
from dfn_cave_studio.services.m7_state import (
    get_holdout,
    get_domain_intervals,
    get_workflow,
)
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def demo_dir():
    return Path(__file__).parent.parent.parent / "examples" / "m7_demo"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_full_project(demo_dir):
    """Build complete M7 project with all data cleaned and holdout locked."""
    project = Project()
    project.metadata.name = "Final Validation"
    project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

    imp = BoreholeImporter()
    cr = imp.import_all(
        collar_path=str(demo_dir / "collars.csv"),
        fractures_path=str(demo_dir / "fractures.csv"),
    )
    project.borehole_collection = cr.collection
    bh_map = {bh.borehole_id: bh for bh in cr.collection}

    raw = pd.read_csv(demo_dir / "surveys.csv")
    max_depths = {bh.borehole_id: bh.collar.final_depth for bh in cr.collection}
    for _, row in raw.iterrows():
        try:
            bh_id = str(row["hole_id"]).strip()
            if bh_id not in bh_map:
                continue
            md = float(row["measured_depth"])
            if md > max_depths.get(bh_id, float("inf")):
                continue
            az = float(row.get("azimuth", 0)) % 360
            dip_val = max(-90.0, min(90.0, float(row.get("dip", -90))))
            existing = {s.measured_depth for s in bh_map[bh_id].survey.stations}
            if md in existing:
                continue
            bh_map[bh_id].survey.stations.append(SurveyStation(measured_depth=md, azimuth=az, dip=dip_val))
        except Exception:
            pass

    hole_ids = sorted(bh_map.keys())
    ho = HoldoutService()
    ho.select_manual(hole_ids, ["BH-02", "BH-07"])
    ho.update_config(validation_fraction=0.25)
    ho.lock()

    raw_di = pd.read_csv(demo_dir / "domain_intervals.csv")
    from dfn_cave_studio.models.data_management import DomainInterval

    intervals = []
    for _, row in raw_di.iterrows():
        intervals.append(
            DomainInterval(
                hole_id=str(row["hole_id"]).strip(),
                from_depth=float(row["from_depth"]),
                to_depth=float(row["to_depth"]),
                domain_id=int(row["domain_id"]),
                domain_name=str(row.get("domain_name", "")),
                assignment_method="imported",
            )
        )

    wf = WorkflowController()
    for step in ["import", "clean", "holdout", "domains", "joint_sets"]:
        wf.complete_step(step)

    project._m7_data = {
        "raw_surveys": raw,
        "raw_fractures": pd.read_csv(demo_dir / "fractures.csv"),
        "raw_rqd": pd.read_csv(demo_dir / "rqd.csv"),
        "raw_domain_intervals": raw_di,
        "excluded_records": cr.fracture_exclusions,
        "domain_intervals": intervals,
        "holdout": ho,
        "workflow": wf,
    }
    return project


def _set_combo_data(combo, data_value):
    """Set QComboBox to item with given data value."""
    for i in range(combo.count()):
        if combo.itemData(i) == data_value:
            combo.setCurrentIndex(i)
            return True
    return False


def _set_mapping_for_nonstandard_collars(dlg):
    """Manually map non-standard collars columns in the dialog's mapping table."""
    _MAPPING = {
        "borehole_id": "HoleName",
        "collar_x": "East",
        "collar_y": "North",
        "collar_z": "RL",
        "final_depth": "HoleLength",
        "azimuth": "Azimuth",
        "dip": "Dip",
    }
    for r in range(dlg._mapping_table.rowCount()):
        std = dlg._mapping_table.item(r, 0).text() if dlg._mapping_table.item(r, 0) else ""
        if std in _MAPPING:
            dlg._mapping_table.item(r, 1).setText(_MAPPING[std])


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CSV field mapping end-to-end
# ═══════════════════════════════════════════════════════════════════════════════


class TestCSVFieldMapping:
    """Non-standard CSV columns -> M7ImportDialog mapping -> project data."""

    def test_nonstandard_csv_imports_correct_boreholes(self, tmp_path, qtbot):
        """CSV with HoleName/East/North/RL/HoleLength + azimuth/dip mapped to standard."""
        from dfn_cave_studio.ui.dialogs.m7_import_dialog import M7ImportDialog

        # Create non-standard CSV with all columns BoreholeImporter needs
        csv_path = tmp_path / "collars_nonstandard.csv"
        csv_path.write_text(
            "HoleName,East,North,RL,HoleLength,Azimuth,Dip\n"
            "BH-A,100.0,200.0,500.0,150.0,0,-90\n"
            "BH-B,150.0,250.0,510.0,200.0,45,-85\n"
        )

        project = Project()
        project.metadata.name = "CSV Map Test"
        project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

        dlg = M7ImportDialog(project)
        qtbot.addWidget(dlg)

        dlg._path_edit.setText(str(csv_path))
        _set_combo_data(dlg._type_combo, "collars")

        # Preview
        dlg._on_preview()
        qtbot.wait(50)

        # Verify preview shows source column names
        preview_headers = [
            dlg._preview_table.horizontalHeaderItem(j).text() for j in range(dlg._preview_table.columnCount())
        ]
        assert "HoleName" in preview_headers, f"Preview headers: {preview_headers}"
        assert "RL" in preview_headers, f"Preview headers: {preview_headers}"

        # Verify mapping table has standard field names
        std_fields = []
        for r in range(dlg._mapping_table.rowCount()):
            std = dlg._mapping_table.item(r, 0).text() if dlg._mapping_table.item(r, 0) else ""
            std_fields.append(std)
        assert "borehole_id" in std_fields, f"Standard fields: {std_fields}"
        assert "collar_x" in std_fields, f"Standard fields: {std_fields}"

        # Manually set mapping for non-standard columns
        _set_mapping_for_nonstandard_collars(dlg)

        # Import
        dlg._on_import()
        qtbot.wait(50)

        summary = dlg.get_import_summary()
        collars_info = summary.get("collars", {})
        imported = collars_info.get("imported", 0)
        assert imported == 2, f"Should import 2 collars, got {imported}. Status: {dlg._status_label.text()}"

        # Accept and verify project
        dlg._on_accept()
        assert project.borehole_collection is not None
        bhs = list(project.borehole_collection)
        assert len(bhs) == 2, f"Should have 2 boreholes, got {len(bhs)}"

        bh_a = next((b for b in bhs if b.borehole_id == "BH-A"), None)
        assert bh_a is not None, "BH-A not found"
        assert abs(bh_a.collar.collar_x - 100.0) < 0.01
        assert abs(bh_a.collar.collar_y - 200.0) < 0.01
        assert abs(bh_a.collar.collar_z - 500.0) < 0.01
        assert abs(bh_a.collar.final_depth - 150.0) < 0.01

        dlg.close()

    def test_missing_required_blocks_import(self, tmp_path, qtbot):
        """Required fields unmapped -> import blocked."""
        from dfn_cave_studio.ui.dialogs.m7_import_dialog import M7ImportDialog

        csv_path = tmp_path / "bad_collars.csv"
        csv_path.write_text("A,B,C\n1,2,3\n")

        project = Project()
        project.metadata.name = "Block Test"
        project.model_bounds = ModelBounds(x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=100)

        dlg = M7ImportDialog(project)
        qtbot.addWidget(dlg)

        dlg._path_edit.setText(str(csv_path))
        _set_combo_data(dlg._type_combo, "collars")

        dlg._on_import()
        qtbot.wait(50)

        # No collars should have been imported (blocked by missing required fields)
        summary = dlg.get_import_summary()
        collars_info = summary.get("collars", {})
        # Import should have been blocked
        assert collars_info.get("imported", 0) == 0, "Import should be blocked when required fields are missing"

        dlg.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. XLSX import end-to-end
# ═══════════════════════════════════════════════════════════════════════════════


class TestXLSXImport:
    """XLSX files -> M7ImportDialog preview + import -> project data."""

    def _make_xlsx(self, tmp_path, name, data_rows, columns):
        """Create a minimal XLSX file with openpyxl."""
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet"  # Ensure sheet name matches default
        ws.append(columns)
        for row in data_rows:
            ws.append(row)
        path = tmp_path / name
        wb.save(str(path))
        return path

    def test_xlsx_collars_import(self, tmp_path, qtbot):
        """XLSX collars -> preview shows columns, import creates boreholes."""
        from dfn_cave_studio.ui.dialogs.m7_import_dialog import M7ImportDialog

        path = self._make_xlsx(
            tmp_path,
            "collars.xlsx",
            data_rows=[["BH-01", 100, 200, 500, 150, 0, -90], ["BH-02", 150, 250, 510, 200, 0, -90]],
            columns=["hole_id", "easting", "northing", "elevation", "total_depth", "azimuth", "dip"],
        )

        project = Project()
        project.metadata.name = "XLSX Collars"
        project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

        dlg = M7ImportDialog(project)
        qtbot.addWidget(dlg)
        dlg._path_edit.setText(str(path))
        _set_combo_data(dlg._type_combo, "collars")

        # Preview
        dlg._on_preview()
        qtbot.wait(50)
        preview_headers = [
            dlg._preview_table.horizontalHeaderItem(j).text() for j in range(dlg._preview_table.columnCount())
        ]
        assert "hole_id" in preview_headers, f"Preview headers: {preview_headers}"
        assert "easting" in preview_headers

        # Import
        dlg._on_import()
        qtbot.wait(50)
        dlg._on_accept()

        assert project.borehole_collection is not None
        assert len(project.borehole_collection) == 2
        dlg.close()

    def test_xlsx_all_five_types(self, tmp_path, qtbot):
        """Import all 5 XLSX data types through M7ImportDialog."""
        from dfn_cave_studio.ui.dialogs.m7_import_dialog import M7ImportDialog

        # Generate minimal XLSX files
        collars_path = self._make_xlsx(
            tmp_path,
            "collars.xlsx",
            [["BH-01", 100, 200, 500, 150, 0, -90]],
            ["hole_id", "easting", "northing", "elevation", "total_depth", "azimuth", "dip"],
        )

        surveys_path = self._make_xlsx(
            tmp_path,
            "surveys.xlsx",
            [["BH-01", 0, 0, -90], ["BH-01", 50, 0, -90], ["BH-01", 100, 0, -90]],
            ["hole_id", "measured_depth", "azimuth", "dip"],
        )

        fractures_path = self._make_xlsx(
            tmp_path,
            "fractures.xlsx",
            [["BH-01", 30, 45, 60, 1], ["BH-01", 80, 135, 75, 2]],
            ["hole_id", "depth", "dip_direction", "dip", "set_id"],
        )

        rqd_path = self._make_xlsx(
            tmp_path,
            "rqd.xlsx",
            [["BH-01", 0, 50, 85], ["BH-01", 50, 100, 72]],
            ["hole_id", "from_depth", "to_depth", "rqd"],
        )

        domain_path = self._make_xlsx(
            tmp_path,
            "domain_intervals.xlsx",
            [["BH-01", 0, 75, 1], ["BH-01", 75, 150, 2]],
            ["hole_id", "from_depth", "to_depth", "domain_id"],
        )

        project = Project()
        project.metadata.name = "XLSX All Five"
        project.model_bounds = ModelBounds(x_min=0, x_max=500, y_min=0, y_max=500, z_min=0, z_max=600)

        dlg = M7ImportDialog(project)
        qtbot.addWidget(dlg)

        # 1. Import collars
        dlg._path_edit.setText(str(collars_path))
        _set_combo_data(dlg._type_combo, "collars")
        dlg._on_import()
        qtbot.wait(50)

        # 2. Import surveys
        dlg._path_edit.setText(str(surveys_path))
        _set_combo_data(dlg._type_combo, "surveys")
        dlg._on_import()
        qtbot.wait(50)

        # 3. Import fractures
        dlg._path_edit.setText(str(fractures_path))
        _set_combo_data(dlg._type_combo, "fractures")
        dlg._on_import()
        qtbot.wait(50)

        # 4. Import RQD
        dlg._path_edit.setText(str(rqd_path))
        _set_combo_data(dlg._type_combo, "rqd")
        dlg._on_import()
        qtbot.wait(50)

        # 5. Import domain intervals
        dlg._path_edit.setText(str(domain_path))
        _set_combo_data(dlg._type_combo, "domain_intervals")
        dlg._on_import()
        qtbot.wait(50)

        # Accept and verify
        dlg._on_accept()

        # Collars: 1 borehole
        assert project.borehole_collection is not None
        assert len(project.borehole_collection) == 1

        # Surveys: 3 raw rows
        m7 = getattr(project, "_m7_data", {})
        assert len(m7.get("raw_surveys", [])) == 3

        # Fractures: 2 observations on BH-01
        bh = list(project.borehole_collection)[0]
        assert len(bh.fracture_observations) == 2

        # RQD: 2 rows
        assert len(m7.get("raw_rqd", [])) == 2

        # Domain intervals: 2 rows
        assert len(m7.get("raw_domain_intervals", [])) == 2

        dlg.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. StructuralDomain persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructuralDomainPersistence:
    """StructuralDomain definitions survive .dfnproj round-trip."""

    def test_domain_properties_roundtrip(self, demo_dir, tmp_path):
        """Custom domain name/color/notes + Domain 4 without intervals."""
        project = _build_full_project(demo_dir)

        sd = project.structural_domains
        # Ensure Domain 1 and 2 exist with custom properties
        domain1 = sd.get_domain(1)
        if domain1 is None:
            domain1 = StructuralDomain(domain_id=1, name="Domain 1")
            sd.domains.append(domain1)
        domain1.name = "Modified Domain 1"
        domain1.color = "#ff0000"
        domain1.description = "Custom notes for domain 1"

        domain2 = sd.get_domain(2)
        if domain2 is None:
            domain2 = StructuralDomain(domain_id=2, name="Domain 2")
            sd.domains.append(domain2)
        domain2.name = "Domain Two"
        domain2.color = "#00ff00"
        domain2.description = "Second domain notes"

        # Domain 4 — no intervals assigned yet
        domain4 = StructuralDomain(domain_id=4, name="Future Domain 4", color="#0000ff", description="No intervals yet")
        sd.domains.append(domain4)

        # Save
        save_path = tmp_path / "domains_test.dfnproj"
        zps = ZipProjectStore()
        zps.save(project, save_path)

        # Reopen
        reopened = zps.load(save_path)

        # Verify domain 1 restored
        rd1 = reopened.structural_domains.get_domain(1)
        assert rd1 is not None, "Domain 1 not restored"
        assert rd1.name == "Modified Domain 1", f"Name: {rd1.name}"
        assert rd1.color == "#ff0000", f"Color: {rd1.color}"
        assert rd1.description == "Custom notes for domain 1"

        # Verify domain 2 restored
        rd2 = reopened.structural_domains.get_domain(2)
        assert rd2 is not None, "Domain 2 not restored"
        assert rd2.name == "Domain Two"
        assert rd2.color == "#00ff00"

        # Verify domain 4 restored (no intervals assigned)
        rd4 = reopened.structural_domains.get_domain(4)
        assert rd4 is not None, "Domain 4 (no intervals) not restored"
        assert rd4.name == "Future Domain 4"
        assert rd4.color == "#0000ff"
        assert rd4.description == "No intervals yet"

        # Verify domain intervals survived
        di = get_domain_intervals(reopened)
        assert len(di) == 11

    def test_domain4_without_intervals_persists(self, demo_dir, tmp_path):
        """A domain with zero intervals survives round-trip."""
        project = _build_full_project(demo_dir)
        project.structural_domains.domains.append(
            StructuralDomain(domain_id=5, name="Empty Domain", color="#abcdef", description="Has no intervals")
        )

        save_path = tmp_path / "empty_domain.dfnproj"
        zps = ZipProjectStore()
        zps.save(project, save_path)

        reopened = zps.load(save_path)
        rd5 = reopened.structural_domains.get_domain(5)
        assert rd5 is not None, "Empty domain not restored"
        assert rd5.name == "Empty Domain"
        assert rd5.color == "#abcdef"
        assert rd5.description == "Has no intervals"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MainWindow save/open round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestMainWindowRoundTrip:
    """Real MainWindow: Save As -> New -> Open -> Save (mocked file dialogs)."""

    def test_save_as_new_open_save_preserves_all_state(self, demo_dir, tmp_path, qtbot):
        """End-to-end MainWindow save/open cycle with mocked file dialogs."""
        from dfn_cave_studio.ui.main_window import MainWindow

        save_path = tmp_path / "roundtrip_final.dfnproj"

        w = MainWindow()
        qtbot.addWidget(w)

        # Build full project and bind to store
        project = _build_full_project(demo_dir)

        # Set up custom structural domains
        project.structural_domains.domains = [
            StructuralDomain(domain_id=0, name="Global"),
            StructuralDomain(domain_id=1, name="Custom D1", color="#aa0000", description="Primary domain"),
            StructuralDomain(domain_id=2, name="Custom D2", color="#00aa00", description="Secondary domain"),
            StructuralDomain(domain_id=3, name="Custom D3", color="#0000aa"),
            StructuralDomain(domain_id=4, name="Empty D4", color="#aaaa00", description="No intervals"),
        ]

        w._project_store.adopt_project(project)
        w._restore_project_to_ui(project)

        # ── Save As ─────────────────────────────────────────────────
        with patch.object(
            QFileDialog, "getSaveFileName", return_value=(str(save_path), "DFN Cave Studio ZIP Projects (*.dfnproj)")
        ):
            w._on_save_project_as()
        qtbot.wait(100)

        assert save_path.exists(), f"Save file not created: {save_path}"
        assert w._project_store.current_path == save_path
        assert w._project_store.is_dirty is False

        # ── New Project ────────────────────────────────────────────
        w._on_new_project()
        qtbot.wait(100)

        new_proj = w._project_store.current_project
        assert new_proj is not project
        new_m7 = getattr(new_proj, "_m7_data", {}) or {}
        assert new_m7.get("holdout") is None

        # ── Open ───────────────────────────────────────────────────
        with patch.object(
            QFileDialog, "getOpenFileName", return_value=(str(save_path), "DFN Cave Studio ZIP Projects (*.dfnproj)")
        ):
            w._on_open_project()
        qtbot.wait(100)

        loaded = w._project_store.current_project
        assert loaded is not None
        assert w._project_store.current_project is loaded
        assert w._project_store.current_path == save_path
        assert w._project_store.is_dirty is False

        # ── Verify all data restored ────────────────────────────────
        assert loaded.borehole_collection is not None
        assert len(loaded.borehole_collection) == 8
        stations = sum(len(bh.survey.stations) for bh in loaded.borehole_collection)
        assert stations == 148, f"Stations: {stations}"
        fracs = sum(len(bh.fracture_observations) for bh in loaded.borehole_collection)
        assert fracs == 80, f"Fractures: {fracs}"

        di = get_domain_intervals(loaded)
        assert len(di) == 11

        # Structural domains with custom properties
        sd = loaded.structural_domains
        rd1 = sd.get_domain(1)
        assert rd1 is not None and rd1.name == "Custom D1"
        assert rd1.color == "#aa0000" and rd1.description == "Primary domain"

        rd4 = sd.get_domain(4)
        assert rd4 is not None, "Domain 4 (no intervals) not restored"
        assert rd4.name == "Empty D4"

        # Holdout
        ho = get_holdout(loaded)
        assert ho is not None and ho.is_locked
        assert len(ho.calibration_holes) == 6
        assert len(ho.validation_holes) == 2
        assert "BH-02" in ho.validation_holes
        assert "BH-07" in ho.validation_holes
        assert abs(ho.config.validation_fraction - 0.25) < 0.01

        # Workflow
        wf = get_workflow(loaded)
        assert wf is not None
        for step in ["import", "clean", "holdout", "domains", "joint_sets"]:
            assert wf.is_step_done(step), f"Step '{step}' not done"

        # ── Save again to same path ─────────────────────────────────
        w._on_save_project()
        qtbot.wait(100)
        assert w._project_store.current_path == save_path
        assert w._project_store.is_dirty is False
        assert save_path.exists()

        w.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Joint Set reproducibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestJointSetReproducibility:
    """Same seed + same data -> same assignments and sorted counts."""

    def test_identical_seed_produces_identical_counts(self, demo_dir):
        """Run identify_auto twice with seed=42, assert same results."""
        project = _build_full_project(demo_dir)
        collection = project.borehole_collection
        ho = get_holdout(project)
        cal_holes = set(ho.calibration_holes)
        val_holes = set(ho.validation_holes)

        # First run
        js1 = JointSetService(random_seed=42)
        result1 = js1.identify_auto(collection, cal_holes, val_holes, n_clusters=3, random_seed=42)

        # Second run
        js2 = JointSetService(random_seed=42)
        result2 = js2.identify_auto(collection, cal_holes, val_holes, n_clusters=3, random_seed=42)

        # Verify counts
        assert result1.calibration_count == 60
        assert result2.calibration_count == 60
        assert result1.validation_count == 20
        assert result2.validation_count == 20

        # Total assigned
        total1 = sum(len([a for a in result1.assignments.values() if a == sid]) for sid in result1.sets)
        total2 = sum(len([a for a in result2.assignments.values() if a == sid]) for sid in result2.sets)
        assert total1 == 60 and total2 == 60

        # Per-set counts must be reproducible without hard-coding the split.
        counts1 = sorted(len([a for a in result1.assignments.values() if a == sid]) for sid in result1.sets)
        counts2 = sorted(len([a for a in result2.assignments.values() if a == sid]) for sid in result2.sets)
        assert counts1 == counts2, f"Run1={counts1}, Run2={counts2}"

        # Per-record assignments identical
        for rec_id in result1.assignments:
            assert result1.assignments[rec_id] == result2.assignments.get(rec_id), f"Assignment mismatch for {rec_id}"

    def test_different_seed_assigns_all_calibration_fractures(self, demo_dir):
        """Different seed still assigns all 60 calibration fractures."""
        project = _build_full_project(demo_dir)
        collection = project.borehole_collection
        ho = get_holdout(project)
        cal_holes = set(ho.calibration_holes)
        val_holes = set(ho.validation_holes)

        js = JointSetService(random_seed=42)
        result = js.identify_auto(collection, cal_holes, val_holes, n_clusters=3, random_seed=999)

        total = sum(len([a for a in result.assignments.values() if a == sid]) for sid in result.sets)
        assert total == 60
        assert result.validation_count == 20
