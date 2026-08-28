"""Real M8 preview/stage GUI regression for M10 collar aliases."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dfn_cave_studio.models.project import Project
from dfn_cave_studio.ui.dialogs.m7_import_dialog import M7ImportDialog
from dfn_cave_studio.ui.dialogs.m8_import_dialog import M8ImportDialog
from dfn_cave_studio.ui.qt_adapter import QComboBox, QMessageBox

M10_COLLARS = Path(__file__).parents[2] / "examples/m10_demo/collars.csv"
M7_DEMO = Path(__file__).parents[2] / "examples/m7_demo"


def _m8_mapping(dialog: M8ImportDialog) -> dict[str, str]:
    return {
        dialog._mapping_table.item(row, 0).text(): dialog._mapping_table.cellWidget(row, 1).currentText()
        for row in range(dialog._mapping_table.rowCount())
        if isinstance(dialog._mapping_table.cellWidget(row, 1), QComboBox)
        and dialog._mapping_table.cellWidget(row, 1).currentText()
    }


def _select_data_type(dialog: M8ImportDialog, data_type: str) -> None:
    index = dialog._type_combo.findData(data_type)
    assert index >= 0
    dialog._type_combo.setCurrentIndex(index)


def test_m10_collars_preview_and_stage_use_the_same_automatic_mapping(qtbot):
    project = Project()
    dialog = M8ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._path_edit.setText(str(M10_COLLARS))

    dialog._load_preview()
    preview_mapping = _m8_mapping(dialog)
    assert preview_mapping == {
        "HoleName": "borehole_id",
        "East": "collar_x",
        "North": "collar_y",
        "RL": "collar_z",
        "HoleLength": "final_depth",
    }

    dialog._stage_import()
    assert dialog.results[-1]["formal"] == 8
    assert project.borehole_database.counts("collars") == {
        "raw": 8,
        "formal": 8,
        "excluded": 0,
        "pending": 0,
    }
    records = project.borehole_database.query("collars", raw=True)
    assert all(record.source_field_mapping == preview_mapping for record in records)
    actual = {
        borehole.borehole_id: (
            borehole.collar.collar_x,
            borehole.collar.collar_y,
            borehole.collar.collar_z,
            borehole.collar.final_depth,
        )
        for borehole in project.borehole_collection
    }
    assert actual["BH-01"] == (100.0, 200.0, 502.7, 150.0)
    assert actual["BH-08"] == (450.0, 410.0, 502.9, 160.0)


def test_standard_field_cells_have_one_combo_after_ten_previews_for_every_data_type(qtbot):
    dialog = M8ImportDialog(Project())
    qtbot.addWidget(dialog)
    paths = {
        "collars": M10_COLLARS,
        "surveys": M7_DEMO / "surveys.csv",
        "fractures": M7_DEMO / "fractures.csv",
        "rqd": M7_DEMO / "rqd.csv",
        "domain_intervals": M7_DEMO / "domain_intervals.csv",
    }
    for data_type, path in paths.items():
        _select_data_type(dialog, data_type)
        dialog._path_edit.setText(str(path))
        for _ in range(10):
            dialog._load_preview()
            qtbot.wait(1)
            row_count = dialog._mapping_table.rowCount()
            combos = dialog._mapping_table.findChildren(QComboBox)
            assert len(combos) == row_count
            for row in range(row_count):
                assert isinstance(dialog._mapping_table.cellWidget(row, 1), QComboBox)
                assert dialog._mapping_table.item(row, 1) is None


def test_switching_collars_and_surveys_removes_old_mapping_widgets(qtbot):
    dialog = M8ImportDialog(Project())
    qtbot.addWidget(dialog)
    for data_type, path in (
        ("collars", M10_COLLARS),
        ("surveys", M7_DEMO / "surveys.csv"),
        ("collars", M10_COLLARS),
    ):
        _select_data_type(dialog, data_type)
        assert dialog._mapping_table.rowCount() == 0
        assert dialog._mapping_table.findChildren(QComboBox) == []
        dialog._path_edit.setText(str(path))
        dialog._load_preview()
        qtbot.wait(1)
        assert len(dialog._mapping_table.findChildren(QComboBox)) == dialog._mapping_table.rowCount()


def test_m7_dialog_uses_the_same_m10_collar_alias_mapping(qtbot):
    project = Project()
    dialog = M7ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._path_edit.setText(str(M10_COLLARS))
    dialog._on_preview()

    mapping = {
        dialog._mapping_table.item(row, 0).text(): dialog._mapping_table.item(row, 1).text()
        for row in range(dialog._mapping_table.rowCount())
        if dialog._mapping_table.item(row, 1).text()
    }
    assert mapping["borehole_id"] == "HoleName"
    assert mapping["collar_x"] == "East"
    assert mapping["collar_y"] == "North"
    assert mapping["collar_z"] == "RL"
    assert mapping["final_depth"] == "HoleLength"
    dialog._on_import()
    assert dialog.get_import_summary()["collars"]["imported"] == 8


def test_m8_dialog_blocks_missing_required_and_reports_ambiguous_mapping(qtbot, tmp_path, monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda _parent, _title, message: messages.append(str(message)))

    ambiguous = tmp_path / "ambiguous.csv"
    ambiguous.write_text(
        "hole_id,borehole_id,easting,northing,elevation,total_depth\nA,B,1,2,3,4\n",
        encoding="utf-8",
    )
    dialog = M8ImportDialog(Project())
    qtbot.addWidget(dialog)
    dialog._path_edit.setText(str(ambiguous))
    dialog._load_preview()
    assert "Mapping conflict" in dialog._status.text()
    assert "multiple candidate columns" in dialog._status.text()

    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text("hole_id,easting,northing,elevation\nA,1,2,3\n", encoding="utf-8")
    dialog._path_edit.setText(str(incomplete))
    dialog._load_preview()
    dialog._stage_import()
    assert messages
    assert "Missing required fields: final_depth" in messages[-1]
    assert dialog._repository.database.counts("collars")["raw"] == 0
