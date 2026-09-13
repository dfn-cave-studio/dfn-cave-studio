"""GUI mapping and transaction tests for multi-source observation imports."""

from __future__ import annotations

import pandas as pd
import numpy as np

from dfn_cave_studio.models.borehole_database import FractureObservationMode
from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, ScalarFieldMetadata, ScalarFieldResult
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.ui.dialogs.m8_import_dialog import M8ImportDialog
from dfn_cave_studio.ui.qt_adapter import QComboBox, QPushButton, Qt


def _button(dialog: M8ImportDialog, text: str) -> QPushButton:
    return next(button for button in dialog.findChildren(QPushButton) if button.text() == text)


def _project() -> Project:
    project = Project()
    BoreholeRepository(project).import_dataframe(
        "collars",
        pd.DataFrame(
            [{"borehole_id": "SYN-1", "collar_x": 0, "collar_y": 0, "collar_z": 10, "final_depth": 10}]
        ),
        "synthetic-collars.csv",
    )
    return project


def test_spacing_mode_template_mapping_unit_and_cancel_are_transactional(qtbot, tmp_path) -> None:
    path = tmp_path / "synthetic-spacing.csv"
    pd.DataFrame(
        [{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "fracture_spacing": 10, "set_id": None}]
    ).to_csv(path, index=False)
    project = _project()
    dialog = M8ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._type_combo.setCurrentText("fractures")
    dialog._fracture_mode_combo.setCurrentIndex(
        dialog._fracture_mode_combo.findData(FractureObservationMode.INTERVAL_SPACING)
    )
    dialog._path_edit.setText(str(path))
    qtbot.mouseClick(_button(dialog, "Preview"), Qt.MouseButton.LeftButton)
    mapping = {
        dialog._mapping_table.item(row, 0).text(): dialog._mapping_table.cellWidget(row, 1).currentText()
        for row in range(dialog._mapping_table.rowCount())
        if isinstance(dialog._mapping_table.cellWidget(row, 1), QComboBox)
    }
    assert mapping["fracture_spacing"] == "fracture_spacing"
    assert "spacing: selected unit" in dialog._template.text()
    qtbot.mouseClick(_button(dialog, "Stage This Import"), Qt.MouseButton.LeftButton)
    assert project.borehole_database.counts("fractures")["formal"] == 1
    assert project.borehole_database.query("fractures", "formal")[0].values["derived_p10"] == 10.0
    dialog.reject()
    assert project.borehole_database.counts("fractures")["raw"] == 0


def test_orientation_point_aliases_map_without_coordinate_deduplication(qtbot, tmp_path) -> None:
    path = tmp_path / "synthetic-points.csv"
    pd.DataFrame(
        [
            {"observation_id": "P001-A", "point_id": "001", "E": 1, "N": 2, "R": 3, "dip": 20, "dip_direction": 30, "local_set_id": "A", "joint_spacing_m": 0.5, "joint_num": 2},
            {"observation_id": "P001-B", "point_id": "001", "E": 1, "N": 2, "R": 3, "dip": 40, "dip_direction": 50, "local_set_id": "B", "joint_spacing_m": 1.0, "joint_num": 1},
        ]
    ).to_csv(path, index=False)
    project = Project()
    dialog = M8ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._type_combo.setCurrentText("orientation_points")
    dialog._path_edit.setText(str(path))
    qtbot.mouseClick(_button(dialog, "Preview"), Qt.MouseButton.LeftButton)
    mapping = {
        dialog._mapping_table.item(row, 0).text(): dialog._mapping_table.cellWidget(row, 1).currentText()
        for row in range(dialog._mapping_table.rowCount())
    }
    assert mapping | {"E": "x", "N": "y", "R": "z"} == mapping
    qtbot.mouseClick(_button(dialog, "Stage This Import"), Qt.MouseButton.LeftButton)
    assert project.borehole_database.counts("orientation_points")["formal"] == 2


def test_rmr_import_synchronizes_generic_scalar_samples_and_cancel_restores(qtbot, tmp_path) -> None:
    path = tmp_path / "synthetic-rmr.csv"
    pd.DataFrame([{"hole_id": "SYN-1", "from_depth": 0, "to_depth": 5, "rmr": 0}]).to_csv(path, index=False)
    project = _project()
    project.m9_state.scalar_fields = [
        ScalarFieldResult(
            metadata=ScalarFieldMetadata(
                field_id="rmr-field",
                parameter_name="rmr",
                unit="score",
                method=DensityMethod.GLOBAL_CONSTANT,
                shape=(1, 1, 1),
                origin=(0, 0, 0),
                spacing=(1, 1, 1),
                array_names=["estimate"],
                config_hash="original",
            ),
            settings=DensitySettings(),
            arrays={"estimate": np.asarray([[[50.0]]], dtype=np.float32)},
        )
    ]
    dialog = M8ImportDialog(project)
    qtbot.addWidget(dialog)
    dialog._type_combo.setCurrentText("rmr")
    dialog._path_edit.setText(str(path))
    qtbot.mouseClick(_button(dialog, "Preview"), Qt.MouseButton.LeftButton)
    qtbot.mouseClick(_button(dialog, "Stage This Import"), Qt.MouseButton.LeftButton)
    assert [(item.parameter_name, item.value) for item in project.m9_state.scalar_samples] == [("rmr", 0.0)]
    assert project.m9_state.scalar_fields == []
    dialog.reject()
    assert project.m9_state.scalar_samples == []
    assert [field.metadata.field_id for field in project.m9_state.scalar_fields] == ["rmr-field"]
    np.testing.assert_array_equal(project.m9_state.scalar_fields[0].arrays["estimate"], np.asarray([[[50.0]]]))
