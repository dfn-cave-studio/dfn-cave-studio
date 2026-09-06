"""GUI regressions for generic M9 scalar interpolation settings."""

from __future__ import annotations

import numpy as np
import pytest

from dfn_cave_studio.models.m9 import DensityMethod, DensitySettings, ScalarFieldMetadata, ScalarFieldResult, ScalarParameterSample
from dfn_cave_studio.models.borehole_database import BoreholeRecord, RecordState
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.spatial_grid import SpatialGridConfig
from dfn_cave_studio.ui.main_window import MainWindow
from dfn_cave_studio.ui.dialogs.m9_scalar_field_dialog import M9ScalarFieldDialog
from dfn_cave_studio.ui.dialogs.m9_dialogs import M9DensityDialog
from dfn_cave_studio.ui.qt_adapter import QDialogButtonBox, QMessageBox
from dfn_cave_studio.services.workflow_controller import WorkflowController
from dfn_cave_studio.voxel.parameter_field import CELL_STATE_CODES
from dfn_cave_studio.models.spatial_grid import VoxelCellState


def _sample() -> ScalarParameterSample:
    return ScalarParameterSample(sample_id="synthetic-1", borehole_id="H1", from_depth=0, to_depth=1,
                                 parameter_name="UCS", value=10, unit="MPa", midpoint_x=0,
                                 midpoint_y=0, midpoint_z=0, role="calibration")


def _candidate(field_id: str = "late") -> ScalarFieldResult:
    return ScalarFieldResult(
        metadata=ScalarFieldMetadata(
            field_id=field_id,
            parameter_name="UCS",
            unit="MPa",
            method="global_constant",
            shape=(1, 1, 1),
            origin=(0, 0, 0),
            spacing=(1, 1, 1),
            array_names=[],
            config_hash=field_id,
        ),
        settings=DensitySettings(),
    )


class _CancellableWorker:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def test_method_and_variogram_combos_use_stable_item_data(qtbot) -> None:
    project = Project(); project.m9_state.scalar_samples = [_sample()]
    dialog = M9ScalarFieldDialog(project); qtbot.addWidget(dialog)
    dialog.method.setCurrentIndex(dialog.method.findData(DensityMethod.ORDINARY_KRIGING.value))
    dialog.mode.setCurrentIndex(dialog.mode.findData("manual"))
    assert dialog.method.currentData() == "ordinary_kriging"
    assert dialog.mode.currentData() == "manual"
    assert dialog.nugget.isEnabled()
    dialog.retranslateUi = lambda: None  # display text may change; stable values must not
    assert dialog.method.currentData() == "ordinary_kriging"


def test_idw_minimum_one_does_not_validate_unused_kriging_controls(qtbot) -> None:
    project = Project(); project.m9_state.scalar_samples = [_sample()]
    dialog = M9ScalarFieldDialog(project); qtbot.addWidget(dialog)
    dialog.method.setCurrentIndex(dialog.method.findData(DensityMethod.IDW.value))
    dialog.minimum.setValue(1)
    settings = dialog._settings()
    assert settings.method == DensityMethod.IDW
    assert settings.min_neighbors == 1
    assert settings.kriging.minimum_neighbors >= 2


def test_manual_kriging_settings_and_display_fields_use_stable_values(qtbot) -> None:
    project = Project(); project.m9_state.scalar_samples = [_sample()]
    dialog = M9ScalarFieldDialog(project); qtbot.addWidget(dialog)
    dialog.method.setCurrentIndex(dialog.method.findData(DensityMethod.ORDINARY_KRIGING.value))
    dialog.mode.setCurrentIndex(dialog.mode.findData("manual"))
    dialog.nugget.setValue(2.0)
    dialog.sill.setValue(1.0)
    with pytest.raises(ValueError, match="sill"):
        dialog._settings()
    assert {dialog.display.itemData(index) for index in range(dialog.display.count())} == {
        "estimate", "kriging_variance"
    }


def test_cancel_restores_uncommitted_scalar_samples(qtbot) -> None:
    project = Project(); dialog = M9ScalarFieldDialog(project); qtbot.addWidget(dialog)
    project.m9_state.scalar_samples.append(_sample()); dialog.committed_changes = True
    dialog.reject()
    assert project.m9_state.scalar_samples == []
    dialog._done(_candidate())
    assert project.m9_state.scalar_fields == []


def test_cancel_discards_late_candidate_before_requesting_worker_stop(qtbot) -> None:
    project = Project()
    dialog = M9ScalarFieldDialog(project)
    qtbot.addWidget(dialog)
    worker = _CancellableWorker()
    dialog._worker = worker

    dialog._cancel()
    assert worker.cancelled is True
    assert dialog._discard_worker_result is True
    dialog._done(_candidate())

    assert project.m9_state.scalar_fields == []


def test_new_computation_resets_cancel_discard_guard(qtbot, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project()
    project.m9_state.scalar_samples = [_sample()]
    dialog = M9ScalarFieldDialog(project)
    qtbot.addWidget(dialog)
    dialog._discard_worker_result = True

    class _Pool:
        @staticmethod
        def start(_worker) -> None:
            return None

    class _ThreadPool:
        @staticmethod
        def globalInstance() -> _Pool:
            return _Pool()

    monkeypatch.setattr("dfn_cave_studio.ui.dialogs.m9_scalar_field_dialog.QThreadPool", _ThreadPool)
    dialog._build()
    assert dialog._discard_worker_result is False
    dialog._done(_candidate("new-computation"))

    assert [item.metadata.field_id for item in project.m9_state.scalar_fields] == ["new-computation"]


def test_scalar_field_over_budget_requires_confirmation_before_worker(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project()
    project.m9_state.scalar_samples = [_sample()]
    bounds = ModelBounds(x_min=0, x_max=300, y_min=0, y_max=300, z_min=0, z_max=300)
    project.spatial_grid_config = SpatialGridConfig(analysis_domain=bounds, generation_domain=bounds)
    project.voxel_config = VoxelConfig()
    dialog = M9ScalarFieldDialog(project)
    qtbot.addWidget(dialog)
    dialog.memory_budget_gib.setValue(0.25)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args: warnings.append(_args) or QMessageBox.StandardButton.No)
    dialog._build()
    assert warnings
    assert dialog._worker is None
    assert "27,000,000 voxels" in dialog.summary.text()


def test_cancelled_worker_cannot_commit_after_replacement_worker_starts(
    qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project()
    project.m9_state.scalar_samples = [_sample()]
    dialog = M9ScalarFieldDialog(project)
    qtbot.addWidget(dialog)
    old_worker = _CancellableWorker()
    dialog._worker = old_worker
    dialog._cancel()

    class _Pool:
        @staticmethod
        def start(_worker) -> None:
            return None

    class _ThreadPool:
        @staticmethod
        def globalInstance() -> _Pool:
            return _Pool()

    monkeypatch.setattr("dfn_cave_studio.ui.dialogs.m9_scalar_field_dialog.QThreadPool", _ThreadPool)
    dialog._build()
    new_worker = dialog._worker

    dialog._done(_candidate("stale"), old_worker)
    assert project.m9_state.scalar_fields == []
    assert dialog._worker is new_worker
    dialog._done(_candidate("current"), new_worker)
    assert [item.metadata.field_id for item in project.m9_state.scalar_fields] == ["current"]


def test_accept_is_blocked_while_worker_can_still_deliver_result(qtbot) -> None:
    project = Project()
    dialog = M9ScalarFieldDialog(project)
    qtbot.addWidget(dialog)
    worker = _CancellableWorker()
    dialog._worker = worker
    dialog._set_running(True)

    dialog.accept()
    assert dialog.result() == 0
    assert dialog.dialog_buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled() is False

    dialog.reject()
    dialog._done(_candidate())
    assert worker.cancelled is True
    assert project.m9_state.scalar_fields == []


def test_cancel_restores_scalar_database_audit_rows(qtbot) -> None:
    project = Project(); dialog = M9ScalarFieldDialog(project); qtbot.addWidget(dialog)
    project.borehole_database.records.append(
        BoreholeRecord(data_type="scalar_parameters", hole_id="H1", source_file="synthetic.csv",
                       source_row=2, original_values={"value": 1.0}, values={"value": 1.0},
                       state=RecordState.FORMAL)
    )
    dialog.reject()
    assert project.borehole_database.records == []


def test_window_is_resizable_and_fits_compact_screen(qtbot) -> None:
    dialog = M9ScalarFieldDialog(Project()); qtbot.addWidget(dialog)
    assert dialog.minimumWidth() < 1000
    assert dialog.minimumHeight() < 800
    assert dialog.size().width() <= 900


def test_cloud_slice_box_and_layer_controls_use_existing_m9_registry(qtbot) -> None:
    project = Project(); project.m9_state.scalar_samples = [_sample()]
    arrays = {
        "estimate": np.arange(27, dtype=np.float32).reshape(3, 3, 3),
        "kriging_variance": np.ones((3, 3, 3), dtype=np.float32),
        "cell_state": np.full((3, 3, 3), CELL_STATE_CODES[VoxelCellState.MODELED_VALUE], dtype=np.uint8),
    }
    project.m9_state.scalar_fields = [ScalarFieldResult(
        metadata=ScalarFieldMetadata(field_id="ucs", parameter_name="UCS", unit="MPa", method="ordinary_kriging",
                                     shape=(3, 3, 3), origin=(0, 0, 0), spacing=(1, 1, 1),
                                     array_names=sorted(arrays), config_hash="synthetic"),
        settings=DensitySettings(method="ordinary_kriging"), arrays=arrays)]
    window = MainWindow(); qtbot.addWidget(window); window._project_store.adopt_project(project)
    manager = window._get_m9_layer_manager(); dialog = M9ScalarFieldDialog(project, manager, window); qtbot.addWidget(dialog)
    for mode in ("outer_surface", "orthogonal_section", "box_cutaway"):
        dialog.display_mode.setCurrentIndex(dialog.display_mode.findData(mode)); dialog._render()
    records = [item for item in manager.list_layers() if item.layer_id.startswith("m9_slice:scalar_")]
    assert len(records) == 3
    dialog.layers.selectRow(0); dialog._toggle_layer(); assert records[0].visible is False
    dialog._remove_layer(); assert len([item for item in manager.list_layers() if item.layer_id.startswith("m9_slice:scalar_")]) == 2
    dialog._clear_layers(); assert not [item for item in manager.list_layers() if item.layer_id.startswith("m9_slice:scalar_")]


def test_p32_density_dialog_restores_ordinary_kriging_settings(qtbot) -> None:
    project = Project(); project.m9_state.density_settings = DensitySettings(
        method="ordinary_kriging", kriging={"mode": "manual", "model": "gaussian", "nugget": 0.2,
                                            "sill": 3.0, "range": 44.0, "minimum_neighbors": 4,
                                            "maximum_neighbors": 9, "search_radius": 88.0})
    dialog = M9DensityDialog(project, WorkflowController()); qtbot.addWidget(dialog)
    assert dialog.method.currentData() == "ordinary_kriging"
    assert dialog.variogram_mode.currentData() == "manual"
    assert dialog.variogram_model.currentData() == "gaussian"
    assert dialog.nugget.value() == 0.2
    assert dialog.variogram_range.value() == 44.0
