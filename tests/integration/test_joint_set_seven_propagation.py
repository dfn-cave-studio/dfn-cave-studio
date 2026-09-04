"""End-to-end M9-field to M10 propagation for dynamic joint-set IDs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyvista as pv
import pytest

from dfn_cave_studio.dfn.m10_generator import SOURCE_CODES
from dfn_cave_studio.models.fracture_set import JointSetConfig, OrientationDistribution
from dfn_cave_studio.models.m10 import M10GenerationConfig, M10FractureSource
from dfn_cave_studio.models.m9 import SizeModel, SizeModelSource
from dfn_cave_studio.models.m9 import DensitySettings
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
from dfn_cave_studio.services.m10_service import M10Service
from dfn_cave_studio.services.borehole_quality_service import BoreholeQualityService
from dfn_cave_studio.services.borehole_repository import BoreholeRepository
from dfn_cave_studio.services.holdout_service import HoldoutService
from dfn_cave_studio.services.joint_set_service import JointSetService
from dfn_cave_studio.services.m7_state import set_holdout
from dfn_cave_studio.services.m9_service import M9Service
from dfn_cave_studio.visualization.dfn_layer_manager import (
    DFNLayerManager,
    stable_category_hex,
)
from tests.integration.test_m10_persistence_export import make_m10_project


DEMO = Path(__file__).parents[2] / "examples" / "m7_demo"


class FakePlotter:
    """Minimal actor registry for layer tests without OpenGL."""

    def __init__(self):
        self.actors = {}
        self.remove_actor = MagicMock(side_effect=self._remove)
        self.render = MagicMock()

    def add_mesh(self, mesh, *, name, **kwargs):
        del mesh, kwargs
        actor = MagicMock()
        self.actors[name] = actor
        return actor

    def _remove(self, actor, render=False):
        del render
        for name, candidate in list(self.actors.items()):
            if candidate is actor:
                self.actors.pop(name)


def _seven_set_project(set_ids=(1, 2, 3, 4, 5, 6, 7)):
    project = make_m10_project()
    shape = project.m9_state.parameter_field_metadata.shape
    project.joint_sets = []
    project.m9_state.size_models = []
    project.m9_state.parameter_field_metadata.set_ids = list(set_ids)
    arrays = project.m9_state.parameter_field_arrays
    for name in [name for name in arrays if name.startswith("set_")]:
        arrays.pop(name)
    total = np.zeros(shape, dtype=np.float32)
    for index, set_id in enumerate(set_ids):
        p32 = np.full(shape, 0.04 + index * 0.002, dtype=np.float32)
        arrays[f"set_{set_id}_p32"] = p32
        arrays[f"set_{set_id}_dip_direction"] = np.full(shape, (index * 47.0) % 360.0, dtype=np.float32)
        arrays[f"set_{set_id}_dip"] = np.full(shape, 35.0 + index * 5.0, dtype=np.float32)
        arrays[f"set_{set_id}_kappa"] = np.full(shape, 60.0, dtype=np.float32)
        total += p32
        project.joint_sets.append(
            JointSetConfig(
                set_id=set_id,
                name=f"Synthetic Set {set_id}",
                orientation=OrientationDistribution(
                    mean_dip_direction=(index * 47.0) % 360.0,
                    mean_dip=35.0 + index * 5.0,
                    kappa=60.0,
                ),
            )
        )
        project.m9_state.size_models.append(
            SizeModel(
                domain_id=1,
                set_id=set_id,
                distribution_type="fixed",
                parameters={"radius": 1.0},
                min_radius=1.0,
                max_radius=1.0,
                mean_radius=1.0,
                mean_squared_radius=1.0,
                source=SizeModelSource.ASSUMED,
            )
        )
    arrays["p32_total"] = total
    project.m10_state.deterministic_structures = []
    return project


def _generate(project):
    config = M10GenerationConfig(
        base_seed=42,
        condition_calibration_observations=False,
        enabled_size_classes=["SMALL", "MEDIUM", "LARGE"],
    )
    return M10Service(project).generate_batch(config)[0]


def test_real_demo_k7_assignments_are_the_same_ids_consumed_by_m9() -> None:
    """The real M7 demo crosses the canonical M8 repository-to-M9 boundary."""
    project = Project()
    repository = BoreholeRepository(project)
    for name in ("surveys", "collars", "fractures", "rqd", "domain_intervals"):
        repository.import_dataframe(name, pd.read_csv(DEMO / f"{name}.csv"), str(DEMO / f"{name}.csv"))
    quality = BoreholeQualityService(project)
    quality.run_checks()
    quality.apply_auto_fixes()
    quality.confirm_exclusions()
    holdout = HoldoutService()
    holdout.select_manual(
        sorted(hole.borehole_id for hole in project.borehole_collection),
        ["BH-07", "BH-08"],
    )
    holdout.lock()
    set_holdout(project, holdout)
    result = JointSetService().identify_auto(
        project.borehole_collection,
        set(holdout.calibration_holes),
        set(holdout.validation_holes),
        n_clusters=7,
        random_seed=42,
    )
    repository.apply_joint_set_assignments({**result.assignments, **result.validation_assignments})
    project.joint_sets = list(result.sets.values())

    M9Service(project).calculate_density(DensitySettings(monte_carlo_samples=100))

    assert {item.set_id for item in project.joint_sets} == set(range(1, 8))
    assert {
        observation.set_id
        for hole in project.borehole_collection
        for observation in hole.fracture_observations
    } == set(range(1, 8))
    assert {row.set_id for row in project.m9_state.p10_intervals} == set(range(1, 8))
    assert {row.set_id for row in project.m9_state.p32_estimates} == set(range(1, 8))


def test_seven_complete_parameter_sets_generate_seven_nonempty_groups() -> None:
    project = _seven_set_project()
    realization = _generate(project)
    random_mask = realization.geometry_arrays["source_code"] == SOURCE_CODES[M10FractureSource.STOCHASTIC.value]
    set_ids, counts = np.unique(realization.geometry_arrays["set_id"][random_mask], return_counts=True)

    assert set_ids.tolist() == list(range(1, 8))
    assert np.all(counts > 0)
    diagnostics = M10Service(project).joint_set_diagnostics(realization=realization)
    assert [row["set_id"] for row in diagnostics] == list(range(1, 8))
    assert all(row["expected"] > 0 and row["actual"] > 0 for row in diagnostics)
    assert all(row["unresolved_reason"] == "" for row in diagnostics)


def test_seven_set_render_save_reopen_and_exports_preserve_ids(tmp_path: Path) -> None:
    project = _seven_set_project()
    realization = _generate(project)
    plotter = FakePlotter()
    manager = DFNLayerManager(plotter)
    for set_id in range(1, 8):
        manager.render_realization(realization, set_id=set_id)
    assert len(manager.list_layers()) == len(plotter.actors) == 7
    assert len({stable_category_hex(set_id) for set_id in range(1, 8)}) == 7

    project_path = tmp_path / "seven-sets.dfnproj"
    ZipProjectStore().save(project, project_path)
    restored = ZipProjectStore().load(project_path)
    assert restored.m9_state.parameter_field_metadata.set_ids == list(range(1, 8))
    np.testing.assert_array_equal(
        restored.m10_state.realizations[0].geometry_arrays["set_id"],
        realization.geometry_arrays["set_id"],
    )

    export_dir = tmp_path / "export"
    M10Service(restored).export_realization(realization.realization_id, export_dir)
    with (export_dir / "fractures.csv").open(encoding="utf-8", newline="") as handle:
        exported_ids = {int(row["set_id"]) for row in csv.DictReader(handle) if int(row["set_id"]) > 0}
    with np.load(export_dir / "fractures.npz", allow_pickle=False) as archive:
        npz_ids = set(map(int, archive["set_id"])) - {0}
    summary = json.loads((export_dir / "realization_summary.json").read_text(encoding="utf-8"))
    summary_ids = {int(item["set_id"]) for item in summary["quality"]["domain_set"]}
    mesh = pv.read(export_dir / "fractures.vtp")
    vtp_ids = set(map(int, mesh.cell_data["set_id"])) - {0}
    assert exported_ids == npz_ids == summary_ids == vtp_ids == set(range(1, 8))


def test_zero_and_invalid_orientation_sets_remain_explicit_in_diagnostics() -> None:
    project = _seven_set_project()
    arrays = project.m9_state.parameter_field_arrays
    arrays["set_6_p32"].fill(0.0)
    arrays["set_7_dip_direction"].fill(np.nan)
    arrays["p32_total"] = sum(arrays[f"set_{set_id}_p32"] for set_id in range(1, 8))
    realization = _generate(project)
    generated_ids = set(map(int, realization.geometry_arrays["set_id"])) - {0}
    rows = {row["set_id"]: row for row in M10Service(project).joint_set_diagnostics(realization=realization)}

    assert 6 not in generated_ids and rows[6]["unresolved_reason"] == "TARGET_P32_ZERO"
    assert 7 not in generated_ids
    assert rows[7]["unresolved_reason"] == "INSUFFICIENT_ORIENTATION_DATA"
    assert rows[7]["p32_unresolved_orientation"] > 0.0


def test_non_contiguous_and_more_than_base_palette_ids_are_not_compressed() -> None:
    ids = (1, 2, 4, 7, 9, 12, 15)
    project = _seven_set_project(ids)
    realization = _generate(project)
    generated_ids = sorted(set(map(int, realization.geometry_arrays["set_id"])) - {0})
    assert generated_ids == list(ids)
    assert len({stable_category_hex(set_id) for set_id in ids}) == len(ids)


def test_positive_group_missing_size_model_is_blocked_before_generation() -> None:
    project = _seven_set_project()
    project.m9_state.size_models = [model for model in project.m9_state.size_models if model.set_id != 7]
    errors = M10Service(project).validate_readiness(
        M10GenerationConfig(condition_calibration_observations=False)
    )
    assert any("Domain 1 / Set 7" in error for error in errors)


@pytest.mark.parametrize("worker_count", [1, 2, 4, 8])
def test_seven_set_seeded_geometry_is_worker_count_independent(worker_count: int) -> None:
    config = M10GenerationConfig(
        base_seed=42,
        worker_count=worker_count,
        condition_calibration_observations=False,
        enabled_size_classes=["SMALL", "MEDIUM", "LARGE"],
    )
    reference_project = _seven_set_project()
    candidate_project = _seven_set_project()
    reference = M10Service(reference_project)._generator(
        config.model_copy(update={"worker_count": 1}), 0, actual_worker_count=1
    ).generate(0)
    candidate = M10Service(candidate_project)._generator(
        config, 0, actual_worker_count=worker_count
    ).generate(0)
    for name, expected in reference.geometry_arrays.items():
        np.testing.assert_array_equal(candidate.geometry_arrays[name], expected)
