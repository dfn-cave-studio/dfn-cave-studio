"""M8 migration and M9 compressed-array persistence."""

import numpy as np

from dfn_cave_studio.dfn.size_models import MLESizeModelFitter
from dfn_cave_studio.models.m9 import DensitySettings, ParameterFieldMetadata, P10Interval
from dfn_cave_studio.models.project import Project
from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore


def test_m9_state_and_arrays_round_trip(tmp_path):
    project = Project()
    project.m9_state.density_settings = DensitySettings(interval_length=5)
    project.m9_state.p10_intervals = [
        P10Interval(hole_id="BH-1", from_depth=0, to_depth=5, observation_count=0, sample_length=5, p10=0, data_state="true_zero")
    ]
    project.m9_state.parameter_field_metadata = ParameterFieldMetadata(
        shape=(2, 1, 1), origin=(0, 0, 0), spacing=(1, 1, 1), field_names=["p32_total"], set_ids=[], density_method="global_constant", random_seed=42, estimated_bytes=8
    )
    project.m9_state.parameter_field_arrays = {"p32_total": np.array([[[0]], [[np.nan]]], dtype=np.float32)}
    path = tmp_path / "m9.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)
    assert restored.metadata.software_version == "0.9.0"
    assert restored.m9_state.p10_intervals == project.m9_state.p10_intervals
    np.testing.assert_equal(restored.m9_state.parameter_field_arrays["p32_total"], project.m9_state.parameter_field_arrays["p32_total"])


def test_m8_project_model_migrates_to_empty_m9_state():
    data = Project().model_dump(mode="json", exclude={"m9_state"})
    data["schema_version"] = 2
    restored = Project.from_dict(data)
    assert restored.schema_version == 3
    assert restored.m9_state.p10_intervals == []


def test_experimental_size_fit_status_round_trips(tmp_path):
    project = Project()
    project.m9_state.size_models = [
        MLESizeModelFitter().fit(
            [1.0, 1.2, 1.6, 2.0, 2.8],
            domain_id=1,
            set_id=1,
            source_field="radius",
        )
    ]
    path = tmp_path / "experimental-size-fit.dfnproj"
    ZipProjectStore().save(project, path)
    restored = ZipProjectStore().load(path)

    assert restored.m9_state.size_models == project.m9_state.size_models
    model = restored.m9_state.size_models[0]
    assert model.source.value == "experimental"
    assert model.fit_status == "experimental"
    assert all(candidate.sample_count == 5 for candidate in model.candidates)
    assert all(candidate.optimizer_message for candidate in model.candidates)
