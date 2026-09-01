"""
ZIP-based project persistence (.dfnproj format).

Provides ZipProjectStore for saving and loading complete DFN Cave Studio
projects as ZIP archives. Each .dfnproj file contains:
  - project.json           — metadata, bounds, voxel config, seed
  - inputs/                — borehole collections, fracture observations
  - parameters/            — joint set configurations with provenance
  - results/               — DFN realizations, voxel P32, connectivity
  - metadata/              — version, timestamps

Complements the legacy .dfncs JSON format (handled by ProjectStore).
Format detection is automatic based on file extension.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone

_logger = logging.getLogger(__name__)
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from dfn_cave_studio.models.project import Project


class ZipProjectStore:
    """Save and load projects as .dfnproj ZIP archives.

    Usage:
        store = ZipProjectStore()
        store.save(project, "my_project.dfnproj")
        project = store.load("my_project.dfnproj")
    """

    # ── Save ──────────────────────────────────────────────────────────────

    @staticmethod
    def validate_project_metadata(project: "Project") -> None:
        """Validate serialized M10/M11 metadata without copying their NumPy arrays."""
        from dfn_cave_studio.models.m10 import M10State
        from dfn_cave_studio.models.m11 import M11State

        m10_state = getattr(project, "m10_state", None)
        if m10_state is not None:
            M10State.model_validate_json(m10_state.model_dump_json())
        m11_state = getattr(project, "m11_state", None)
        if m11_state is not None:
            M11State.model_validate_json(m11_state.model_dump_json())

    @staticmethod
    @contextmanager
    def _atomic_archive_path(target: Path):
        """Yield a sibling temporary archive and replace the target only on success."""
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            yield temporary
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def save(self, project: "Project", path: Path) -> None:
        """Save a Project to a .dfnproj ZIP file.

        Args:
            project: DFN Cave Studio Project instance.
            path: Target file path (should end with .dfnproj).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.validate_project_metadata(project)
        project.metadata.software_version = "0.11.0"
        project.metadata.modified_at = datetime.now(timezone.utc)
        project.schema_version = 5

        with self._atomic_archive_path(path) as archive_path, tempfile.TemporaryDirectory(
            prefix="dfn-cave-save-"
        ) as temporary_directory, zipfile.ZipFile(
            archive_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True
        ) as zf:
            # --- metadata ---
            zf.writestr("metadata/version.txt", "0.11.0")
            zf.writestr("metadata/created_at.txt", datetime.now(timezone.utc).isoformat())
            zf.writestr("metadata/format.txt", "dfnproj/1.0")

            # --- project.json (core metadata) ---
            project_dict = self._serialize_project_core(project)
            zf.writestr("project.json", json.dumps(project_dict, indent=2, default=str))

            # --- inputs ---
            if hasattr(project, "borehole_collection") and project.borehole_collection is not None:
                bh_data = self._serialize_borehole_collection(project.borehole_collection)
                zf.writestr("inputs/boreholes.json", json.dumps(bh_data, indent=2, default=str))
            borehole_database = getattr(project, "borehole_database", None)
            if borehole_database is not None:
                zf.writestr(
                    "inputs/borehole_database.json",
                    borehole_database.model_dump_json(indent=2),
                )
            spatial_grid_config = getattr(project, "spatial_grid_config", None)
            if spatial_grid_config is not None:
                zf.writestr(
                    "parameters/spatial_grid_config.json",
                    spatial_grid_config.model_dump_json(indent=2),
                )
            m9_state = getattr(project, "m9_state", None)
            if m9_state is not None:
                zf.writestr("parameters/m9_state.json", m9_state.model_dump_json(indent=2))
                if m9_state.parameter_field_arrays:
                    npz_path = Path(temporary_directory) / "voxel_parameter_field.npz"
                    np.savez_compressed(npz_path, **m9_state.parameter_field_arrays)
                    zf.write(npz_path, "results/voxel_parameter_field.npz", compress_type=zipfile.ZIP_STORED)
            m10_state = getattr(project, "m10_state", None)
            if m10_state is not None:
                zf.writestr("parameters/m10_state.json", m10_state.model_dump_json(indent=2))
                for realization in m10_state.realizations:
                    if realization.geometry_arrays:
                        npz_path = Path(temporary_directory) / f"{realization.realization_id}.npz"
                        np.savez_compressed(npz_path, **realization.geometry_arrays)
                        zf.write(
                            npz_path,
                            f"results/m10/{realization.realization_id}.npz",
                            compress_type=zipfile.ZIP_STORED,
                        )
            m11_state = getattr(project, "m11_state", None)
            if m11_state is not None:
                zf.writestr("parameters/m11_state.json", m11_state.model_dump_json(indent=2))
                for result in m11_state.results:
                    if result.arrays:
                        npz_path = Path(temporary_directory) / f"{result.realization_id}.npz"
                        np.savez_compressed(npz_path, **result.arrays)
                        zf.write(
                            npz_path,
                            f"results/m11/{result.realization_id}.npz",
                            compress_type=zipfile.ZIP_STORED,
                        )

            # --- parameters ---
            joint_sets = getattr(project, "joint_sets", [])
            if joint_sets:
                js_data = [self._serialize_joint_set(js) for js in joint_sets]
                zf.writestr("parameters/joint_sets.json", json.dumps(js_data, indent=2, default=str))

            voxel_config = getattr(project, "voxel_config", None)
            if voxel_config is not None:
                zf.writestr(
                    "parameters/voxel_config.json", json.dumps(voxel_config.model_dump(), indent=2, default=str)
                )

            # --- results ---
            # DFN realization (fracture geometries)
            realizations = getattr(project, "dfn_realizations", [])
            if realizations:
                dfn_data = self._serialize_realizations(realizations)
                zf.writestr("results/dfn_realizations.json", json.dumps(dfn_data, indent=2, default=str))

            # Voxel P32
            voxel_p32 = getattr(project, "voxel_p32_results", None)
            if voxel_p32 is not None:
                zf.writestr("results/voxel_p32.json", json.dumps(voxel_p32, indent=2, default=str))

            # Connectivity
            conn = getattr(project, "connectivity_results", None)
            clusters = getattr(project, "connectivity_clusters", None)
            if conn is not None:
                conn_out = self._serialize_connectivity(conn)
                # Embed cluster labels if available
                if clusters is not None:
                    if isinstance(clusters, (list, tuple)):
                        conn_out["component_labels"] = [int(c) for c in clusters]
                    elif isinstance(clusters, np.ndarray):
                        conn_out["component_labels"] = clusters.tolist()
                zf.writestr("results/connectivity.json", json.dumps(conn_out, indent=2, default=str))

            # ── M7 data ──────────────────────────────────────────────────
            m7_data = getattr(project, "_m7_data", None)
            if m7_data is None:
                m7_data = {}
            # Raw DataFrames — save as CSV inside ZIP
            for key in [
                "raw_surveys",
                "raw_fractures",
                "raw_rqd",
                "raw_domain_intervals",
                "cleaned_rqd",
            ]:
                df = m7_data.get(key)
                if df is not None and hasattr(df, "to_csv"):
                    zf.writestr(f"m7/{key}.csv", df.to_csv(index=False))
            # Workflow state
            wf = m7_data.get("workflow")
            if wf is not None:
                zf.writestr("m7/workflow.json", json.dumps(wf.to_dict(), indent=2, default=str))
            # Holdout
            ho = m7_data.get("holdout")
            if ho is not None:
                zf.writestr("m7/holdout.json", json.dumps(ho.to_dict(), indent=2, default=str))
            # Domain intervals
            di = m7_data.get("domain_intervals", [])
            if di:
                zf.writestr(
                    "m7/domain_intervals.json",
                    json.dumps([d.model_dump() if hasattr(d, "model_dump") else d for d in di], indent=2, default=str),
                )
            # Structural domains (full definitions with name, color, notes, etc.)
            sd = getattr(project, "structural_domains", None)
            if sd is not None and hasattr(sd, "domains") and sd.domains:
                zf.writestr(
                    "m7/structural_domains.json",
                    json.dumps([d.model_dump() for d in sd.domains], indent=2, default=str),
                )
            # Field mappings
            fm = m7_data.get("field_mappings", {})
            if fm:
                zf.writestr(
                    "m7/field_mappings.json",
                    json.dumps(
                        {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in fm.items()},
                        indent=2,
                        default=str,
                    ),
                )
            # Quality issues
            qi = m7_data.get("quality_issues", [])
            if qi:
                zf.writestr(
                    "m7/quality_issues.json",
                    json.dumps([q.model_dump() if hasattr(q, "model_dump") else q for q in qi], indent=2, default=str),
                )
            excluded = m7_data.get("excluded_records", [])
            if excluded:
                zf.writestr(
                    "m7/excluded_records.json",
                    json.dumps(excluded, indent=2, default=str),
                )

            # Summary
            summary = self._build_summary(project)
            zf.writestr("results/summary.json", json.dumps(summary, indent=2, default=str))

    # ── Load ──────────────────────────────────────────────────────────────

    def load(self, path: Path) -> "Project":
        """Load a Project from a .dfnproj ZIP file.

        Args:
            path: Path to .dfnproj file.

        Returns:
            Reconstructed Project instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a valid .dfnproj archive.
        """
        from dfn_cave_studio.models.project import Project

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {path}")

        project = Project()

        with tempfile.TemporaryDirectory(prefix="dfn-cave-load-") as temporary_directory, zipfile.ZipFile(path, "r") as zf:
            # --- project.json ---
            if "project.json" in zf.namelist():
                project_dict = json.loads(zf.read("project.json").decode("utf-8"))
                self._deserialize_project_core(project, project_dict)

            # --- inputs ---
            if "inputs/boreholes.json" in zf.namelist():
                bh_data = json.loads(zf.read("inputs/boreholes.json").decode("utf-8"))
                project.borehole_collection = self._deserialize_borehole_collection(bh_data)
            if "inputs/borehole_database.json" in zf.namelist():
                from dfn_cave_studio.models.borehole_database import BoreholeDatabase

                project.borehole_database = BoreholeDatabase.model_validate_json(
                    zf.read("inputs/borehole_database.json").decode("utf-8")
                )

            # --- parameters ---
            if "parameters/joint_sets.json" in zf.namelist():
                js_data = json.loads(zf.read("parameters/joint_sets.json").decode("utf-8"))
                project.joint_sets = self._deserialize_joint_sets(js_data)

            if "parameters/voxel_config.json" in zf.namelist():
                from dfn_cave_studio.models.bounds import VoxelConfig

                vc_dict = json.loads(zf.read("parameters/voxel_config.json").decode("utf-8"))
                project.voxel_config = VoxelConfig(**vc_dict)
            if "parameters/spatial_grid_config.json" in zf.namelist():
                from dfn_cave_studio.models.spatial_grid import SpatialGridConfig

                project.spatial_grid_config = SpatialGridConfig.model_validate_json(
                    zf.read("parameters/spatial_grid_config.json").decode("utf-8")
                )
            if "parameters/m9_state.json" in zf.namelist():
                from dfn_cave_studio.models.m9 import M9State

                project.m9_state = M9State.model_validate_json(zf.read("parameters/m9_state.json").decode("utf-8"))
                if "results/voxel_parameter_field.npz" in zf.namelist():
                    extracted = Path(zf.extract("results/voxel_parameter_field.npz", temporary_directory))
                    with np.load(extracted, allow_pickle=False) as archive:
                        project.m9_state.parameter_field_arrays = {name: archive[name].copy() for name in archive.files}
            if "parameters/m10_state.json" in zf.namelist():
                from dfn_cave_studio.models.m10 import M10State

                project.m10_state = M10State.model_validate_json(
                    zf.read("parameters/m10_state.json").decode("utf-8")
                )
                for realization in project.m10_state.realizations:
                    array_path = f"results/m10/{realization.realization_id}.npz"
                    if array_path not in zf.namelist():
                        raise ValueError(f"M10 realization geometry is missing: {array_path}")
                    extracted = Path(zf.extract(array_path, temporary_directory))
                    with np.load(extracted, allow_pickle=False) as archive:
                        realization.geometry_arrays = {name: archive[name].copy() for name in archive.files}
                    from dfn_cave_studio.dfn.m10_geometry import migrate_legacy_geometry

                    migrate_legacy_geometry(realization)
            if "parameters/m11_state.json" in zf.namelist():
                from dfn_cave_studio.models.m11 import M11State

                project.m11_state = M11State.model_validate_json(
                    zf.read("parameters/m11_state.json").decode("utf-8")
                )
                for result in project.m11_state.results:
                    array_path = f"results/m11/{result.realization_id}.npz"
                    if array_path not in zf.namelist():
                        raise ValueError(f"M11 second-voxelization arrays are missing: {array_path}")
                    extracted = Path(zf.extract(array_path, temporary_directory))
                    with np.load(extracted, allow_pickle=False) as archive:
                        result.arrays = {name: archive[name].copy() for name in archive.files}

            # --- results ---
            if "results/dfn_realizations.json" in zf.namelist():
                dfn_data = json.loads(zf.read("results/dfn_realizations.json").decode("utf-8"))
                project.dfn_realizations = self._deserialize_realizations(dfn_data)

            # Voxel P32 and connectivity are stored as plain dicts
            if "results/voxel_p32.json" in zf.namelist():
                project.voxel_p32_results = json.loads(zf.read("results/voxel_p32.json").decode("utf-8"))

            if "results/connectivity.json" in zf.namelist():
                conn_data = json.loads(zf.read("results/connectivity.json").decode("utf-8"))
                project.connectivity_results = conn_data
                # Restore connectivity_clusters from percolation/cluster data
                if "component_labels" in conn_data:
                    project.connectivity_clusters = conn_data["component_labels"]

            # ── M7 data restore ────────────────────────────────────────
            m7_data = {}
            # Raw DataFrames from CSV
            for key in [
                "raw_surveys",
                "raw_fractures",
                "raw_rqd",
                "raw_domain_intervals",
                "cleaned_rqd",
            ]:
                csv_path = f"m7/{key}.csv"
                if csv_path in zf.namelist():
                    import io as _io

                    try:
                        m7_data[key] = pd.read_csv(_io.BytesIO(zf.read(csv_path)))
                    except (
                        OSError,
                        UnicodeDecodeError,
                        ValueError,
                        pd.errors.ParserError,
                    ):
                        _logger.warning(
                            "Failed to restore M7 DataFrame '%s' from %s in %s", key, csv_path, path, exc_info=True
                        )
            if "m7/workflow.json" in zf.namelist():
                from dfn_cave_studio.services.workflow_controller import WorkflowController

                wf_data = json.loads(zf.read("m7/workflow.json").decode("utf-8"))
                wf = WorkflowController()
                wf.from_dict(wf_data)
                m7_data["workflow"] = wf
            if "m7/holdout.json" in zf.namelist():
                from dfn_cave_studio.services.holdout_service import HoldoutService

                ho_data = json.loads(zf.read("m7/holdout.json").decode("utf-8"))
                m7_data["holdout"] = HoldoutService.from_dict(ho_data)
            if "m7/domain_intervals.json" in zf.namelist():
                from dfn_cave_studio.models.data_management import DomainInterval

                di_data = json.loads(zf.read("m7/domain_intervals.json").decode("utf-8"))
                m7_data["domain_intervals"] = [DomainInterval(**d) for d in di_data]
            if "m7/structural_domains.json" in zf.namelist():
                from dfn_cave_studio.models.structural_domain import (
                    StructuralDomain,
                )

                sd_data = json.loads(zf.read("m7/structural_domains.json").decode("utf-8"))
                restored_domains = [StructuralDomain(**d) for d in sd_data]
                # Replace project's structural domains (preserves Global if not in file)
                project.structural_domains.domains = restored_domains
                # Ensure Global (id=0) always exists
                if not any(d.domain_id == 0 for d in restored_domains):
                    project.structural_domains.domains.insert(0, StructuralDomain(domain_id=0, name="Global Domain"))
            if "m7/field_mappings.json" in zf.namelist():
                from dfn_cave_studio.models.data_management import FieldMapping

                fm_data = json.loads(zf.read("m7/field_mappings.json").decode("utf-8"))
                m7_data["field_mappings"] = {k: FieldMapping(**v) for k, v in fm_data.items()}
            if "m7/quality_issues.json" in zf.namelist():
                from dfn_cave_studio.models.data_management import DataQualityIssue

                qi_data = json.loads(zf.read("m7/quality_issues.json").decode("utf-8"))
                m7_data["quality_issues"] = [DataQualityIssue(**q) for q in qi_data]
            if "m7/excluded_records.json" in zf.namelist():
                m7_data["excluded_records"] = json.loads(zf.read("m7/excluded_records.json").decode("utf-8"))
            if m7_data:
                project._m7_data = m7_data

        from dfn_cave_studio.services.borehole_repository import BoreholeRepository

        repository = BoreholeRepository(project)
        if not repository.database.records:
            repository.migrate_m7()
        else:
            repository.migrate_orientation_completeness()
            repository.rebuild_formal_collection()
        return project

    # ── Serialization Helpers ─────────────────────────────────────────────

    def _serialize_project_core(self, project) -> dict:
        """Serialize core project metadata."""

        d = {
            "name": project.metadata.name if hasattr(project, "metadata") else "",
            "schema_version": getattr(project, "schema_version", 1),
        }
        if hasattr(project, "metadata"):
            d["metadata"] = project.metadata.model_dump(mode="json")
        if hasattr(project, "model_bounds") and project.model_bounds is not None:
            b = project.model_bounds
            d["model_bounds"] = {
                "x_min": b.x_min,
                "x_max": b.x_max,
                "y_min": b.y_min,
                "y_max": b.y_max,
                "z_min": b.z_min,
                "z_max": b.z_max,
            }
        if hasattr(project, "master_seed"):
            d["master_seed"] = project.master_seed
        elif hasattr(project, "config") and hasattr(project.config, "master_seed"):
            d["master_seed"] = project.config.master_seed
        return d

    def _deserialize_project_core(self, project, d: dict) -> None:
        """Deserialize core project metadata into a Project instance."""
        from dfn_cave_studio.models.bounds import ModelBounds

        if "name" in d and hasattr(project, "metadata"):
            project.metadata.name = d["name"]
        if "metadata" in d and hasattr(project, "metadata"):
            from dfn_cave_studio.models.project import ProjectMetadata

            project.metadata = ProjectMetadata(**d["metadata"])
        if "master_seed" in d:
            if hasattr(project, "config"):
                project.config.master_seed = d["master_seed"]
        if "model_bounds" in d:
            b = d["model_bounds"]
            project.model_bounds = ModelBounds(
                x_min=b["x_min"],
                x_max=b["x_max"],
                y_min=b["y_min"],
                y_max=b["y_max"],
                z_min=b["z_min"],
                z_max=b["z_max"],
            )

    def _serialize_borehole_collection(self, collection) -> dict:
        """Serialize a BoreholeCollection to JSON-safe dict."""
        boreholes = []
        for bh in collection:
            bh_dict = {
                "borehole_id": bh.borehole_id,
                "name": bh.name,
                "collar": {
                    "x": bh.collar.collar_x,
                    "y": bh.collar.collar_y,
                    "z": bh.collar.collar_z,
                    "azimuth": bh.collar.azimuth,
                    "dip": bh.collar.dip,
                    "final_depth": bh.collar.final_depth,
                },
                "fracture_observations": [],
            }
            # Survey stations
            stations = []
            for s in bh.survey.stations:
                stations.append(
                    {
                        "measured_depth": s.measured_depth,
                        "azimuth": s.azimuth,
                        "dip": s.dip,
                    }
                )
            bh_dict["survey_stations"] = stations

            # Fracture observations with 3D positions
            points, _ = bh.compute_trajectory()
            for obs in bh.fracture_observations:
                pos = bh.locate_observation(obs)
                obs_dict = {
                    "measured_depth": obs.measured_depth,
                    "dip_direction": obs.dip_direction,
                    "dip": obs.dip,
                    "orientation_completeness": obs.orientation_completeness.value,
                    "aperture": obs.aperture,
                    "fracture_type": (
                        str(obs.fracture_type.value) if hasattr(obs.fracture_type, "value") else str(obs.fracture_type)
                    ),
                    "confidence": obs.confidence,
                    "set_id": obs.set_id,
                    "position_3d": [float(pos[0]), float(pos[1]), float(pos[2])] if pos is not None else None,
                }
                bh_dict["fracture_observations"].append(obs_dict)
            boreholes.append(bh_dict)
        return {"boreholes": boreholes, "name": collection.name}

    def _deserialize_borehole_collection(self, data: dict):
        """Deserialize a BoreholeCollection from dict."""
        from dfn_cave_studio.models.borehole import (
            Borehole,
            BoreholeCollection,
            BoreholeSurvey,
            Collar,
            FractureObservation,
            SurveyStation,
        )
        from dfn_cave_studio.models.enums import FractureType

        collection = BoreholeCollection(name=data.get("name", ""))
        for bh_dict in data.get("boreholes", []):
            c = bh_dict["collar"]
            collar = Collar(
                borehole_id=bh_dict["borehole_id"],
                collar_x=c["x"],
                collar_y=c["y"],
                collar_z=c["z"],
                azimuth=c.get("azimuth", 0),
                dip=c.get("dip", -90),
                final_depth=c.get("final_depth", 100),
            )
            survey = BoreholeSurvey(stations=[SurveyStation(**s) for s in bh_dict.get("survey_stations", [])])
            bh = Borehole(
                borehole_id=bh_dict["borehole_id"],
                name=bh_dict.get("name", ""),
                collar=collar,
                survey=survey,
            )
            for obs_dict in bh_dict.get("fracture_observations", []):
                ft_str = obs_dict.get("fracture_type", "joint")
                try:
                    ft = FractureType(ft_str)
                except (ValueError, TypeError):
                    ft = FractureType.JOINT
                obs = FractureObservation(
                    measured_depth=obs_dict["measured_depth"],
                    dip_direction=obs_dict.get("dip_direction", 0),
                    dip=obs_dict.get("dip", 0),
                    orientation_completeness=obs_dict.get("orientation_completeness", "full_orientation"),
                    aperture=obs_dict.get("aperture"),
                    fracture_type=ft,
                    confidence=obs_dict.get("confidence", 1.0),
                    set_id=obs_dict.get("set_id"),
                )
                bh.fracture_observations.append(obs)
            collection.add(bh)
        return collection

    def _serialize_joint_set(self, js) -> dict:
        """Serialize a JointSetConfig to dict."""
        return {
            "set_id": js.set_id,
            "name": js.name,
            "color": js.color,
            "orientation": {
                "mean_dip_direction": js.orientation.mean_dip_direction,
                "mean_dip": js.orientation.mean_dip,
                "kappa": js.orientation.kappa,
            },
            "size": {
                "distribution_type": (
                    str(js.size.distribution_type.value)
                    if hasattr(js.size.distribution_type, "value")
                    else str(js.size.distribution_type)
                ),
                "min_radius": js.size.min_radius,
                "max_radius": js.size.max_radius,
                "lognormal_mu": js.size.lognormal_mu,
                "lognormal_sigma": js.size.lognormal_sigma,
                "power_law_exponent": js.size.power_law_exponent,
            },
            "target_p32": js.target_p32,
            "p32_tolerance": js.p32_tolerance,
            "provenance": dict(js.provenance) if hasattr(js, "provenance") else {},
        }

    def _deserialize_joint_sets(self, data: list) -> list:
        """Deserialize a list of JointSetConfigs from dicts."""
        from dfn_cave_studio.models.enums import SizeDistributionType
        from dfn_cave_studio.models.fracture_set import (
            JointSetConfig,
            OrientationDistribution,
            SizeDistribution,
        )

        result = []
        for d in data:
            sd = d.get("size", {})
            dist_type_str = sd.get("distribution_type", "lognormal")
            try:
                dist_type = SizeDistributionType(dist_type_str)
            except (ValueError, TypeError):
                dist_type = SizeDistributionType.LOGNORMAL

            js = JointSetConfig(
                set_id=d.get("set_id", 0),
                name=d.get("name", ""),
                color=d.get("color", "#1976d2"),
                orientation=OrientationDistribution(**d.get("orientation", {})),
                size=SizeDistribution(
                    distribution_type=dist_type,
                    min_radius=sd.get("min_radius", 0.5),
                    max_radius=sd.get("max_radius", 10.0),
                    lognormal_mu=sd.get("lognormal_mu", 1.0),
                    lognormal_sigma=sd.get("lognormal_sigma", 0.5),
                    power_law_exponent=sd.get("power_law_exponent", 3.0),
                ),
                target_p32=d.get("target_p32", 1.0),
                p32_tolerance=d.get("p32_tolerance", 0.05),
            )
            if "provenance" in d:
                js.provenance = d["provenance"]
            result.append(js)
        return result

    def _serialize_realizations(self, realizations: list) -> list:
        """Serialize a list of DFNRealizations to JSON-safe dicts."""
        result = []
        for r in realizations:
            fractures = []
            for f in getattr(r, "stochastic_fractures", []):
                g = f.geometry
                fractures.append(
                    {
                        "fracture_id": str(f.fracture_id) if hasattr(f, "fracture_id") else "",
                        "set_id": f.set_id,
                        "set_name": getattr(f, "set_name", ""),
                        "center": [float(g.center_x), float(g.center_y), float(g.center_z)],
                        "normal": [float(g.normal_x), float(g.normal_y), float(g.normal_z)],
                        "radius": float(f.radius if f.radius > 0 else (g.radius or 1.0)),
                        "dip_direction": float(g.dip_direction) if g.dip_direction is not None else 0.0,
                        "dip": float(g.dip) if g.dip is not None else 0.0,
                    }
                )
            gr = getattr(r, "generation_result", None)
            provenance = {}
            if gr is not None:
                provenance = dict(getattr(gr, "parameter_provenance", {}))
            result.append(
                {
                    "realization_number": getattr(r, "realization_number", 0),
                    "name": getattr(r, "name", ""),
                    "total_p32": getattr(r, "total_p32", 0.0),
                    "fractures": fractures,
                    "parameter_provenance": provenance,
                }
            )
        return result

    def _deserialize_realizations(self, data: list) -> list:
        """Deserialize DFN realizations from dicts."""
        from dfn_cave_studio.models.dfn_realization import (
            DFNGenerationResult,
            DFNRealization,
        )
        from dfn_cave_studio.models.enums import FractureSource
        from dfn_cave_studio.models.fracture import (
            FractureGeometry,
            StochasticFracture,
        )

        result = []
        for rd in data:
            fractures = []
            for fd in rd.get("fractures", []):
                c = fd["center"]
                n = fd["normal"]
                geo = FractureGeometry(
                    geometry_type="disk",
                    center_x=c[0],
                    center_y=c[1],
                    center_z=c[2],
                    normal_x=n[0],
                    normal_y=n[1],
                    normal_z=n[2],
                    radius=fd["radius"],
                    dip_direction=fd.get("dip_direction", 0.0),
                    dip=fd.get("dip", 0.0),
                    area=3.141592653589793 * fd["radius"] ** 2,
                )
                f = StochasticFracture(
                    geometry=geo,
                    set_id=fd.get("set_id", 0),
                    set_name=fd.get("set_name", ""),
                    radius=fd["radius"],
                    source=FractureSource.STOCHASTIC,
                )
                fractures.append(f)

            gr = DFNGenerationResult(stochastic_fractures=fractures)
            if "parameter_provenance" in rd:
                gr.parameter_provenance = rd["parameter_provenance"]

            realization = DFNRealization(
                realization_number=rd.get("realization_number", 0),
                name=rd.get("name", ""),
                stochastic_fractures=fractures,
                generation_result=gr,
                total_p32=rd.get("total_p32", 0.0),
            )
            result.append(realization)
        return result

    def _serialize_connectivity(self, conn: Any) -> dict:
        """Serialize connectivity results to JSON-safe dict.

        Saves: component labels per fracture, edge list, percolation flags,
        statistics, and graph metadata.  Uses fracture indices (int) as
        node IDs for compact storage; fracture_id strings are kept in the
        DFN realization file.
        """
        if isinstance(conn, dict):
            result = {}
            for k, v in conn.items():
                if isinstance(v, (int, float, str, bool, list, type(None))):
                    result[k] = v
                elif isinstance(v, dict):
                    result[k] = {
                        str(k2): v2
                        for k2, v2 in v.items()
                        if isinstance(v2, (int, float, str, bool, list, dict, type(None)))
                    }
                elif isinstance(v, (np.integer,)):
                    result[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    result[k] = float(v)
                elif isinstance(v, np.ndarray):
                    result[k] = v.tolist()
                else:
                    result[k] = str(v)
            return result
        if isinstance(conn, (list, tuple)):
            return [
                (
                    self._serialize_connectivity(c)
                    if isinstance(c, dict)
                    else (
                        int(c)
                        if isinstance(c, (np.integer,))
                        else (
                            float(c)
                            if isinstance(c, (np.floating,))
                            else c.tolist() if isinstance(c, np.ndarray) else c
                        )
                    )
                )
                for c in conn
            ]
        return {"raw": str(conn)}

    @staticmethod
    def extract_voxel_p32_results(grid, connectivity_labels=None) -> list:
        """Extract real per-voxel P32 results from a VoxelGrid.

        Returns a list of dicts suitable for JSON serialization:
          [{i, j, k, x, y, z, local_p32, fracture_area, fracture_count,
            connectivity_cluster, cell_size}, ...]

        Args:
            grid: VoxelGrid with computed intersection attributes.
            connectivity_labels: Optional list/array of cluster labels per voxel.

        Returns:
            List of dicts, one per active voxel with fracture data.
        """
        results = []
        active_voxels = list(grid.iter_active_voxels())
        for ix, iy, iz in active_voxels:
            # Read per-voxel attributes using VoxelGrid.get_voxel
            fc_val = grid.get_voxel(ix, iy, iz, "fracture_count")
            fa_val = grid.get_voxel(ix, iy, iz, "fracture_area")  # stored as mm²*1e6
            lp_val = grid.get_voxel(ix, iy, iz, "local_p32")  # stored as milli-P32

            fracture_count = int(fc_val) if fc_val is not None else 0
            fracture_area_m2 = float(fa_val) / 1e6 if fa_val is not None else 0.0
            local_p32 = float(lp_val) / 1000.0 if lp_val is not None else 0.0

            if fracture_count == 0 and local_p32 == 0.0:
                continue  # Skip empty voxels

            entry = {
                "i": int(ix),
                "j": int(iy),
                "k": int(iz),
                "x": float(grid.x_min + ix * grid.cell_size_x + grid.cell_size_x / 2),
                "y": float(grid.y_min + iy * grid.cell_size_y + grid.cell_size_y / 2),
                "z": float(grid.z_min + iz * grid.cell_size_z + grid.cell_size_z / 2),
                "local_p32": local_p32,
                "fracture_area": fracture_area_m2,
                "fracture_count": fracture_count,
                "connectivity_cluster": -1,
                "dx": float(grid.cell_size_x),
                "dy": float(grid.cell_size_y),
                "dz": float(grid.cell_size_z),
            }
            results.append(entry)

        # Add grid-level metadata as first entry
        csx = float(getattr(grid, "cell_size_x", 1.0))
        csy = float(getattr(grid, "cell_size_y", 1.0))
        csz = float(getattr(grid, "cell_size_z", 1.0))
        gx_min = float(getattr(grid, "x_min", 0.0))
        gy_min = float(getattr(grid, "y_min", 0.0))
        gz_min = float(getattr(grid, "z_min", 0.0))
        nx = int(getattr(grid, "nx", 0))
        ny = int(getattr(grid, "ny", 0))
        nz = int(getattr(grid, "nz", 0))
        gx_max = gx_min + nx * csx
        gy_max = gy_min + ny * csy
        gz_max = gz_min + nz * csz
        results.insert(
            0,
            {
                "i": -1,
                "j": -1,
                "k": -1,
                "x": -1,
                "y": -1,
                "z": -1,
                "local_p32": -1.0,
                "fracture_area": -1.0,
                "fracture_count": -1,
                "connectivity_cluster": -1,
                "dx": csx,
                "dy": csy,
                "dz": csz,
                "x_min": gx_min,
                "x_max": gx_max,
                "y_min": gy_min,
                "y_max": gy_max,
                "z_min": gz_min,
                "z_max": gz_max,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "_meta": "grid_metadata",
            },
        )
        return results

    def _build_summary(self, project) -> dict:
        """Build a summary dict from the project state."""
        summary = {
            "name": project.metadata.name if hasattr(project, "metadata") else "",
            "version": "0.11.0",
            "borehole_count": 0,
            "observation_count": 0,
            "joint_set_count": 0,
            "fracture_count": 0,
            "m10_realization_count": 0,
            "m10_fracture_count": 0,
            "has_voxel_p32": False,
            "has_connectivity": False,
        }
        if hasattr(project, "borehole_collection") and project.borehole_collection is not None:
            summary["borehole_count"] = len(project.borehole_collection)
            for bh in project.borehole_collection:
                summary["observation_count"] += len(bh.fracture_observations)
        summary["joint_set_count"] = len(getattr(project, "joint_sets", []))
        for r in getattr(project, "dfn_realizations", []):
            summary["fracture_count"] += len(getattr(r, "stochastic_fractures", []))
        m10_state = getattr(project, "m10_state", None)
        if m10_state is not None:
            summary["m10_realization_count"] = len(m10_state.realizations)
            summary["m10_fracture_count"] = sum(item.fracture_count for item in m10_state.realizations)
        summary["has_voxel_p32"] = getattr(project, "voxel_p32_results", None) is not None
        summary["has_connectivity"] = getattr(project, "connectivity_results", None) is not None
        return summary
