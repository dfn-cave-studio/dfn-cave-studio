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
import zipfile
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from numpy.typing import NDArray


class ZipProjectStore:
    """Save and load projects as .dfnproj ZIP archives.

    Usage:
        store = ZipProjectStore()
        store.save(project, "my_project.dfnproj")
        project = store.load("my_project.dfnproj")
    """

    # ── Save ──────────────────────────────────────────────────────────────

    def save(self, project: "Project", path: Path) -> None:
        """Save a Project to a .dfnproj ZIP file.

        Args:
            project: DFN Cave Studio Project instance.
            path: Target file path (should end with .dfnproj).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # --- metadata ---
            zf.writestr("metadata/version.txt", "0.6.3")
            zf.writestr("metadata/created_at.txt",
                        datetime.now(timezone.utc).isoformat())
            zf.writestr("metadata/format.txt", "dfnproj/1.0")

            # --- project.json (core metadata) ---
            project_dict = self._serialize_project_core(project)
            zf.writestr("project.json", json.dumps(project_dict, indent=2, default=str))

            # --- inputs ---
            if hasattr(project, "borehole_collection") and project.borehole_collection is not None:
                bh_data = self._serialize_borehole_collection(project.borehole_collection)
                zf.writestr("inputs/boreholes.json",
                            json.dumps(bh_data, indent=2, default=str))

            # --- parameters ---
            joint_sets = getattr(project, "joint_sets", [])
            if joint_sets:
                js_data = [self._serialize_joint_set(js) for js in joint_sets]
                zf.writestr("parameters/joint_sets.json",
                            json.dumps(js_data, indent=2, default=str))

            voxel_config = getattr(project, "voxel_config", None)
            if voxel_config is not None:
                zf.writestr("parameters/voxel_config.json",
                            json.dumps(voxel_config.model_dump(), indent=2, default=str))

            # --- results ---
            # DFN realization (fracture geometries)
            realizations = getattr(project, "dfn_realizations", [])
            if realizations:
                dfn_data = self._serialize_realizations(realizations)
                zf.writestr("results/dfn_realizations.json",
                            json.dumps(dfn_data, indent=2, default=str))

            # Voxel P32
            voxel_p32 = getattr(project, "voxel_p32_results", None)
            if voxel_p32 is not None:
                zf.writestr("results/voxel_p32.json",
                            json.dumps(voxel_p32, indent=2, default=str))

            # Connectivity
            conn = getattr(project, "connectivity_results", None)
            if conn is not None:
                zf.writestr("results/connectivity.json",
                            json.dumps(self._serialize_connectivity(conn), indent=2, default=str))

            # Summary
            summary = self._build_summary(project)
            zf.writestr("results/summary.json",
                        json.dumps(summary, indent=2, default=str))

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

        with zipfile.ZipFile(path, "r") as zf:
            # --- project.json ---
            if "project.json" in zf.namelist():
                project_dict = json.loads(zf.read("project.json").decode("utf-8"))
                self._deserialize_project_core(project, project_dict)

            # --- inputs ---
            if "inputs/boreholes.json" in zf.namelist():
                bh_data = json.loads(zf.read("inputs/boreholes.json").decode("utf-8"))
                project.borehole_collection = self._deserialize_borehole_collection(bh_data)

            # --- parameters ---
            if "parameters/joint_sets.json" in zf.namelist():
                js_data = json.loads(zf.read("parameters/joint_sets.json").decode("utf-8"))
                project.joint_sets = self._deserialize_joint_sets(js_data)

            if "parameters/voxel_config.json" in zf.namelist():
                from dfn_cave_studio.models.bounds import VoxelConfig
                vc_dict = json.loads(zf.read("parameters/voxel_config.json").decode("utf-8"))
                project.voxel_config = VoxelConfig(**vc_dict)

            # --- results ---
            if "results/dfn_realizations.json" in zf.namelist():
                dfn_data = json.loads(zf.read("results/dfn_realizations.json").decode("utf-8"))
                project.dfn_realizations = self._deserialize_realizations(dfn_data)

            # Voxel P32 and connectivity are stored as plain dicts
            if "results/voxel_p32.json" in zf.namelist():
                project.voxel_p32_results = json.loads(
                    zf.read("results/voxel_p32.json").decode("utf-8"))

            if "results/connectivity.json" in zf.namelist():
                conn_data = json.loads(zf.read("results/connectivity.json").decode("utf-8"))
                project.connectivity_results = conn_data

        return project

    # ── Serialization Helpers ─────────────────────────────────────────────

    def _serialize_project_core(self, project) -> dict:
        """Serialize core project metadata."""
        from dfn_cave_studio.models.bounds import ModelBounds
        d = {
            "name": project.metadata.name if hasattr(project, "metadata") else "",
            "schema_version": getattr(project, "schema_version", 1),
        }
        if hasattr(project, "model_bounds") and project.model_bounds is not None:
            b = project.model_bounds
            d["model_bounds"] = {
                "x_min": b.x_min, "x_max": b.x_max,
                "y_min": b.y_min, "y_max": b.y_max,
                "z_min": b.z_min, "z_max": b.z_max,
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
        if "master_seed" in d:
            if hasattr(project, "config"):
                project.config.master_seed = d["master_seed"]
        if "model_bounds" in d:
            b = d["model_bounds"]
            project.model_bounds = ModelBounds(
                x_min=b["x_min"], x_max=b["x_max"],
                y_min=b["y_min"], y_max=b["y_max"],
                z_min=b["z_min"], z_max=b["z_max"],
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
                stations.append({
                    "measured_depth": s.measured_depth,
                    "azimuth": s.azimuth,
                    "dip": s.dip,
                })
            bh_dict["survey_stations"] = stations

            # Fracture observations with 3D positions
            points, _ = bh.compute_trajectory()
            for obs in bh.fracture_observations:
                pos = bh.locate_observation(obs)
                obs_dict = {
                    "measured_depth": obs.measured_depth,
                    "dip_direction": obs.dip_direction,
                    "dip": obs.dip,
                    "aperture": obs.aperture,
                    "fracture_type": str(obs.fracture_type.value) if hasattr(obs.fracture_type, "value") else str(obs.fracture_type),
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
            BoreholeCollection, Borehole, Collar, BoreholeSurvey,
            SurveyStation, FractureObservation,
        )
        from dfn_cave_studio.models.enums import FractureType

        collection = BoreholeCollection(name=data.get("name", ""))
        for bh_dict in data.get("boreholes", []):
            c = bh_dict["collar"]
            collar = Collar(
                borehole_id=bh_dict["borehole_id"],
                collar_x=c["x"], collar_y=c["y"], collar_z=c["z"],
                azimuth=c.get("azimuth", 0), dip=c.get("dip", -90),
                final_depth=c.get("final_depth", 100),
            )
            survey = BoreholeSurvey(stations=[
                SurveyStation(**s) for s in bh_dict.get("survey_stations", [])
            ])
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
                "distribution_type": str(js.size.distribution_type.value) if hasattr(js.size.distribution_type, "value") else str(js.size.distribution_type),
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
        from dfn_cave_studio.models.fracture_set import (
            JointSetConfig, OrientationDistribution, SizeDistribution,
        )
        from dfn_cave_studio.models.enums import SizeDistributionType

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
                fractures.append({
                    "fracture_id": str(f.fracture_id) if hasattr(f, "fracture_id") else "",
                    "set_id": f.set_id,
                    "set_name": getattr(f, "set_name", ""),
                    "center": [float(g.center_x), float(g.center_y), float(g.center_z)],
                    "normal": [float(g.normal_x), float(g.normal_y), float(g.normal_z)],
                    "radius": float(f.radius if f.radius > 0 else (g.radius or 1.0)),
                    "dip_direction": float(g.dip_direction) if g.dip_direction is not None else 0.0,
                    "dip": float(g.dip) if g.dip is not None else 0.0,
                })
            gr = getattr(r, "generation_result", None)
            provenance = {}
            if gr is not None:
                provenance = dict(getattr(gr, "parameter_provenance", {}))
            result.append({
                "realization_number": getattr(r, "realization_number", 0),
                "name": getattr(r, "name", ""),
                "total_p32": getattr(r, "total_p32", 0.0),
                "fractures": fractures,
                "parameter_provenance": provenance,
            })
        return result

    def _deserialize_realizations(self, data: list) -> list:
        """Deserialize DFN realizations from dicts."""
        from dfn_cave_studio.models.fracture import (
            StochasticFracture, FractureGeometry,
        )
        from dfn_cave_studio.models.dfn_realization import (
            DFNRealization, DFNGenerationResult,
        )
        from dfn_cave_studio.models.enums import FractureSource

        result = []
        for rd in data:
            fractures = []
            for fd in rd.get("fractures", []):
                c = fd["center"]
                n = fd["normal"]
                geo = FractureGeometry(
                    geometry_type="disk",
                    center_x=c[0], center_y=c[1], center_z=c[2],
                    normal_x=n[0], normal_y=n[1], normal_z=n[2],
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
        """Serialize connectivity results to JSON-safe dict."""
        if isinstance(conn, dict):
            return {k: v for k, v in conn.items()
                    if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
        return {"raw": str(conn)}

    def _build_summary(self, project) -> dict:
        """Build a summary dict from the project state."""
        summary = {
            "name": project.metadata.name if hasattr(project, "metadata") else "",
            "version": "0.6.3",
            "borehole_count": 0,
            "observation_count": 0,
            "joint_set_count": 0,
            "fracture_count": 0,
            "has_voxel_p32": False,
            "has_connectivity": False,
        }
        if (hasattr(project, "borehole_collection")
                and project.borehole_collection is not None):
            summary["borehole_count"] = len(project.borehole_collection)
            for bh in project.borehole_collection:
                summary["observation_count"] += len(bh.fracture_observations)
        summary["joint_set_count"] = len(getattr(project, "joint_sets", []))
        for r in getattr(project, "dfn_realizations", []):
            summary["fracture_count"] += len(getattr(r, "stochastic_fractures", []))
        summary["has_voxel_p32"] = getattr(project, "voxel_p32_results", None) is not None
        summary["has_connectivity"] = getattr(project, "connectivity_results", None) is not None
        return summary
