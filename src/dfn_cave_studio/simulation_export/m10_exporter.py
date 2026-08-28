"""Basic, non-solver-specific M10 explicit DFN exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from dfn_cave_studio.dfn.m10_geometry import (
    code_name,
    fracture_ids,
    iter_polygons,
    record_id,
    source_mask,
    source_name,
)
from dfn_cave_studio.geometry.coordinate import normal_to_dip_dir_dip
from dfn_cave_studio.models.m10 import M10FractureSource, M10Realization


class M10Exporter:
    """Export one complete M10 realization in auditable generic formats."""

    def export_all(self, realization: M10Realization, directory: Path) -> list[Path]:
        """Write all supported M10 artifacts and return their paths."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = [
            self.export_csv(realization, directory / "fractures.csv"),
            self.export_summary_json(realization, directory / "realization_summary.json"),
            self.export_summary_csv(realization, directory / "realization_summary.csv"),
            self.export_npz(realization, directory / "fractures.npz"),
            self.export_vtp(realization, directory / "fractures.vtp"),
        ]
        for source, filename in (
            (M10FractureSource.CONDITIONED_OBSERVATION.value, "conditioned_fractures.csv"),
            (M10FractureSource.DETERMINISTIC_STRUCTURE.value, "deterministic_structures.csv"),
        ):
            paths.append(self.export_csv(realization, directory / filename, source_filter=source))
        return paths

    def export_csv(self, realization: M10Realization, path: Path, source_filter: str | None = None) -> Path:
        """Export fracture centers, orientations, radii and provenance as CSV."""
        arrays = realization.geometry_arrays
        indices = range(realization.fracture_count)
        if source_filter is not None:
            indices = np.flatnonzero(source_mask(realization, source_filter)).tolist()
        ids = fracture_ids(realization)
        fields = [
            "fracture_id", "center_x", "center_y", "center_z", "normal_x", "normal_y", "normal_z",
            "dip_direction", "dip", "radius", "original_area", "clipped_area", "domain_id", "set_id",
            "source", "observation_record_id", "distribution", "size_source", "orientation_source",
            "size_class", "dip_only_density_evidence", "voxel_i", "voxel_j", "voxel_k",
            "provenance",
        ]
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in indices:
                center, normal, voxel = arrays["center"][index], arrays["normal"][index], arrays["voxel_index"][index]
                dip_direction, dip = normal_to_dip_dir_dip(np.array(normal, dtype=float, copy=True))
                if "distribution_code" in arrays:
                    distribution = code_name(realization, "distribution_codes", arrays["distribution_code"][index])
                    size_source = code_name(realization, "size_source_codes", arrays["size_source_code"][index])
                    orientation_source = code_name(
                        realization, "orientation_source_codes", arrays["orientation_source_code"][index]
                    )
                else:
                    distribution = arrays["distribution"][index]
                    size_source = arrays["size_source"][index]
                    orientation_source = arrays["orientation_source"][index]
                writer.writerow(
                    {
                        "fracture_id": ids[index],
                        "center_x": center[0], "center_y": center[1], "center_z": center[2],
                        "normal_x": normal[0], "normal_y": normal[1], "normal_z": normal[2],
                        "dip_direction": dip_direction, "dip": dip,
                        "radius": arrays["radius"][index], "original_area": arrays["original_area"][index],
                        "clipped_area": arrays["clipped_area"][index], "domain_id": arrays["domain_id"][index],
                        "set_id": arrays["set_id"][index], "source": source_name(realization, index),
                        "observation_record_id": record_id(realization, index),
                        "distribution": distribution, "size_source": size_source,
                        "orientation_source": orientation_source,
                        "size_class": realization.provenance.get("size_class_codes", {}).get(
                            str(int(arrays.get("size_class", np.zeros(realization.fracture_count, dtype=np.uint8))[index])),
                            "UNKNOWN",
                        ),
                        "dip_only_density_evidence": bool(arrays["dip_only_density_evidence"][index]),
                        "voxel_i": voxel[0], "voxel_j": voxel[1], "voxel_k": voxel[2],
                        "provenance": json.dumps(realization.provenance.get("model_sources", {}), sort_keys=True),
                    }
                )
        return Path(path)

    @staticmethod
    def export_summary_json(realization: M10Realization, path: Path) -> Path:
        """Export complete realization metadata and quality report as JSON."""
        payload = realization.model_dump(mode="json", exclude={"geometry_arrays"})
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return Path(path)

    @staticmethod
    def export_summary_csv(realization: M10Realization, path: Path) -> Path:
        """Export one-row realization summary as CSV."""
        quality = realization.quality
        row = {
            "realization_id": realization.realization_id,
            "seed": realization.seed,
            "fracture_count": quality.fracture_count,
            "stochastic_count": quality.stochastic_count,
            "conditioned_count": quality.conditioned_count,
            "deterministic_count": quality.deterministic_count,
            "target_p32": quality.target_p32,
            "generated_original_p32": quality.generated_original_p32,
            "generated_clipped_p32": quality.generated_clipped_p32,
            "p32_explicit_target": quality.p32_explicit_target,
            "p32_subgrid": quality.p32_subgrid,
            "p32_unresolved_orientation": quality.p32_unresolved_orientation,
            "p32_target_conservation_error": quality.p32_target_conservation_error,
            "local_p32_status": quality.local_p32_status.value,
        }
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return Path(path)

    @staticmethod
    def export_npz(realization: M10Realization, path: Path) -> Path:
        """Export complete geometry arrays as compressed NPZ."""
        np.savez_compressed(path, **realization.geometry_arrays)
        return Path(path)

    @staticmethod
    def export_vtp(realization: M10Realization, path: Path) -> Path:
        """Export clipped fracture polygons as generic VTP PolyData."""
        import pyvista as pv

        arrays = realization.geometry_arrays
        points: list[np.ndarray] = []
        faces: list[int] = []
        fracture_indices: list[int] = []
        offset = 0
        for index, polygon in iter_polygons(realization):
            count = len(polygon)
            if count < 3:
                continue
            points.extend(polygon)
            faces.extend([count, *range(offset, offset + count)])
            fracture_indices.append(index)
            offset += count
        mesh = pv.PolyData(np.asarray(points, dtype=float).reshape((-1, 3)), np.asarray(faces, dtype=np.int64))
        if fracture_indices:
            mesh.cell_data["set_id"] = arrays["set_id"][fracture_indices]
            mesh.cell_data["domain_id"] = arrays["domain_id"][fracture_indices]
            mesh.cell_data["radius"] = arrays["radius"][fracture_indices]
            mesh.cell_data["original_area"] = arrays["original_area"][fracture_indices]
            mesh.cell_data["clipped_area"] = arrays["clipped_area"][fracture_indices]
            mesh.cell_data["size_class"] = arrays.get(
                "size_class", np.zeros(realization.fracture_count, dtype=np.uint8)
            )[fracture_indices]
        mesh.save(path)
        return Path(path)
