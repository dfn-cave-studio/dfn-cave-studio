"""End-to-end tests: import → DFN → voxel → connectivity → export.

Uses real CSV data with three fracture sets (set_id 1, 2, 3).
Verifies the full data chain without manual set_id overriding.
"""

import csv
import math
import tempfile
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import (
    JointSetConfig, OrientationDistribution, SizeDistribution,
)
from dfn_cave_studio.models.dfn_realization import DFNGenerationConfig, DFNRealization
from dfn_cave_studio.models.enums import SizeDistributionType
from dfn_cave_studio.dfn.generator import DFNGenerator
from dfn_cave_studio.voxel.voxel_grid import VoxelGrid
from dfn_cave_studio.voxel.dfn_voxel_intersection import DFNVoxelIntersectionEngine
from dfn_cave_studio.geometry.intersection import disk_aabb_intersection_area
from dfn_cave_studio.connectivity.connectivity_graph import ConnectivityGraph
from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter
from dfn_cave_studio.borehole.orientation_statistics import OrientationStatisticsCalculator


class TestEndToEndPipeline:
    """Full pipeline from import through connectivity."""

    @pytest.fixture
    def demo_dir(self):
        return Path(__file__).parent.parent.parent / "examples" / "end_to_end_demo"

    # ── Small-scale model for fast end-to-end testing ──────────────────
    # Model: 5×5×5 m = 125 m³. With FIXED r≈1m, E[πR²] ≈ π ≈ 3.14 m².
    # N = P32_target * V / E[πR²] = 2.0 * 125 / 3.14 ≈ 80 per set.
    # Total ~160 fractures → ~12,720 pairwise tests (manageable).
    # Capped at max_fractures_per_set=100 for safety.

    @pytest.fixture
    def bounds(self):
        return ModelBounds(x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5)

    @pytest.fixture
    def voxel_config(self):
        return VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)

    @pytest.fixture
    def joint_sets(self):
        return [
            JointSetConfig(
                set_id=1, name="Main", target_p32=2.0,
                orientation=OrientationDistribution(mean_dip_direction=45, mean_dip=60, kappa=30),
                size=SizeDistribution(distribution_type=SizeDistributionType.FIXED,
                                      min_radius=0.99, max_radius=1.01),
            ),
            JointSetConfig(
                set_id=2, name="Secondary", target_p32=2.0,
                orientation=OrientationDistribution(mean_dip_direction=135, mean_dip=75, kappa=25),
                size=SizeDistribution(distribution_type=SizeDistributionType.FIXED,
                                      min_radius=0.99, max_radius=1.01),
            ),
        ]

    @pytest.mark.slow
    def test_csv_import(self, demo_dir):
        """Import collar CSV and verify borehole collection."""
        importer = BoreholeImporter()
        result = importer.import_all(collar_path=str(demo_dir / "collars.csv"))
        assert result.success, f"Import errors: {result.errors}"
        assert len(result.collection) == 5
        bh = result.collection["BH-001"]
        assert abs(bh.collar.collar_x - 25.0) < 0.01
        assert abs(bh.collar.final_depth - 200.0) < 0.01

    @pytest.mark.slow
    def test_borehole_trajectory(self, demo_dir):
        """Import with survey and verify trajectory spatial length."""
        importer = BoreholeImporter()
        result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            survey_path=str(demo_dir / "surveys.csv"),
        )
        assert result.success
        bh = result.collection["BH-001"]
        points, mds = bh.compute_trajectory(step_length=5.0)
        spatial_len = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        assert abs(spatial_len - 200.0) < 5.0  # ~200m hole

    @pytest.mark.slow
    def test_fracture_import_with_set_id(self, demo_dir):
        """Fractures CSV preserves real set_id 1, 2, 3 — no manual override."""
        importer = BoreholeImporter()
        result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            survey_path=str(demo_dir / "surveys.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        assert result.success, f"Import errors: {result.errors}"
        collection = result.collection

        # Collect set_ids
        set_ids = set()
        for bh in collection:
            for obs in bh.fracture_observations:
                set_ids.add(obs.set_id)
        assert 1 in set_ids, "set_id=1 missing"
        assert 2 in set_ids, "set_id=2 missing"
        assert 3 in set_ids, "set_id=3 missing"
        assert None not in set_ids, "None set_id found"

        # Verify import counts
        assert result.rows_imported > 50
        assert result.rows_skipped == 0

    @pytest.mark.slow
    def test_dfn_reproducibility(self, bounds, joint_sets):
        """Same seed yields identical DFN."""
        config = DFNGenerationConfig(master_seed=42, joint_sets=joint_sets, max_fractures_per_set=100)
        gen1 = DFNGenerator(config, bounds)
        gen2 = DFNGenerator(config, bounds)
        r1 = gen1.generate(0)
        r2 = gen2.generate(0)
        assert r1.generation_result.total_fractures == r2.generation_result.total_fractures
        assert abs(r1.generation_result.achieved_p32 - r2.generation_result.achieved_p32) < 1e-10

    @pytest.mark.slow
    def test_voxel_area_conservation(self):
        """Horizontal disk through 4 voxels: sum of clipped areas = disk area."""
        r = math.sqrt(2.0 / math.pi)  # ≈ 0.798m, disk area = 2.0 m²
        center = np.array([1.0, 1.0, 0.5])
        normal = np.array([0.0, 0.0, 1.0])

        voxels = [
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            ([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
            ([0.0, 1.0, 0.0], [1.0, 2.0, 1.0]),
            ([1.0, 1.0, 0.0], [2.0, 2.0, 1.0]),
        ]

        total = 0.0
        for vmin, vmax in voxels:
            area = disk_aabb_intersection_area(
                center, normal, r,
                np.array(vmin), np.array(vmax),
            )
            total += area

        disk_area = math.pi * r ** 2
        assert abs(total - disk_area) < 0.02, f"Sum={total:.4f} != disk_area={disk_area:.4f}"

    @pytest.mark.slow
    def test_connectivity_pipeline(self, bounds, joint_sets):
        """Generate DFN → compute connectivity."""
        config = DFNGenerationConfig(master_seed=42, joint_sets=joint_sets, max_fractures_per_set=100)
        gen = DFNGenerator(config, bounds)
        realization = gen.generate(0)

        graph = ConnectivityGraph(realization)
        n_edges = graph.compute_edges()
        comps = graph.find_components()

        assert len(comps) > 0
        stats = graph.statistics()
        assert stats["n_edges"] == n_edges
        assert isinstance(stats.get("n_edges", 0), int)

    @pytest.mark.slow
    def test_percolation_detection(self, bounds, joint_sets):
        """Percolation analysis returns valid results."""
        config = DFNGenerationConfig(master_seed=42, joint_sets=joint_sets, max_fractures_per_set=100)
        gen = DFNGenerator(config, bounds)
        realization = gen.generate(0)

        graph = ConnectivityGraph(realization)
        graph.compute_edges()
        graph.find_components()

        detail = graph.percolation_detail(
            (bounds.x_min, bounds.x_max),
            (bounds.y_min, bounds.y_max),
            (bounds.z_min, bounds.z_max),
        )
        assert "percolates_x" in detail
        assert "percolates_y" in detail
        assert "percolates_z" in detail
        assert detail["geometric_only"] is True

    @pytest.mark.slow
    def test_voxel_pipeline(self, bounds, voxel_config, joint_sets):
        """Full DFN → voxel pipeline with intersection engine."""
        config = DFNGenerationConfig(master_seed=42, joint_sets=joint_sets, max_fractures_per_set=100)
        gen = DFNGenerator(config, bounds)
        realization = gen.generate(0)

        grid = VoxelGrid.from_bounds(bounds, voxel_config)
        from dfn_cave_studio.models.rock_mask import RockMask, MaskType
        mask = RockMask(
            mask_type=MaskType.BOX,
            x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5,
        )
        grid.apply_mask(mask)

        engine = DFNVoxelIntersectionEngine(grid, realization)
        n_pairs = engine.compute_intersections()

        assert n_pairs > 0, "Expected fracture-voxel intersection pairs"
        assert grid.active_voxel_count > 0

        # Verify real voxel P32 data can be extracted
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        voxel_data = ZipProjectStore.extract_voxel_p32_results(grid)
        assert len(voxel_data) > 0, "No voxel P32 data extracted"
        # Verify non-empty P32 values exist
        p32_values = [v["local_p32"] for v in voxel_data]
        assert sum(p32_values) > 0, f"All voxel P32 values are zero: {voxel_data[:3]}"

    @pytest.mark.slow
    def test_full_workflow(self, demo_dir, tmp_path):
        """Complete workflow: import → real set_ids → observations → stats
        → DFN → real voxel P32 → connectivity → save .dfnproj → reopen
        → verify all data → export CSV/VTU."""
        # ── Step 1: Import demo data (collars, surveys, fractures) ──────
        importer = BoreholeImporter()
        result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            survey_path=str(demo_dir / "surveys.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        assert result.success, f"Import errors: {result.errors}"
        collection = result.collection
        assert len(collection) == 5

        # ── Step 2: Verify real set_ids from CSV (NOT manually assigned) ─
        set_ids = set()
        for bh in collection:
            for obs in bh.fracture_observations:
                set_ids.add(obs.set_id)
        assert 1 in set_ids, "set_id=1 not imported from CSV"
        assert 2 in set_ids, "set_id=2 not imported from CSV"
        assert 3 in set_ids, "set_id=3 not imported from CSV"
        assert None not in set_ids, "None set_id found in imported data"

        # ── Step 3: Build trajectories ──────────────────────────────────
        for bh in collection:
            points, mds = bh.compute_trajectory(step_length=5.0)
            assert len(points) >= 2, f"No trajectory for {bh.borehole_id}"

        # ── Step 4: Locate observations in 3D ───────────────────────────
        for bh in collection:
            for obs in bh.fracture_observations:
                pos = bh.locate_observation(obs)
                assert pos is not None, (
                    f"Cannot locate observation at {obs.measured_depth}m "
                    f"in {bh.borehole_id}"
                )

        # ── Step 5: Compute Fisher stats (uses real set_ids from CSV) ────
        calc = OrientationStatisticsCalculator()
        stats = calc.compute_by_set(collection)
        assert len(stats) >= 3, f"Expected stats for 3 sets, got {len(stats)}"
        for set_id in [1, 2, 3]:
            assert set_id in stats, f"No stats for set_id={set_id}"
            orient = stats[set_id]
            assert 0 <= orient.mean_dip_direction < 360
            assert 0 <= orient.mean_dip <= 90
            assert orient.kappa > 0

        # ── Step 6: Create joint sets from Fisher stats ──────────────────
        joint_sets = []
        for set_id in sorted(stats.keys()):
            orient = stats[set_id]
            js = JointSetConfig(
                set_id=set_id,
                name=f"Joint Set {set_id}",
                orientation=orient,
                size=SizeDistribution(
                    distribution_type=SizeDistributionType.FIXED,
                    min_radius=0.99, max_radius=1.01,
                ),
                target_p32=2.0,
                provenance={"orientation": "borehole", "size": "user", "p32": "user"},
            )
            joint_sets.append(js)

        # ── Step 7: Generate DFN ────────────────────────────────────────
        bounds = ModelBounds(x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5)
        voxel_config = VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)
        config = DFNGenerationConfig(master_seed=42, joint_sets=joint_sets, max_fractures_per_set=100)
        gen = DFNGenerator(config, bounds)
        realization = gen.generate(0)
        n_fractures = realization.generation_result.total_fractures
        assert n_fractures > 0, "DFN generation produced no fractures"
        assert n_fractures <= 300, f"DFN too large: {n_fractures} fractures"

        # Verify provenance
        prov = getattr(realization.generation_result, "parameter_provenance", {})
        assert len(prov) == 3, f"Expected provenance for 3 sets, got {len(prov)}"

        # ── Step 8: Real voxel P32 ─────────────────────────────────────
        grid = VoxelGrid.from_bounds(bounds, voxel_config)
        from dfn_cave_studio.models.rock_mask import RockMask, MaskType
        mask = RockMask(
            mask_type=MaskType.BOX,
            x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5,
        )
        grid.apply_mask(mask)
        engine = DFNVoxelIntersectionEngine(grid, realization)
        n_pairs = engine.compute_intersections()
        assert n_pairs > 0, "No fracture-voxel intersections"

        # Extract REAL voxel P32 (not empty dict)
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        voxel_data = ZipProjectStore.extract_voxel_p32_results(grid)
        assert len(voxel_data) > 0, "Extracted voxel data is empty"
        assert any(v["local_p32"] > 0 for v in voxel_data), "All local_p32 values are zero"

        # ── Step 9: Connectivity with cluster labels ────────────────────
        graph = ConnectivityGraph(realization)
        n_edges = graph.compute_edges()
        comps = graph.find_components()
        stats_out = graph.statistics()
        cluster_labels = graph.component_labels().tolist()
        assert len(cluster_labels) == n_fractures
        assert stats_out["n_components"] > 0

        detail = graph.percolation_detail(
            (bounds.x_min, bounds.x_max),
            (bounds.y_min, bounds.y_max),
            (bounds.z_min, bounds.z_max),
        )
        assert detail["geometric_only"] is True

        # ── Step 10: Save as .dfnproj ───────────────────────────────────
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        from dfn_cave_studio.models.project import Project

        project = Project()
        project.metadata.name = "E2E Test Project"
        project.model_bounds = bounds
        project.voxel_config = voxel_config
        project.config.master_seed = 42
        project.borehole_collection = collection
        project.joint_sets = joint_sets
        project.dfn_realizations = [realization]
        project.connectivity_results = stats_out
        project.connectivity_results["percolation"] = detail
        project.connectivity_clusters = cluster_labels
        project.voxel_p32_results = voxel_data

        save_path = tmp_path / "e2e_test.dfnproj"
        zps = ZipProjectStore()
        zps.save(project, save_path)
        assert save_path.exists(), ".dfnproj file not created"

        # Verify ZIP structure
        with zipfile.ZipFile(save_path, "r") as zf:
            names = zf.namelist()
            required = [
                "project.json",
                "inputs/boreholes.json",
                "parameters/joint_sets.json",
                "results/dfn_realizations.json",
                "results/voxel_p32.json",
                "results/connectivity.json",
                "results/summary.json",
            ]
            for r in required:
                assert r in names, f"Missing {r} in .dfnproj"

        # ── Step 11: Reopen and verify ──────────────────────────────────
        reopened = zps.load(save_path)
        assert reopened.metadata.name == "E2E Test Project"
        assert reopened.borehole_collection is not None
        assert len(reopened.borehole_collection) == 5

        # Verify fracture observations kept set_ids
        reopened_set_ids = set()
        for bh in reopened.borehole_collection:
            for obs in bh.fracture_observations:
                reopened_set_ids.add(obs.set_id)
        assert 1 in reopened_set_ids
        assert 2 in reopened_set_ids
        assert 3 in reopened_set_ids

        # Verify joint sets
        assert len(reopened.joint_sets) == 3
        for js in reopened.joint_sets:
            assert js.provenance.get("orientation") == "borehole"

        # Verify DFN
        assert len(reopened.dfn_realizations) == 1
        reopened_fractures = getattr(
            reopened.dfn_realizations[0], "stochastic_fractures", [])
        assert len(reopened_fractures) == n_fractures

        # Verify voxel P32 is NOT empty
        reopened_voxel = reopened.voxel_p32_results
        assert reopened_voxel is not None, "voxel_p32_results is None after reopen"
        assert len(reopened_voxel) > 0, "voxel_p32_results is empty after reopen"
        assert any(isinstance(v, dict) and v.get("local_p32", 0) > 0
                   for v in reopened_voxel), "All reopened P32 values are zero"

        # Verify connectivity clusters preserved
        assert reopened.connectivity_results is not None
        assert reopened.connectivity_results.get("n_edges") == n_edges
        assert reopened.connectivity_clusters is not None
        assert len(reopened.connectivity_clusters) == n_fractures

        # ── Step 12: Export CSV with content check ──────────────────────
        export_path = tmp_path / "export.csv"
        with open(export_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "x", "y", "z", "radius", "set_id"])
            for frac in realization.stochastic_fractures:
                g = frac.geometry
                writer.writerow([
                    getattr(frac, "fracture_id", ""),
                    g.center_x, g.center_y, g.center_z,
                    frac.radius, frac.set_id,
                ])
        assert export_path.stat().st_size > 0, "CSV export is empty"

        # Verify CSV content
        with open(export_path, "r") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == n_fractures, f"CSV rows {len(rows)} != {n_fractures}"
        assert all("x" in r and "y" in r and "z" in r for r in rows)
        assert any(int(r["set_id"]) == 1 for r in rows), "set_id=1 missing from CSV"

        # ── Step 13: VTU/VTI export ──────────────────────────────────────
        vti_path = tmp_path / "voxel_p32.vti"
        from dfn_cave_studio.visualization.dfn_renderer import DFNRenderer
        renderer = DFNRenderer()
        export_ok = renderer.export_voxels_vtu(
            voxel_data, bounds, voxel_config, str(vti_path),
        )
        assert export_ok, "VTU/VTI export failed"
        assert vti_path.exists(), "VTI file not created"
        assert vti_path.stat().st_size > 0, "VTI file is empty"

        # Re-read VTI file and verify P32 data
        try:
            import pyvista as pv
            imported_grid = pv.read(str(vti_path))
            assert imported_grid is not None, "Failed to re-read VTI file"
            cell_data = imported_grid.cell_data
            assert "local_p32" in cell_data, "local_p32 not in re-read VTI"
            re_read_p32 = cell_data["local_p32"]
            assert len(re_read_p32) > 0
            assert np.sum(re_read_p32) > 0, "Re-read P32 values all zero"
        except ImportError:
            pass  # PyVista not available in this env

        # ── Step 14: Size distribution consistency ──────────────────────
        # Test truncated power-law sampling matches analytic moments
        size_test = SizeDistribution(
            distribution_type=SizeDistributionType.POWER_LAW,
            min_radius=0.5, max_radius=5.0, power_law_exponent=3.0,
        )
        er_theory = size_test.mean_radius
        er2_theory = size_test.mean_squared_radius
        assert er_theory > 0
        assert er2_theory > er_theory ** 2  # Jensen's gap

        # Analytical vs sampled moments using 100K samples
        rng = np.random.default_rng(42)
        # Manual truncated power-law sampling (same as generator now)
        D = 3.0
        r_min, r_max = 0.5, 5.0
        c = r_max ** (-D) - r_min ** (-D)
        u = rng.random(100000)
        samples = (-u * c + r_max ** (-D)) ** (-1.0 / D)
        er_sample = np.mean(samples)
        er2_sample = np.mean(samples ** 2)

        assert abs(er_sample - er_theory) / er_theory < 0.02, (
            f"E[R] theory={er_theory:.5f} sample={er_sample:.5f} "
            f"error={abs(er_sample-er_theory)/er_theory*100:.2f}%"
        )
        assert abs(er2_sample - er2_theory) / er2_theory < 0.02, (
            f"E[R²] theory={er2_theory:.5f} sample={er2_sample:.5f} "
            f"error={abs(er2_sample-er2_theory)/er2_theory*100:.2f}%"
        )
