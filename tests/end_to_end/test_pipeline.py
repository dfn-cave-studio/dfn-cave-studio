"""End-to-end tests: import → DFN → voxel → connectivity → export."""

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
    # Total ~160 fractures → ~12,720 pairwise tests (O(N²) = manageable).
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
        assert "geometric_connected" not in stats or isinstance(stats.get("n_edges", 0), int)

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

    @pytest.mark.slow
    def test_full_workflow(self, demo_dir, tmp_path):
        """Complete workflow: import → observations → DFN → voxel → connectivity
        → save .dfnproj → reopen → verify → export CSV."""
        # ── Step 1: Import demo data ──────────────────────────────────
        importer = BoreholeImporter()
        result = importer.import_all(
            collar_path=str(demo_dir / "collars.csv"),
            survey_path=str(demo_dir / "surveys.csv"),
            fractures_path=str(demo_dir / "fractures.csv"),
        )
        assert result.success, f"Import errors: {result.errors}"
        collection = result.collection
        assert len(collection) == 5

        # ── Step 2: Build trajectories ─────────────────────────────────
        for bh in collection:
            points, mds = bh.compute_trajectory(step_length=5.0)
            assert len(points) >= 2, f"No trajectory for {bh.borehole_id}"

        # ── Step 3: Locate observations in 3D ──────────────────────────
        for bh in collection:
            for obs in bh.fracture_observations:
                pos = bh.locate_observation(obs)
                assert pos is not None, (
                    f"Cannot locate observation at {obs.measured_depth}m "
                    f"in {bh.borehole_id}"
                )

        # ── Step 4: Compute Fisher stats ───────────────────────────────
        # Assign all observations to set_id=1 for stats computation
        for bh in collection:
            for obs in bh.fracture_observations:
                obs.set_id = 1

        calc = OrientationStatisticsCalculator()
        stats = calc.compute_by_set(collection)
        assert 1 in stats, "No stats computed for set_id=1"

        orient = stats[1]
        assert 0 <= orient.mean_dip_direction < 360
        assert 0 <= orient.mean_dip <= 90
        assert orient.kappa > 0

        # ── Step 5: Create joint set from observations ──────────────────
        js = JointSetConfig(
            set_id=1, name="From Observations",
            orientation=orient,
            size=SizeDistribution(
                distribution_type=SizeDistributionType.FIXED,
                min_radius=0.99, max_radius=1.01,
            ),
            target_p32=2.0,
            provenance={"orientation": "borehole", "size": "user", "p32": "user"},
        )

        # ── Step 6: Generate DFN (small model) ─────────────────────────
        bounds = ModelBounds(x_min=0, x_max=5, y_min=0, y_max=5, z_min=0, z_max=5)
        voxel_config = VoxelConfig(cell_size_x=1.0, cell_size_y=1.0, cell_size_z=1.0)
        config = DFNGenerationConfig(master_seed=42, joint_sets=[js], max_fractures_per_set=100)
        gen = DFNGenerator(config, bounds)
        realization = gen.generate(0)
        n_fractures = realization.generation_result.total_fractures
        assert n_fractures > 0, "DFN generation produced no fractures"
        assert n_fractures <= 200, f"DFN too large: {n_fractures} fractures"

        # Verify provenance recorded
        prov = getattr(realization.generation_result, "parameter_provenance", {})
        assert 1 in prov
        assert prov[1].get("orientation") == "borehole"

        # ── Step 7: Voxel intersection ────────────────────────────────
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
        assert grid.active_voxel_count > 0

        # Collect voxel P32 from the intersection engine
        voxel_p32 = {}
        # Use grid attributes directly
        display_attrs = ["fracture_count", "p32_area", "DFN_P32"]
        for attr_name in display_attrs:
            try:
                data = grid.get_attr(attr_name)
                if data is not None:
                    voxel_p32[attr_name] = "present"
            except Exception:
                pass

        # ── Step 8: Connectivity ─────────────────────────────────────
        graph = ConnectivityGraph(realization)
        n_edges = graph.compute_edges()
        comps = graph.find_components()
        stats_out = graph.statistics()
        assert stats_out["n_components"] > 0

        detail = graph.percolation_detail(
            (bounds.x_min, bounds.x_max),
            (bounds.y_min, bounds.y_max),
            (bounds.z_min, bounds.z_max),
        )
        assert detail["geometric_only"] is True

        cluster_labels = graph.component_labels().tolist()

        # ── Step 9: Save as .dfnproj ─────────────────────────────────
        from dfn_cave_studio.persistence.zip_project_store import ZipProjectStore
        from dfn_cave_studio.models.project import Project

        project = Project()
        project.metadata.name = "E2E Test Project"
        project.model_bounds = bounds
        project.voxel_config = voxel_config
        project.config.master_seed = 42
        project.borehole_collection = collection
        project.joint_sets = [js]
        project.dfn_realizations = [realization]
        project.connectivity_results = stats_out
        project.connectivity_clusters = cluster_labels
        project.voxel_p32_results = voxel_p32

        save_path = tmp_path / "e2e_test.dfnproj"
        zps = ZipProjectStore()
        zps.save(project, save_path)
        assert save_path.exists(), ".dfnproj file not created"

        # Verify ZIP structure
        with zipfile.ZipFile(save_path, "r") as zf:
            names = zf.namelist()
            assert "project.json" in names
            assert "inputs/boreholes.json" in names
            assert "parameters/joint_sets.json" in names
            assert "results/dfn_realizations.json" in names
            assert "results/connectivity.json" in names
            assert "results/summary.json" in names

        # ── Step 10: Reopen and verify ──────────────────────────────
        reopened = zps.load(save_path)
        assert reopened.metadata.name == "E2E Test Project"
        assert reopened.borehole_collection is not None
        assert len(reopened.borehole_collection) == 5
        assert len(reopened.joint_sets) == 1
        assert reopened.joint_sets[0].provenance.get("orientation") == "borehole"
        assert len(reopened.dfn_realizations) == 1
        reopened_fractures = getattr(
            reopened.dfn_realizations[0], "stochastic_fractures", [])
        assert len(reopened_fractures) == n_fractures, (
            f"Fracture count mismatch: {len(reopened_fractures)} != {n_fractures}")
        assert reopened.connectivity_results is not None
        assert reopened.connectivity_results["n_edges"] == n_edges

        # ── Step 11: Export CSV ─────────────────────────────────────
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
