"""End-to-end tests: import → DFN → voxel → connectivity → export."""

import math
import tempfile
import os
from pathlib import Path

import numpy as np
import pytest

from dfn_cave_studio.models.bounds import ModelBounds, VoxelConfig
from dfn_cave_studio.models.fracture_set import (
    JointSetConfig, OrientationDistribution, SizeDistribution,
)
from dfn_cave_studio.models.dfn_realization import DFNGenerationConfig
from dfn_cave_studio.models.enums import SizeDistributionType
from dfn_cave_studio.dfn.generator import DFNGenerator
from dfn_cave_studio.voxel.voxel_grid import VoxelGrid
from dfn_cave_studio.voxel.dfn_voxel_intersection import DFNVoxelIntersectionEngine
from dfn_cave_studio.geometry.intersection import disk_aabb_intersection_area
from dfn_cave_studio.connectivity.connectivity_graph import ConnectivityGraph
from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter


class TestEndToEndPipeline:
    """Full pipeline from import through connectivity."""

    @pytest.fixture
    def demo_dir(self):
        return Path(__file__).parent.parent.parent / "examples" / "end_to_end_demo"

    @pytest.fixture
    def bounds(self):
        return ModelBounds(x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=200)

    @pytest.fixture
    def voxel_config(self):
        return VoxelConfig(cell_size_x=10.0, cell_size_y=10.0, cell_size_z=10.0)

    @pytest.fixture
    def joint_sets(self):
        return [
            JointSetConfig(
                set_id=1, name="Main", target_p32=0.3,
                orientation=OrientationDistribution(mean_dip_direction=45, mean_dip=60, kappa=30),
                size=SizeDistribution(distribution_type=SizeDistributionType.FIXED,
                                      min_radius=2.99, max_radius=3.01),
            ),
            JointSetConfig(
                set_id=2, name="Secondary", target_p32=0.2,
                orientation=OrientationDistribution(mean_dip_direction=135, mean_dip=75, kappa=25),
                size=SizeDistribution(distribution_type=SizeDistributionType.FIXED,
                                      min_radius=1.99, max_radius=2.01),
            ),
        ]

    def test_csv_import(self, demo_dir):
        """Import collar CSV and verify borehole collection."""
        importer = BoreholeImporter()
        result = importer.import_all(collar_path=str(demo_dir / "collars.csv"))
        assert result.success, f"Import errors: {result.errors}"
        assert len(result.collection) == 5
        bh = result.collection["BH-001"]
        assert abs(bh.collar.collar_x - 25.0) < 0.01
        assert abs(bh.collar.final_depth - 200.0) < 0.01

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

    def test_dfn_reproducibility(self, bounds, joint_sets):
        """Same seed yields identical DFN."""
        config = DFNGenerationConfig(master_seed=42, joint_sets=joint_sets)
        gen1 = DFNGenerator(config, bounds)
        gen2 = DFNGenerator(config, bounds)
        r1 = gen1.generate(0)
        r2 = gen2.generate(0)
        assert r1.generation_result.total_fractures == r2.generation_result.total_fractures
        assert abs(r1.generation_result.achieved_p32 - r2.generation_result.achieved_p32) < 1e-10

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

    def test_connectivity_pipeline(self, bounds, joint_sets):
        """Generate DFN → compute connectivity."""
        config = DFNGenerationConfig(master_seed=42, joint_sets=joint_sets)
        gen = DFNGenerator(config, bounds)
        realization = gen.generate(0)

        graph = ConnectivityGraph(realization)
        n_edges = graph.compute_edges()
        comps = graph.find_components()

        assert len(comps) > 0
        stats = graph.statistics()
        assert stats["n_edges"] == n_edges
        assert "geometric_connected" not in stats or isinstance(stats.get("n_edges", 0), int)

    def test_percolation_detection(self, bounds, joint_sets):
        """Percolation analysis returns valid results."""
        config = DFNGenerationConfig(master_seed=42, joint_sets=joint_sets)
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

    def test_voxel_pipeline(self, bounds, voxel_config, joint_sets):
        """Full DFN → voxel pipeline with intersection engine."""
        config = DFNGenerationConfig(master_seed=42, joint_sets=joint_sets)
        gen = DFNGenerator(config, bounds)
        realization = gen.generate(0)

        grid = VoxelGrid.from_bounds(bounds, voxel_config)
        from dfn_cave_studio.models.rock_mask import RockMask
        from dfn_cave_studio.models.enums import MaskType
        mask = RockMask(
            mask_type=MaskType.BOX,
            x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=200,
        )
        grid.apply_mask(mask)

        engine = DFNVoxelIntersectionEngine(grid, realization)
        n_pairs = engine.compute_intersections()

        assert n_pairs > 0, "Expected fracture-voxel intersection pairs"
        assert grid.active_voxel_count > 0
