"""Integration tests for borehole import → trajectory pipeline."""

import math
import tempfile
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dfn_cave_studio.models.borehole import (
    Collar, Borehole, BoreholeCollection, BoreholeSurvey,
    SurveyStation, FractureObservation, RQDInterval,
)
from dfn_cave_studio.borehole.borehole_importer import BoreholeImporter


class TestBoreholeImportTrajectoryPipeline:
    """End-to-end: CSV import → trajectory → spatial coordinates."""

    def test_import_straight_vertical_hole(self):
        """Straight vertical hole: spatial length = measured depth."""
        # Create temp CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("borehole_id,collar_x,collar_y,collar_z,azimuth,dip,final_depth\n")
            f.write("BH-001,100,200,500,0,-90,10.0\n")
            collar_path = f.name

        try:
            importer = BoreholeImporter()
            result = importer.import_all(collar_path=collar_path)
            assert result.success, f"Import failed: {result.errors}"
            assert result.rows_imported == 1

            collection = result.collection
            bh = collection["BH-001"]
            points, mds = bh.compute_trajectory(step_length=1.0)

            # Spatial length should equal measured depth
            spatial_length = float(np.linalg.norm(points[-1] - points[0]))
            assert abs(spatial_length - 10.0) < 0.01, f"Expected 10m, got {spatial_length}m"
            # Z should decrease (vertical down)
            assert points[-1][2] < points[0][2]
            assert abs(points[-1][2] - (500 - 10.0)) < 0.01
        finally:
            os.unlink(collar_path)

    def test_import_with_survey_straight(self):
        """Straight hole with survey stations must preserve spatial length."""
        collar_csv = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        collar_csv.write("borehole_id,collar_x,collar_y,collar_z,azimuth,dip,final_depth\n")
        collar_csv.write("BH-S,0,0,0,45,-45,20.0\n")
        collar_csv.close()

        survey_csv = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        survey_csv.write("borehole_id,measured_depth,azimuth,dip\n")
        survey_csv.write("BH-S,0,45,-45\n")
        survey_csv.write("BH-S,10,45,-45\n")
        survey_csv.write("BH-S,20,45,-45\n")
        survey_csv.close()

        try:
            importer = BoreholeImporter()
            result = importer.import_all(
                collar_path=collar_csv.name,
                survey_path=survey_csv.name,
            )
            assert result.success, f"Import failed: {result.errors}"

            bh = result.collection["BH-S"]
            points, mds = bh.compute_trajectory(step_length=1.0)
            spatial_length = float(np.linalg.norm(points[-1] - points[0]))
            # With minimum curvature (straight), spatial = measured depth
            assert abs(spatial_length - 20.0) < 0.02, f"Expected ~20m, got {spatial_length}m"
        finally:
            os.unlink(collar_csv.name)
            os.unlink(survey_csv.name)

    def test_import_with_fractures(self):
        """Import collar + fracture observations and verify attachment."""
        collar_csv = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        collar_csv.write("borehole_id,collar_x,collar_y,collar_z,azimuth,dip,final_depth\n")
        collar_csv.write("BH-F,0,0,0,0,-90,50.0\n")
        collar_csv.close()

        frac_csv = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        frac_csv.write("borehole_id,measured_depth,dip_direction,dip,aperture,filling,fracture_type\n")
        frac_csv.write("BH-F,12.5,45,60,2.0,calcite,joint\n")
        frac_csv.write("BH-F,25.0,135,75,0.5,,joint\n")
        frac_csv.write("BH-F,38.2,270,30,,chlorite,shear_zone\n")
        frac_csv.close()

        try:
            importer = BoreholeImporter()
            result = importer.import_all(
                collar_path=collar_csv.name,
                fractures_path=frac_csv.name,
            )
            assert result.success, f"Import failed: {result.errors}"

            bh = result.collection["BH-F"]
            assert bh.observed_fracture_count == 3
            assert bh.fracture_observations[0].measured_depth == 12.5
            assert bh.fracture_observations[0].aperture == 2.0
            assert bh.fracture_observations[1].aperture == 0.5
            assert bh.fracture_observations[2].aperture is None

            # Verify all observations belong to BH-F
            for obs in bh.fracture_observations:
                assert obs.borehole_id == "BH-F"
        finally:
            os.unlink(collar_csv.name)
            os.unlink(frac_csv.name)

    def test_mandatory_field_rejection(self):
        """Rows missing mandatory fields must be rejected with errors."""
        collar_csv = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        collar_csv.write("borehole_id,collar_x,collar_y,collar_z,azimuth,dip,final_depth\n")
        collar_csv.write("BH-OK,100,200,300,0,-90,50\n")
        collar_csv.write("BH-BAD,100,200,,0,-90,50\n")  # missing collar_z
        collar_csv.close()

        try:
            importer = BoreholeImporter()
            result = importer.import_all(collar_path=collar_csv.name)
            # Should have errors about missing mandatory field
            assert len(result.errors) >= 1
            assert "collar_z" in str(result.errors).lower() or "missing mandatory" in str(result.errors).lower()
            # BH-OK should still be imported
            assert result.rows_imported >= 1
        finally:
            os.unlink(collar_csv.name)

    def test_locate_fractures_in_3d(self):
        """Fracture observations should be locatable in 3D space."""
        collar = Collar(borehole_id="BH-3D", collar_x=100, collar_y=200, collar_z=500,
                        azimuth=0, dip=-90, final_depth=30.0)
        bh = Borehole(borehole_id="BH-3D", collar=collar)
        obs = FractureObservation(borehole_id="BH-3D", measured_depth=15.0,
                                  dip_direction=45, dip=60)
        bh.fracture_observations.append(obs)

        pos = bh.locate_observation(obs)
        assert pos is not None
        # Vertical hole at (100,200,500): at depth 15, pos = (100,200,485)
        assert abs(pos[0] - 100) < 0.01
        assert abs(pos[1] - 200) < 0.01
        assert abs(pos[2] - 485) < 0.01

    def test_duplicate_id_detection(self):
        """Duplicate borehole IDs must be caught."""
        collar_csv = tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False)
        collar_csv.write("borehole_id,collar_x,collar_y,collar_z,azimuth,dip,final_depth\n")
        collar_csv.write("BH-DUP,100,200,300,0,-90,10\n")
        collar_csv.write("BH-DUP,150,250,350,45,-45,20\n")
        collar_csv.close()

        try:
            importer = BoreholeImporter()
            result = importer.import_all(collar_path=collar_csv.name)
            assert len(result.errors) >= 1
            assert "duplicate" in str(result.errors).lower()
        finally:
            os.unlink(collar_csv.name)
