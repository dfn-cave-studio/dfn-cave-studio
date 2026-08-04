"""Tests for M1 data models: structural domains, rock masks, mechanical properties."""

import math
import numpy as np
import pytest

from dfn_cave_studio.models.structural_domain import (
    StructuralDomain, StructuralDomainCollection, DomainPolygonBoundary,
    DomainBoundaryType,
)
from dfn_cave_studio.models.rock_mask import (
    RockMask, ExcavationMask, SurfaceModel, MaskType, SpatialAttributeConfig,
)
from dfn_cave_studio.models.mechanical_properties import (
    FractureMechanicalProperties, MechanicalPropertyTemplate,
    MechanicalPropertyLibrary,
)
from dfn_cave_studio.models.enums import FractureType


# =============================================================================
# Structural Domain Tests
# =============================================================================

class TestStructuralDomain:
    def test_global_domain_contains_all(self):
        domain = StructuralDomain(domain_id=0, boundary_type="global")
        assert domain.contains_point(0, 0, 0)
        assert domain.contains_point(1e9, 1e9, 1e9)
        assert domain.contains_point(-1e9, -1e9, -1e9)

    def test_box_domain(self):
        domain = StructuralDomain(
            domain_id=1, name="Box Domain", boundary_type="manual_box",
            x_min=0, x_max=100, y_min=0, y_max=100, z_min=50, z_max=150,
        )
        assert domain.contains_point(50, 50, 100)
        assert not domain.contains_point(50, 50, 25)
        assert not domain.contains_point(200, 50, 100)

    def test_polygon_domain(self):
        polygon = DomainPolygonBoundary(
            vertices=[[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0]],
            z_min=0, z_max=100,
        )
        domain = StructuralDomain(
            domain_id=2, name="Polygon Domain", boundary_type="manual_polygon",
            polygon_boundary=polygon,
        )
        assert domain.contains_point(50, 50, 50)
        assert not domain.contains_point(50, 150, 50)  # Outside polygon


class TestStructuralDomainCollection:
    def test_default_global_domain(self):
        coll = StructuralDomainCollection()
        assert len(coll.domains) == 1
        assert coll.domains[0].domain_id == 0

    def test_find_domain(self):
        coll = StructuralDomainCollection()
        coll.domains.append(StructuralDomain(
            domain_id=1, boundary_type="manual_box",
            x_min=0, x_max=50, y_min=0, y_max=50, z_min=0, z_max=50,
        ))
        domain = coll.find_domain(25, 25, 25)
        assert domain.domain_id == 1

        domain2 = coll.find_domain(75, 75, 75)
        assert domain2.domain_id == 0  # Falls back to global


# =============================================================================
# Rock Mask Tests
# =============================================================================

class TestSurfaceModel:
    def test_empty_surface(self):
        s = SurfaceModel()
        assert s.vertex_array is None
        assert s.elevation_at(0, 0) is None

    def test_simple_triangle(self):
        # Single triangle: (0,0,0), (10,0,0), (0,10,10)
        s = SurfaceModel(
            name="Slope",
            vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 10]],
            faces=[[0, 1, 2]],
        )
        # Point inside triangle
        z = s.elevation_at(1, 1)
        assert z is not None
        assert 0 <= z <= 10  # Should be interpolated

    def test_point_outside_mesh(self):
        s = SurfaceModel(
            vertices=[[0, 0, 0], [10, 0, 0], [0, 10, 10]],
            faces=[[0, 1, 2]],
        )
        z = s.elevation_at(100, 100)
        assert z is None


class TestRockMask:
    def test_box_mask(self):
        mask = RockMask(
            mask_type=MaskType.BOX,
            x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=100,
        )
        assert mask.contains_point(50, 50, 50)
        assert not mask.contains_point(150, 50, 50)

    def test_disabled_mask(self):
        mask = RockMask(enabled=False)
        assert mask.contains_point(50, 50, 50)  # Everything is rock

    def test_contains_aabb(self):
        mask = RockMask(
            mask_type=MaskType.BOX,
            x_min=0, x_max=100, y_min=0, y_max=100, z_min=0, z_max=100,
        )
        inside_box = np.array([25, 25, 25])
        outside_box = np.array([26, 26, 26])
        result = mask.contains_aabb(inside_box, outside_box)
        assert result == 2  # Fully inside

        result2 = mask.contains_aabb(np.array([-10, -10, -10]), np.array([-5, -5, -5]))
        assert result2 == 0  # Fully outside


class TestExcavationMask:
    def test_empty_mask(self):
        mask = ExcavationMask()
        assert not mask.is_excavated(50, 50, 50)

    def test_box_excavation(self):
        mask = ExcavationMask(
            excavation_boxes=[(0, 10, 0, 10, 0, 10)],
        )
        assert mask.is_excavated(5, 5, 5)
        assert not mask.is_excavated(50, 50, 50)

    def test_disabled_mask(self):
        mask = ExcavationMask(
            enabled=False,
            excavation_boxes=[(0, 10, 0, 10, 0, 10)],
        )
        assert not mask.is_excavated(5, 5, 5)


class TestSpatialAttributeConfig:
    def test_defaults(self):
        config = SpatialAttributeConfig()
        assert config.active_default == 1
        assert config.void_default == -1
        assert config.invalid_default == -2
        # Must NOT use 0 for both active AND void (distinction required)
        assert config.active_default != config.void_default
        assert config.void_default != 0


# =============================================================================
# Mechanical Properties Tests
# =============================================================================

class TestFractureMechanicalProperties:
    def test_defaults(self):
        props = FractureMechanicalProperties()
        assert props.cohesion == 0.0
        assert props.friction_angle == 30.0
        assert props.normal_stiffness == 1e9

    def test_custom(self):
        props = FractureMechanicalProperties(
            cohesion=1e6, friction_angle=35, tensile_strength=5e5,
            normal_stiffness=5e9, shear_stiffness=3e9,
        )
        assert props.cohesion == 1e6
        assert props.friction_angle == 35


class TestMechanicalPropertyTemplate:
    def test_match_by_set(self):
        tpl = MechanicalPropertyTemplate(
            name="Set 1 Props",
            apply_to_set_id=1,
            priority=10,
        )
        assert tpl.matches_fracture(set_id=1)
        assert not tpl.matches_fracture(set_id=2)

    def test_match_by_fracture_type(self):
        tpl = MechanicalPropertyTemplate(
            name="Fault Props",
            apply_to_fracture_type=FractureType.FAULT,
            priority=5,
        )
        assert tpl.matches_fracture(fracture_type=FractureType.FAULT)
        assert not tpl.matches_fracture(fracture_type=FractureType.JOINT)

    def test_match_multiple_criteria(self):
        tpl = MechanicalPropertyTemplate(
            name="Specific Props",
            apply_to_set_id=1,
            apply_to_domain_id=2,
            apply_to_fracture_type=FractureType.JOINT,
            priority=10,
        )
        # All criteria match
        assert tpl.matches_fracture(
            set_id=1, domain_id=2, fracture_type=FractureType.JOINT,
        )
        # One mismatch
        assert not tpl.matches_fracture(
            set_id=1, domain_id=2, fracture_type=FractureType.FAULT,
        )


class TestMechanicalPropertyLibrary:
    def test_default_library(self):
        lib = MechanicalPropertyLibrary()
        assert len(lib.templates) >= 1

    def test_resolve_returns_default(self):
        lib = MechanicalPropertyLibrary()
        props = lib.resolve()
        assert isinstance(props, FractureMechanicalProperties)

    def test_resolve_by_set(self):
        soft_props = FractureMechanicalProperties(cohesion=1e5, friction_angle=25)
        lib = MechanicalPropertyLibrary()
        lib.add_template(MechanicalPropertyTemplate(
            name="Soft Set",
            apply_to_set_id=1,
            properties=soft_props,
            priority=10,
        ))
        resolved = lib.resolve(set_id=1)
        assert resolved.cohesion == 1e5
        assert resolved.friction_angle == 25

    def test_priority_resolution(self):
        """Template with higher priority should win."""
        low_priority = FractureMechanicalProperties(cohesion=1e5, friction_angle=25)
        high_priority = FractureMechanicalProperties(cohesion=2e6, friction_angle=45)

        lib = MechanicalPropertyLibrary()
        lib.add_template(MechanicalPropertyTemplate(
            name="Low Priority", apply_to_set_id=1,
            properties=low_priority, priority=5,
        ))
        lib.add_template(MechanicalPropertyTemplate(
            name="High Priority", apply_to_set_id=1,
            properties=high_priority, priority=10,
        ))

        resolved = lib.resolve(set_id=1)
        assert resolved.cohesion == 2e6  # High priority wins

    def test_add_template(self):
        lib = MechanicalPropertyLibrary()
        initial_count = len(lib.templates)
        lib.add_template(MechanicalPropertyTemplate(name="New Template"))
        assert len(lib.templates) == initial_count + 1
