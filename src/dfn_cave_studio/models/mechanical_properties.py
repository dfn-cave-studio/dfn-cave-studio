"""
Mechanical properties models for fractures and rock mass.

Properties can be assigned:
  - Per fracture set (joint set → same properties for all fractures in set)
  - Per structural domain (domain → overrides per-domain)
  - Per fracture type (joint, fault, shear_zone, etc.)
  - Per filling type (clay, calcite, etc.)
  - Per individual fracture (direct assignment)

Resolution order (highest priority first):
  1. Direct assignment to fracture (mechanical_property_id)
  2. Filling type match
  3. Fracture type match
  4. Structural domain match
  5. Fracture set template

References:
  - Barton, N. & Choubey, V. (1977). The shear strength of rock joints.
  - SCIENTIFIC_SPEC.md Section 9.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from dfn_cave_studio.models.enums import FractureType


# =============================================================================
# Fracture Mechanical Properties (per-fracture instance)
# =============================================================================

class FractureMechanicalProperties(BaseModel):
    """Mechanical properties of a single fracture.

    Follows the Barton-Bandis and Mohr-Coulomb conventions.
    All stress values in Pascals (Pa), angles in degrees.
    """

    # Shear strength (Mohr-Coulomb)
    cohesion: float = Field(default=0.0, ge=0.0, description="Cohesion (Pa)")
    friction_angle: float = Field(default=30.0, ge=0.0, lt=90.0, description="Friction angle (°)")
    tensile_strength: float = Field(default=0.0, ge=0.0, description="Tensile strength (Pa)")

    # Stiffness
    normal_stiffness: float = Field(default=1e9, gt=0.0, description="Normal stiffness (Pa/m)")
    shear_stiffness: float = Field(default=1e9, gt=0.0, description="Shear stiffness (Pa/m)")

    # Dilation
    dilation_angle: float = Field(default=0.0, ge=0.0, lt=90.0, description="Dilation angle (°)")

    # Residual strength (post-peak)
    residual_cohesion: float = Field(default=0.0, ge=0.0, description="Residual cohesion (Pa)")
    residual_friction_angle: float = Field(default=30.0, ge=0.0, lt=90.0, description="Residual friction angle (°)")

    # Physical
    aperture: float = Field(default=0.0, ge=0.0, description="Hydraulic/mechanical aperture (m)")
    filling_type: Optional[str] = Field(default=None, description="Filling material: 'clay', 'calcite', 'gouge', etc.")

    # Version
    model_version: int = 1

    def to_barton_bandis(self, jrc: float, jcs: float, sigma_n: float) -> Dict[str, float]:
        """Estimate Barton-Bandis parameters from Mohr-Coulomb.

        This is an APPROXIMATION. Direct JRC/JCS measurements preferred.

        Args:
            jrc: Joint Roughness Coefficient (0-20).
            jcs: Joint Compressive Strength (Pa).
            sigma_n: Normal stress (Pa).

        Returns:
            Dict with estimated JRC, JCS, phi_r, tau_peak.
        """
        import math
        phi_r = self.residual_friction_angle
        tau_peak = sigma_n * math.tan(math.radians(
            phi_r + jrc * math.log10(jcs / sigma_n) if sigma_n > 0 else phi_r
        ))
        return {"jrc": jrc, "jcs": jcs, "phi_r": phi_r, "tau_peak": tau_peak}


# =============================================================================
# Mechanical Property Template
# =============================================================================

class MechanicalPropertyTemplate(BaseModel):
    """A named template that can be assigned to fractures by category."""

    template_id: UUID = Field(default_factory=uuid4)
    name: str = "Default Template"
    description: str = ""
    properties: FractureMechanicalProperties = Field(default_factory=FractureMechanicalProperties)

    # Assignment rules (all optional — matches if field is non-None)
    apply_to_set_id: Optional[int] = Field(default=None, description="Apply to this fracture set")
    apply_to_domain_id: Optional[int] = Field(default=None, description="Apply to this structural domain")
    apply_to_fracture_type: Optional[FractureType] = Field(default=None, description="Apply to this fracture type")
    apply_to_filling: Optional[str] = Field(default=None, description="Apply to fractures with this filling")

    # Priority (higher = checked first)
    priority: int = Field(default=0)

    model_version: int = 1

    def matches_fracture(
        self,
        set_id: Optional[int] = None,
        domain_id: Optional[int] = None,
        fracture_type: Optional[FractureType] = None,
        filling: Optional[str] = None,
    ) -> bool:
        """Check if this template matches the given fracture attributes.

        Returns True if ALL non-None template criteria match.
        """
        if self.apply_to_set_id is not None and set_id != self.apply_to_set_id:
            return False
        if self.apply_to_domain_id is not None and domain_id != self.apply_to_domain_id:
            return False
        if self.apply_to_fracture_type is not None and fracture_type != self.apply_to_fracture_type:
            return False
        if self.apply_to_filling is not None and filling != self.apply_to_filling:
            return False
        return True


# =============================================================================
# Mechanical Property Library
# =============================================================================

class MechanicalPropertyLibrary(BaseModel):
    """Collection of mechanical property templates with resolution logic.

    When a fracture needs properties, the library finds the best-matching
    template based on the resolution hierarchy.
    """

    templates: List[MechanicalPropertyTemplate] = Field(default_factory=list)
    default_properties: FractureMechanicalProperties = Field(
        default_factory=FractureMechanicalProperties
    )
    model_version: int = 1

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure default template exists
        if not any(t.name == "Default" for t in self.templates):
            self.templates.insert(0, MechanicalPropertyTemplate(
                name="Default",
                properties=self.default_properties,
                priority=-1,
            ))

    def resolve(
        self,
        set_id: Optional[int] = None,
        domain_id: Optional[int] = None,
        fracture_type: Optional[FractureType] = None,
        filling: Optional[str] = None,
    ) -> FractureMechanicalProperties:
        """Find the best-matching mechanical properties for given fracture attributes.

        Resolution order (sorted by priority, descending):
          1. Direct template_id match (handled by caller)
          2. Filling type match
          3. Fracture type match
          4. Structural domain match
          5. Fracture set match
          6. Default

        Args:
            set_id: Fracture set ID.
            domain_id: Structural domain ID.
            fracture_type: Fracture type.
            filling: Filling material type.

        Returns:
            Best-matching FractureMechanicalProperties.
        """
        # Sort by priority (descending)
        sorted_templates = sorted(
            self.templates,
            key=lambda t: t.priority,
            reverse=True,
        )

        # Try specific matches first
        for template in sorted_templates:
            if template.matches_fracture(set_id, domain_id, fracture_type, filling):
                return template.properties

        # Fallback to default
        return self.default_properties

    def add_template(self, template: MechanicalPropertyTemplate) -> None:
        """Add a new template to the library."""
        self.templates.append(template)

    def get_templates_for_set(self, set_id: int) -> List[MechanicalPropertyTemplate]:
        """Get all templates assigned to a specific fracture set."""
        return [t for t in self.templates if t.apply_to_set_id == set_id]

    def get_templates_for_domain(self, domain_id: int) -> List[MechanicalPropertyTemplate]:
        """Get all templates assigned to a specific structural domain."""
        return [t for t in self.templates if t.apply_to_domain_id == domain_id]
