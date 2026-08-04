"""Common enumerations for DFN Cave Studio data models."""

from enum import Enum, auto


class FractureType(str, Enum):
    """Type classification for fractures."""
    JOINT = "joint"
    FAULT = "fault"
    SHEAR_ZONE = "shear_zone"
    BEDDING = "bedding"
    CONTACT = "contact"
    VEIN = "vein"
    UNKNOWN = "unknown"


class FractureSource(str, Enum):
    """Data source for fracture models."""
    BOREHOLE = "borehole"
    SURFACE_MAPPING = "surface_mapping"
    GEOPHYSICS = "geophysics"
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    USER_DEFINED = "user_defined"


class SizeDistributionType(str, Enum):
    """Types of fracture size distributions."""
    LOGNORMAL = "lognormal"
    POWER_LAW = "power_law"
    TRUNCATED_POWER_LAW = "truncated_power_law"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"


class SpatialDistributionType(str, Enum):
    """Types of fracture spatial position distributions."""
    UNIFORM = "uniform"
    CLUSTERED = "clustered"
    ZONE_SPECIFIC = "zone_specific"


class ConnectivityCriteria(str, Enum):
    """Criteria for determining fracture connectivity."""
    GEOMETRIC_INTERSECTION = "geometric_intersection"
    MECHANICAL_CONTACT = "mechanical_contact"
    PROXIMITY_THRESHOLD = "proximity_threshold"


class BoundaryCondition(str, Enum):
    """Boundary condition types for the model."""
    FREE = "free"
    FIXED = "fixed"
    ROLLER = "roller"
    STRESS = "stress"
    VELOCITY = "velocity"


class ExportFormat(str, Enum):
    """Supported export formats."""
    VTK = "vtk"
    OBJ = "obj"
    STL = "stl"
    CSV = "csv"
    JSON = "json"
    HDF5 = "hdf5"
    THREEDEC = "3dec"
    FLAC3D = "flac3d"
    PFC = "pfc"


class ProjectStatus(str, Enum):
    """Project workflow status."""
    NEW = "new"
    DATA_LOADED = "data_loaded"
    DFN_GENERATED = "dfn_generated"
    VOXELIZED = "voxelized"
    ANALYZED = "analyzed"
    VALIDATED = "validated"
    EXPORTED = "exported"
