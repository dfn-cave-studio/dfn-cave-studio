"""
Core configuration management for DFN Cave Studio.

Uses pydantic for validation and JSON for persistence.
Supports versioned configuration for project upgrade compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any
from enum import Enum, auto

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Application Version
# =============================================================================

class AppVersion(BaseModel):
    """Application version tracking."""
    major: int = 0
    minor: int = 1
    patch: int = 0
    milestone: Optional[str] = "M0"

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.milestone:
            base += f"-{self.milestone}"
        return base


# =============================================================================
# Coordinate System Configuration
# =============================================================================

class CoordinateSystem(Enum):
    """Supported coordinate reference systems."""
    LOCAL_CARTESIAN = auto()     # Local mine grid
    UTM = auto()                 # Universal Transverse Mercator
    GEOGRAPHIC = auto()          # Latitude/Longitude


class AngleUnit(Enum):
    """Angle units for input/output."""
    DEGREES = "degrees"
    RADIANS = "radians"


class LengthUnit(Enum):
    """Length units for input/output."""
    METERS = "m"
    MILLIMETERS = "mm"
    FEET = "ft"


class UnitConfig(BaseModel):
    """Unit system configuration."""
    length_unit: LengthUnit = LengthUnit.METERS
    angle_unit: AngleUnit = AngleUnit.DEGREES
    coordinate_system: CoordinateSystem = CoordinateSystem.LOCAL_CARTESIAN

    # Conversion factors to meters
    @property
    def length_to_meters(self) -> float:
        """Get conversion factor from configured length unit to meters."""
        factors = {
            LengthUnit.METERS: 1.0,
            LengthUnit.MILLIMETERS: 0.001,
            LengthUnit.FEET: 0.3048,
        }
        return factors[self.length_unit]


# =============================================================================
# Random Seed Configuration
# =============================================================================

class RandomConfig(BaseModel):
    """Random number generation configuration."""
    master_seed: int = 42
    dfn_seed: Optional[int] = None       # Override for DFN generation
    voxel_seed: Optional[int] = None     # Override for voxel operations
    block_seed: Optional[int] = None     # Override for block cutting

    def get_seed(self, category: str) -> int:
        """Get seed for a specific category, falling back to master seed."""
        seed_map = {
            "dfn": self.dfn_seed,
            "voxel": self.voxel_seed,
            "block": self.block_seed,
        }
        return seed_map.get(category) or self.master_seed


# =============================================================================
# Application Configuration
# =============================================================================

class UIConfig(BaseModel):
    """UI-related configuration."""
    language: str = "en"
    theme: str = "light"
    font_size: int = 10
    show_welcome: bool = True
    recent_files_max: int = 10


class PerformanceConfig(BaseModel):
    """Performance-related configuration."""
    max_threads: int = 0                     # 0 = auto-detect
    voxel_chunk_size: int = 32               # Default chunk size
    max_visible_voxels: int = 10_000_000     # Warning threshold
    max_fractures_display: int = 50_000      # Display cap
    auto_save_interval_minutes: int = 5


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    file_path: str = "logs/dfn_cave_studio.log"
    max_file_size_mb: int = 10
    backup_count: int = 3


class AppConfig(BaseModel):
    """Root application configuration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    version: AppVersion = Field(default_factory=AppVersion)
    units: UnitConfig = Field(default_factory=UnitConfig)
    random: RandomConfig = Field(default_factory=RandomConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Runtime paths (not persisted directly)
    project_root: Optional[Path] = None
    config_dir: Optional[Path] = None
    data_dir: Optional[Path] = None


# =============================================================================
# Global Config Access
# =============================================================================

_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global application configuration singleton."""
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global application configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = AppConfig()
