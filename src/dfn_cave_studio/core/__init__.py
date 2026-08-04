"""
Core package for DFN Cave Studio.

Contains configuration, service locator, and cross-cutting concerns.
"""

from dfn_cave_studio.core.config import (
    AppConfig,
    AppVersion,
    UnitConfig,
    RandomConfig,
    UIConfig,
    PerformanceConfig,
    LoggingConfig,
    CoordinateSystem,
    AngleUnit,
    LengthUnit,
    get_config,
    set_config,
    reset_config,
)

__all__ = [
    "AppConfig",
    "AppVersion",
    "UnitConfig",
    "RandomConfig",
    "UIConfig",
    "PerformanceConfig",
    "LoggingConfig",
    "CoordinateSystem",
    "AngleUnit",
    "LengthUnit",
    "get_config",
    "set_config",
    "reset_config",
]
