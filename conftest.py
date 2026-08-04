"""Root conftest.py - shared test fixtures and configuration for DFN Cave Studio."""

import sys
from pathlib import Path

import pytest

# Ensure src is on path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


# =============================================================================
# Core Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def src_path() -> Path:
    """Return the src directory."""
    return Path(__file__).resolve().parent / "src"


@pytest.fixture
def fixed_seed() -> int:
    """Return a fixed random seed for reproducible tests."""
    return 12345


@pytest.fixture
def rng(fixed_seed: int):
    """Return a seeded numpy random generator."""
    import numpy as np
    return np.random.default_rng(fixed_seed)


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def app_config():
    """Return a default application configuration."""
    from dfn_cave_studio.core.config import AppConfig
    return AppConfig()


@pytest.fixture
def unit_config():
    """Return a default unit configuration (meters, degrees)."""
    from dfn_cave_studio.core.config import UnitConfig
    return UnitConfig()


# =============================================================================
# Model Bounds Fixture
# =============================================================================

@pytest.fixture
def default_bounds():
    """Return a default model bounding box (100m x 100m x 100m)."""
    from dfn_cave_studio.models.bounds import ModelBounds
    return ModelBounds(
        x_min=0.0, x_max=100.0,
        y_min=0.0, y_max=100.0,
        z_min=0.0, z_max=100.0,
    )


# =============================================================================
# GUI Test Fixtures
# =============================================================================

@pytest.fixture
def qapp():
    """Create a QApplication for GUI tests."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def main_window(qapp):
    """Create a MainWindow for GUI tests."""
    from dfn_cave_studio.ui.main_window import MainWindow
    window = MainWindow()
    yield window
    window.close()
