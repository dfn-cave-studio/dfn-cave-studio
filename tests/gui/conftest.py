"""Conftest for GUI tests — ensures offscreen mode before any Qt imports."""

import os


def pytest_configure(config):
    """Force offscreen rendering for all GUI tests."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # Also set PYTHONPATH to include src
    import sys
    from pathlib import Path
    src_dir = str(Path(__file__).parent.parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
