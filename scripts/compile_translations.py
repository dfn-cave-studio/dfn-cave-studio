"""Compile checked-in Qt TS catalogs using the active PySide6 installation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Compile every application TS catalog to a colocated QM file."""
    root = Path(__file__).resolve().parents[1]
    catalog_dir = root / "src" / "dfn_cave_studio" / "resources" / "i18n"
    try:
        import PySide6
    except ImportError as exc:
        raise RuntimeError("PySide6 is required to compile translations") from exc
    executable = Path(PySide6.__file__).resolve().parent / "lrelease.exe"
    if not executable.is_file():
        executable = Path(PySide6.__file__).resolve().parent / "lrelease"
    if not executable.is_file():
        raise RuntimeError(f"PySide6 lrelease was not found beside {PySide6.__file__}")
    catalogs = sorted(catalog_dir.glob("*.ts"))
    if not catalogs:
        raise RuntimeError(f"No TS catalogs found in {catalog_dir}")
    for source in catalogs:
        target = source.with_suffix(".qm")
        subprocess.run([str(executable), str(source), "-qm", str(target)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
