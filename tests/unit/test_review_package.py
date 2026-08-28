"""Regression tests for clean-checkout review/source packaging."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_review_package.py"
_SPEC = importlib.util.spec_from_file_location("build_review_package", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
review_package = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(review_package)


def _make_clean_checkout(root: Path, *, optional: tuple[str, ...] = ()) -> None:
    for directory in review_package.REQUIRED_SOURCE_DIRECTORIES:
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / "tracked.txt").write_text(directory, encoding="utf-8")
    for directory in optional:
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("test", encoding="utf-8")


def _build_source_zip(root: Path) -> list[str]:
    review_dir = root / "review-payload"
    review_dir.mkdir()
    (review_dir / "REVIEW_MANIFEST.json").write_text("{}", encoding="utf-8")
    _, source_zip = review_package.build_archives(root, review_dir, root / "output", "v0.10.1-M10")
    with zipfile.ZipFile(source_zip) as archive:
        return archive.namelist()


def test_source_build_succeeds_without_sample_data(tmp_path: Path) -> None:
    _make_clean_checkout(tmp_path, optional=("resources",))

    names = _build_source_zip(tmp_path)

    assert not any("/sample_data/" in name for name in names)


def test_source_build_succeeds_without_resources(tmp_path: Path) -> None:
    _make_clean_checkout(tmp_path, optional=("sample_data",))

    names = _build_source_zip(tmp_path)

    assert not any("/resources/" in name for name in names)


def test_missing_src_fails_clearly(tmp_path: Path) -> None:
    _make_clean_checkout(tmp_path)
    for path in (tmp_path / "src").iterdir():
        path.unlink()
    (tmp_path / "src").rmdir()

    with pytest.raises(RuntimeError, match=r"Missing required source-package directory:.*src"):
        review_package.source_files(tmp_path)


def test_optional_directory_status_records_omissions(tmp_path: Path) -> None:
    _make_clean_checkout(tmp_path)

    status = review_package.source_directory_status(tmp_path)

    assert status["included_optional"] == []
    assert status["omitted_optional"] == ["resources", "sample_data"]


def test_source_zip_excludes_local_and_large_data(tmp_path: Path) -> None:
    _make_clean_checkout(tmp_path, optional=("resources", "sample_data"))
    excluded = (
        "examples/.git/config",
        "examples/venv/package.py",
        "examples/.pytest_cache/state",
        "examples/__pycache__/module.pyc",
        "examples/user.dfnproj",
        "sample_data/large.h5",
        "sample_data/store.zarr/chunk.bin",
    )
    for relative in excluded:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"excluded")

    names = _build_source_zip(tmp_path)

    assert not any(any(part in name for part in ("/.git/", "/venv/", "/__pycache__/", "/.pytest_cache/")) for name in names)
    assert not any(name.lower().endswith((".dfnproj", ".h5")) or ".zarr/" in name.lower() for name in names)
