#!/usr/bin/env python
"""
Build the review package for a milestone release.

Collects test results, screenshots, and metadata into docs/review/<version>/
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_version() -> str:
    """Extract version from pyproject.toml or git tag."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "0.1.0-M0"


def get_commit_sha() -> str:
    """Get current commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def run_tests_and_collect_results(review_dir: Path) -> dict:
    """Run pytest and collect results."""
    print("Running tests...")
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/", "-v", "--tb=short",
            "--json-report",
            f"--json-report-file={review_dir / 'test_results.json'}",
        ],
        capture_output=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    return {
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
    }


def generate_manifest(
    version: str,
    commit_sha: str,
    review_dir: Path,
    test_results: dict,
) -> None:
    """Generate REVIEW_MANIFEST.json."""
    import sys as _sys

    manifest = {
        "project_name": "DFN Cave Studio",
        "version": version,
        "milestone": "M0",
        "commit_sha": commit_sha,
        "branch": "main",
        "created_at": datetime.utcnow().isoformat(),
        "python_version": f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
        "operating_system": f"{_sys.platform}",
        "dependency_lock_hash": "N/A - initial milestone",
        "completed_features": [
            "Project repository initialized",
            "Directory structure created",
            "Core configuration module",
            "Qt adapter layer",
            "Main window with menus and docks",
            "Basic data models (bounds, enums)",
            "Logging setup",
            "Test framework configured",
            "GitHub Actions CI workflow",
        ],
        "incomplete_features": [
            "DFN generation (M2)",
            "Voxel grid (M3)",
            "Borehole import (M6)",
            "Connectivity analysis (M5)",
            "Fragmentation analysis (M8)",
            "Project save/load (M1)",
            "3D visualization integration",
        ],
        "experimental_features": [],
        "changed_files": [],
        "tests_total": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "coverage_percent": 0.0,
        "scientific_cases": 0,
        "performance_cases": 0,
        "known_issues": [
            "M0 provides only UI skeleton - no scientific functionality",
            "3D view requires PyVistaQt which needs GPU driver support",
            "Some menu items are placeholders",
        ],
        "release_url": "",
        "source_archive": "",
        "previous_version": "N/A",
        "reproducibility_command": (
            "python -m venv .venv && "
            "source .venv/Scripts/activate && "
            "pip install -r requirements.txt && "
            "pytest tests/"
        ),
    }

    manifest_path = review_dir / "REVIEW_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Created {manifest_path}")


def main() -> None:
    """Main entry point."""
    version = get_version()
    commit_sha = get_commit_sha()
    review_dir = Path(__file__).resolve().parent.parent / "docs" / "review" / version

    print(f"Building review package for {version}")
    print(f"  Commit: {commit_sha}")
    print(f"  Review dir: {review_dir}")

    # Create directories
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "screenshots").mkdir(exist_ok=True)
    (review_dir / "sample_outputs").mkdir(exist_ok=True)
    (review_dir / "logs").mkdir(exist_ok=True)

    # Run tests
    test_results = run_tests_and_collect_results(review_dir)

    # Generate manifest
    generate_manifest(version, commit_sha, review_dir, test_results)

    # Copy test results if available
    html_report = Path.cwd() / "test-report.html"
    if html_report.exists():
        shutil.copy(html_report, review_dir / "test_report.html")

    print(f"\nReview package built at: {review_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
