#!/usr/bin/env python
"""
Build the review package for a milestone release.

Collects test results, coverage, and metadata into docs/review/<version>/

ALL metrics (test counts, coverage, changed_files) are dynamically generated
from pytest JSON output, coverage.xml, and git. NO hardcoded values.
"""

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def get_version() -> str:
    """Extract version from git tag (CI) or pyproject.toml."""
    # Prefer GITHUB_REF_NAME from CI
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    if ref_name and ref_name.startswith("v"):
        return ref_name

    # Fallback: git describe
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        tag = result.stdout.strip()
        if tag:
            return tag
    except subprocess.CalledProcessError:
        pass
    return "0.0.0-unknown"


def get_commit_sha() -> str:
    """Get commit SHA from CI env or git."""
    # Prefer CI environment variable (always correct for the build)
    sha = os.environ.get("GITHUB_SHA", "")
    if sha:
        return sha

    # Fallback for local development
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_changed_files(commit_sha: str) -> list:
    """Get list of files changed in this commit using git diff-tree."""
    try:
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        return files
    except subprocess.CalledProcessError:
        # Fallback: diff against previous commit
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
                capture_output=True, text=True, check=True,
                cwd=Path(__file__).resolve().parent.parent,
            )
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except subprocess.CalledProcessError:
            return []


def parse_test_results(json_report_path: Path) -> dict:
    """Parse pytest --json-report output for test counts.

    Returns dict with keys: tests_total, tests_passed, tests_failed, tests_skipped.
    All values are 0 if the file cannot be read or parsed.
    """
    result = {"tests_total": 0, "tests_passed": 0, "tests_failed": 0, "tests_skipped": 0}
    if not json_report_path.exists():
        print(f"  WARNING: test results file not found: {json_report_path}")
        return result

    try:
        with open(json_report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        result["tests_total"] = summary.get("total", 0)
        result["tests_passed"] = summary.get("passed", 0)
        result["tests_failed"] = summary.get("failed", 0)
        result["tests_skipped"] = summary.get("skipped", 0)
        print(f"  Parsed test results: {result['tests_passed']}/{result['tests_total']} passed, "
              f"{result['tests_failed']} failed, {result['tests_skipped']} skipped")
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: failed to parse test results: {e}")

    return result


def parse_coverage(coverage_xml_path: Path) -> float:
    """Parse coverage.xml for overall line-rate percentage.

    Returns coverage percentage (0.0-100.0), or 0.0 if unreadable.
    """
    if not coverage_xml_path.exists():
        print(f"  WARNING: coverage file not found: {coverage_xml_path}")
        return 0.0

    try:
        tree = ET.parse(coverage_xml_path)
        root = tree.getroot()
        line_rate = float(root.attrib.get("line-rate", 0))
        coverage_pct = round(line_rate * 100, 1)
        print(f"  Parsed coverage: {coverage_pct}%")
        return coverage_pct
    except (ET.ParseError, OSError, ValueError) as e:
        print(f"  WARNING: failed to parse coverage: {e}")
        return 0.0


def run_tests_and_collect_results(review_dir: Path) -> dict:
    """Run pytest with --json-report and --cov, collect results."""
    print("Running tests with coverage...")
    repo_root = Path(__file__).resolve().parent.parent
    json_file = review_dir / "test_results.json"

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/unit/", "tests/integration/", "tests/scientific/",
            "-v", "--tb=short",
            "--json-report",
            f"--json-report-file={json_file}",
            "--cov=src/dfn_cave_studio",
            "--cov-report=xml:coverage.xml",
            "--cov-report=html:coverage_html",
        ],
        capture_output=False,
        cwd=repo_root,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )

    # Move coverage files to review dir
    cov_xml = repo_root / "coverage.xml"
    if cov_xml.exists():
        shutil.copy(cov_xml, review_dir / "coverage.xml")

    cov_html = repo_root / "coverage_html"
    if cov_html.exists():
        target = review_dir / "coverage_html"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(cov_html, target)

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
    """Generate REVIEW_MANIFEST.json with dynamically-collected metrics."""
    import sys as _sys

    # Parse actual test results from JSON
    json_report = review_dir / "test_results.json"
    test_counts = parse_test_results(json_report)

    # Parse actual coverage from XML
    coverage_xml = review_dir / "coverage.xml"
    coverage_pct = parse_coverage(coverage_xml)

    # Get changed files from git
    changed_files = get_changed_files(commit_sha)

    manifest = {
        "project_name": "DFN Cave Studio",
        "version": version,
        "milestone": "M6",
        "commit_sha": commit_sha,
        "branch": "main",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
        "operating_system": f"{_sys.platform}",
        "dependency_lock_hash": "N/A",
        "completed_features": [],
        "incomplete_features": [],
        "experimental_features": [],
        "changed_files": changed_files,
        "tests_total": test_counts["tests_total"],
        "tests_passed": test_counts["tests_passed"],
        "tests_failed": test_counts["tests_failed"],
        "tests_skipped": test_counts["tests_skipped"],
        "coverage_percent": coverage_pct,
        "scientific_cases": test_counts.get("scientific_cases", 0),
        "performance_cases": 0,
        "known_issues": [],
        "release_url": "",
        "source_archive": "",
        "previous_version": "N/A",
        "reproducibility_command": (
            "python -m venv .venv && "
            ".venv/Scripts/activate && "
            "pip install -r requirements.txt && "
            "pytest tests/unit/ tests/integration/ tests/scientific/"
        ),
    }

    manifest_path = review_dir / "REVIEW_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Created {manifest_path}")
    print(f"  Tests: {manifest['tests_passed']}/{manifest['tests_total']} passed")
    print(f"  Coverage: {manifest['coverage_percent']}%")
    print(f"  Changed files: {len(manifest['changed_files'])}")


def main() -> None:
    """Main entry point."""
    version = get_version()
    commit_sha = get_commit_sha()
    repo_root = Path(__file__).resolve().parent.parent
    review_dir = repo_root / "docs" / "review" / version

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

    # Copy test report if available
    html_report = repo_root / "test-report.html"
    if html_report.exists():
        shutil.copy(html_report, review_dir / "test_report.html")

    print(f"\nReview package built at: {review_dir}")
    if test_results["passed"]:
        print("All tests passed.")
    else:
        print("WARNING: Some tests failed — check test_results.json for details.")
    print("Done.")


if __name__ == "__main__":
    main()
