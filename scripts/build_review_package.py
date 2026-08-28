#!/usr/bin/env python
"""Build an M10 review package from test artifacts already produced by CI."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile


VERSION_PATTERN = re.compile(r"^(v\d+\.\d+\.\d+-M(?P<milestone>\d+))$")
REQUIRED_RESULTS = (
    "test_results_core.json",
    "test_results_gui.json",
    "junit-core.xml",
    "junit-gui.xml",
    "coverage.xml",
)
BENCHMARK_STEM = "m10_multiscale_closure_2026-08-27"
SOURCE_DIRECTORIES = ("src", "tests", "scripts", "benchmarks", "docs", "examples", "resources", "sample_data")
SOURCE_ROOT_FILES = ("LICENSE", "conftest.py", ".gitignore")
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "coverage_html",
}
EXCLUDED_NAMES = {".coverage", "coverage.xml", "test_results_core.json", "test_results_gui.json", "junit-core.xml", "junit-gui.xml"}
EXCLUDED_SUFFIXES = {".dfnproj", ".dfncs", ".pyc", ".pyo", ".pem", ".key"}


def parse_arguments() -> argparse.Namespace:
    """Parse review-package command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="Exact tag-shaped version, for example v0.10.0-M10")
    parser.add_argument("--results-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def get_version(explicit_version: str | None = None, *, repo_root: Path | None = None) -> str:
    """Resolve an exact release version or fail instead of emitting an unknown package.

    Resolution order is ``--version``, ``REVIEW_VERSION``, ``GITHUB_REF_NAME``,
    then an exact release tag pointing at HEAD in a clean worktree. A nearest historical
    tag is never used, and a dirty worktree requires an explicit version.
    """
    repo_root = repo_root or Path(__file__).resolve().parent.parent
    candidates = (
        explicit_version,
        os.environ.get("REVIEW_VERSION"),
        os.environ.get("GITHUB_REF_NAME"),
    )
    for candidate in candidates:
        if candidate:
            if not VERSION_PATTERN.fullmatch(candidate):
                raise RuntimeError(f"Invalid review version {candidate!r}; expected v<major>.<minor>.<patch>-M<number>")
            return candidate
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if status.returncode or status.stdout.strip():
        raise RuntimeError(
            "Cannot infer a review version from a tagged HEAD while the worktree has uncommitted changes. "
            "Use --version or REVIEW_VERSION/GITHUB_REF_NAME explicitly."
        )
    result = subprocess.run(
        ["git", "tag", "--points-at", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    exact_tags = sorted(tag.strip() for tag in result.stdout.splitlines() if VERSION_PATTERN.fullmatch(tag.strip()))
    if len(exact_tags) == 1:
        return exact_tags[0]
    raise RuntimeError(
        "Cannot determine an exact review version. Use --version or REVIEW_VERSION/GITHUB_REF_NAME; "
        "no formal package was generated."
    )


def milestone_from_version(version: str) -> str:
    """Return the milestone parsed from a validated version tag."""
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"Invalid review version: {version}")
    return f"M{match.group('milestone')}"


def get_commit_sha(repo_root: Path) -> str:
    """Return the CI commit or the exact local HEAD, failing when unavailable."""
    sha = os.environ.get("GITHUB_SHA", "").strip()
    if sha:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            raise RuntimeError(f"Invalid GITHUB_SHA: {sha!r}")
        return sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("Cannot determine commit SHA")
    return result.stdout.strip()


def parse_json_results(path: Path) -> dict[str, int]:
    """Parse one required pytest-json-report artifact and reject failures."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot parse required test result {path}: {exc}") from exc
    summary = data.get("summary", {})
    counts = {
        "total": int(summary.get("total", 0)),
        "passed": int(summary.get("passed", 0)),
        "failed": int(summary.get("failed", 0)) + int(summary.get("error", 0)),
        "skipped": int(summary.get("skipped", 0)) + int(summary.get("xfailed", 0)),
    }
    accounted = counts["passed"] + counts["failed"] + counts["skipped"]
    if counts["total"] <= 0 or accounted != counts["total"]:
        raise RuntimeError(f"Inconsistent pytest JSON counts in {path}: total={counts['total']}, accounted={accounted}")
    if counts["failed"]:
        raise RuntimeError(f"Tests failed in {path}: {counts['failed']}")
    return counts


def parse_junit_results(path: Path) -> dict[str, int]:
    """Parse a required JUnit artifact for independent count validation."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError(f"Cannot parse required JUnit result {path}: {exc}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    total = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failed = sum(int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return {"total": total, "passed": total - failed - skipped, "failed": failed, "skipped": skipped}


def validate_junit_scope(path: Path, required_prefixes: tuple[str, ...], forbidden_prefixes: tuple[str, ...]) -> None:
    """Ensure a JUnit artifact covers its documented test-suite scope."""
    root = ET.parse(path).getroot()
    classnames = [case.attrib.get("classname", "") for case in root.findall(".//testcase")]
    missing = [prefix for prefix in required_prefixes if not any(name.startswith(prefix) for name in classnames)]
    forbidden = sorted({name for name in classnames if name.startswith(forbidden_prefixes)})
    if missing or forbidden or not classnames:
        raise RuntimeError(
            f"JUnit scope validation failed for {path}: missing={missing}, forbidden={forbidden[:5]}"
        )


def parse_coverage(path: Path) -> float:
    """Return coverage percent from a required Cobertura XML artifact."""
    try:
        root = ET.parse(path).getroot()
        return round(float(root.attrib["line-rate"]) * 100.0, 2)
    except (OSError, ET.ParseError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse required coverage result {path}: {exc}") from exc


def validate_results(results_dir: Path) -> tuple[dict[str, dict[str, int]], float]:
    """Validate all CI artifacts and return core/GUI/combined statistics."""
    missing = [name for name in REQUIRED_RESULTS if not (results_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required CI result files: {', '.join(missing)}")
    core = parse_json_results(results_dir / "test_results_core.json")
    gui = parse_json_results(results_dir / "test_results_gui.json")
    for name, counts, junit_name in (
        ("core", core, "junit-core.xml"),
        ("gui", gui, "junit-gui.xml"),
    ):
        junit = parse_junit_results(results_dir / junit_name)
        if counts != junit:
            raise RuntimeError(f"{name} JSON/JUnit statistics disagree: JSON={counts}, JUnit={junit}")
    validate_junit_scope(
        results_dir / "junit-core.xml",
        ("tests.unit.", "tests.integration.", "tests.scientific.", "tests.end_to_end."),
        ("tests.gui.",),
    )
    validate_junit_scope(results_dir / "junit-gui.xml", ("tests.gui.",), ("tests.unit.", "tests.integration."))
    combined = {key: core[key] + gui[key] for key in core}
    coverage = parse_coverage(results_dir / "coverage.xml")
    if coverage < 80.0:
        raise RuntimeError(f"Coverage {coverage:.2f}% is below the required 80%")
    return {"core": core, "gui": gui, "combined": combined}, coverage


def changed_files(repo_root: Path, commit_sha: str) -> list[str]:
    """Return files recorded by the reviewed commit."""
    result = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_sha],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot inspect reviewed commit {commit_sha}")
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def copy_file(source: Path, target: Path) -> None:
    """Copy one artifact with parent creation."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


def build_review_directory(
    repo_root: Path,
    results_dir: Path,
    output_root: Path,
    version: str,
    commit_sha: str,
    statistics: dict[str, dict[str, int]],
    coverage: float,
) -> Path:
    """Create the complete review directory from validated existing artifacts."""
    milestone = milestone_from_version(version)
    review_dir = output_root / version
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_RESULTS:
        copy_file(results_dir / name, review_dir / name)
    benchmark_dir = repo_root / "benchmarks" / "results"
    benchmark_json = benchmark_dir / f"{BENCHMARK_STEM}.json"
    for suffix in (".json", ".md"):
        source = benchmark_dir / f"{BENCHMARK_STEM}{suffix}"
        if not source.is_file():
            raise RuntimeError(f"Missing required M10 benchmark report: {source}")
        copy_file(source, review_dir / "benchmarks" / source.name)
    try:
        benchmark_data = json.loads(benchmark_json.read_text(encoding="utf-8"))
        benchmark_cases = benchmark_data["cases"]
        large_case = max(benchmark_cases, key=lambda case: int(case["actual_count"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot audit required M10 benchmark report {benchmark_json}: {exc}") from exc
    if not benchmark_cases or int(large_case["actual_count"]) < 700_000:
        raise RuntimeError("Required M10 benchmark report does not contain an audited 700k+ fracture case")

    manifest = {
        "project_name": "DFN Cave Studio",
        "version": version,
        "tag": version,
        "milestone": milestone,
        "commit_sha": commit_sha,
        "created_at": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "generator_version": "m10-multiscale-1",
        "geometry_format": "columnar-multiscale-v1",
        "threshold_method": "area-weighted-cdf-1",
        "tests": statistics,
        "coverage_percent": coverage,
        "benchmarks": {
            "report_json": f"benchmarks/{BENCHMARK_STEM}.json",
            "report_markdown": f"benchmarks/{BENCHMARK_STEM}.md",
            "source_project": benchmark_data.get("source_project"),
            "generated_at": benchmark_data.get("generated_at"),
            "large_case_actual_fractures": int(large_case["actual_count"]),
            "large_case_elapsed_seconds": float(large_case["generation"]["elapsed_seconds"]),
            "large_case_final_array_bytes": int(large_case["final_array_bytes"]),
            "large_case_estimate_error_percent": float(large_case["final_estimate_error_percent"]),
        },
        "completed_features": [
            "M9-parameter-field driven conditional explicit DFN generation",
            "columnar authoritative center/normal/radius geometry",
            "SMALL/MEDIUM/LARGE target-P32 partition and P32_subgrid",
            "p32_unresolved_orientation audit budget",
            "seeded multi-realization generation, LOD display, persistence and generic export",
        ],
        "m11_not_implemented": [
            "exact fracture-voxel intersection and second voxelization",
            "fracture-fracture connectivity and percolation",
            "block cutting and fragmentation",
            "formal 3DEC/PFC and machine-learning export",
        ],
        "known_issues": ["Historical repository Ruff F/E9 baseline contains 177 findings; it is not newly passing."],
        "changed_files": changed_files(repo_root, commit_sha),
        "source_archive": f"dfn-cave-studio-{version}-source.zip",
        "review_archive": f"dfn-cave-studio-{version}-review.zip",
    }
    (review_dir / "REVIEW_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (review_dir / "TEST_SUMMARY.json").write_text(
        json.dumps({"tests": statistics, "coverage_percent": coverage}, indent=2), encoding="utf-8"
    )
    report = (
        f"# DFN Cave Studio {version} review\n\n"
        f"Commit: `{commit_sha}`  \nMilestone: `{milestone}`\n\n"
        f"Core: {statistics['core']['passed']}/{statistics['core']['total']} passed.  \n"
        f"GUI: {statistics['gui']['passed']}/{statistics['gui']['total']} passed.  \n"
        f"Combined: {statistics['combined']['passed']}/{statistics['combined']['total']} passed.  \n"
        f"Coverage: {coverage:.2f}%.\n\n"
        "M10 is complete and ready for release review. M11 scientific functionality listed in the manifest is not implemented.\n"
    )
    (review_dir / "MILESTONE_REPORT.md").write_text(report, encoding="utf-8")
    return review_dir


def include_source_file(path: Path, repo_root: Path) -> bool:
    """Return whether a source-package candidate is safe and in scope."""
    relative = path.relative_to(repo_root)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    lowered = path.name.lower()
    if lowered.startswith(".env") or "credential" in lowered or "secret" in lowered:
        return False
    if len(relative.parts) >= 2 and relative.parts[:2] == ("docs", "review"):
        return False
    return path.is_file()


def source_files(repo_root: Path) -> list[Path]:
    """Collect the documented complete source-archive inputs."""
    files: set[Path] = set()
    for directory in SOURCE_DIRECTORIES:
        root = repo_root / directory
        if not root.is_dir():
            raise RuntimeError(f"Missing required source-package directory: {root}")
        files.update(path for path in root.rglob("*") if include_source_file(path, repo_root))
    workflow_root = repo_root / ".github" / "workflows"
    if not workflow_root.is_dir():
        raise RuntimeError(f"Missing required source-package directory: {workflow_root}")
    files.update(path for path in workflow_root.rglob("*") if include_source_file(path, repo_root))
    files.update(
        path for path in repo_root.iterdir()
        if path.is_file() and (path.suffix.lower() in {".md", ".toml", ".txt"} or path.name in SOURCE_ROOT_FILES)
        and include_source_file(path, repo_root)
    )
    return sorted(files)


def build_archives(repo_root: Path, review_dir: Path, dist_dir: Path, version: str) -> tuple[Path, Path]:
    """Build deterministic review and filtered source ZIP archives."""
    dist_dir.mkdir(parents=True, exist_ok=True)
    review_zip = dist_dir / f"dfn-cave-studio-{version}-review.zip"
    source_zip = dist_dir / f"dfn-cave-studio-{version}-source.zip"
    with zipfile.ZipFile(review_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        prefix = f"dfn-cave-studio-{version}-review"
        for path in sorted(item for item in review_dir.rglob("*") if item.is_file()):
            archive.write(path, f"{prefix}/{path.relative_to(review_dir).as_posix()}")
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        prefix = f"dfn-cave-studio-{version}-source"
        for directory in (*SOURCE_DIRECTORIES, ".github/workflows"):
            archive.writestr(f"{prefix}/{directory}/", b"")
        for path in source_files(repo_root):
            archive.write(path, f"{prefix}/{path.relative_to(repo_root).as_posix()}")
    return review_zip, source_zip


def main() -> None:
    """Validate CI artifacts, create review metadata, and build both archives."""
    args = parse_arguments()
    repo_root = Path(__file__).resolve().parent.parent
    version = get_version(args.version, repo_root=repo_root)
    commit_sha = get_commit_sha(repo_root)
    results_dir = args.results_dir.resolve()
    output_root = (args.output_root or repo_root / "docs" / "review").resolve()
    statistics, coverage = validate_results(results_dir)
    review_dir = build_review_directory(
        repo_root, results_dir, output_root, version, commit_sha, statistics, coverage
    )
    review_zip, source_zip = build_archives(repo_root, review_dir, repo_root / "dist", version)
    print(f"Built {review_dir}")
    print(f"Tests: core={statistics['core']}, gui={statistics['gui']}, combined={statistics['combined']}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Archives: {review_zip.name}, {source_zip.name}")


if __name__ == "__main__":
    main()
