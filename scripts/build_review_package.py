#!/usr/bin/env python
"""Build a release review package from independently produced CI artifacts."""

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


VERSION_PATTERN = re.compile(r"^(v\d+\.\d+\.\d+-M(?P<milestone>\d+(?:\.\d+)?))$")
WINDOWS_PYTHON_VERSIONS = ("3.12", "3.13")
WINDOWS_REQUIRED_RESULTS = (
    "test_results_core.json",
    "test_results_gui.json",
    "junit-core.xml",
    "junit-gui.xml",
    "coverage.xml",
)
REAL_VTK_REQUIRED_RESULTS = ("test_results_real_vtk.json", "junit-real-vtk.xml")
M11_BENCHMARK_STEMS = (
    "m11_second_voxelization",
    "m11_cloud_visualization",
    "m11_cancellation",
    "m11_second_voxelization_cancel_fix",
)
PHASE_NAME = "M11.1 Exact Second Voxelization"
BASELINE_COMMITS = {
    "main_baseline": "e066e6cedb05cac2b01caea1464ba86b6a29f930",
    "m11_features": "2feaa4db394863a64da791e33e4ebf58f2ed6305",
    "vtk_ci_isolation": "9bad673538b7fd852868e29965efda09db4bc069",
    "python_313_lifecycle": "b94f87e33c79f854e71047a0d555e0a546cb2306",
}
COMPLETED_SCOPE = [
    "Analytic circular disk-voxel exact area intersection",
    "Exact second voxelization",
    "P32_explicit_intersection + P32_subgrid",
    "Per-joint-set and all-set results",
    "Sparse persistence, reopen, and invalidation management",
    "M11 cloud, orthogonal/arbitrary sections, plane cutaway, and box cutaway",
    "M11 layer and scalar-bar lifecycle",
    "Chinese and English interface",
    "M10 configuration save validation and recovery protection",
    "M9/M11 opaque finite-value occlusion fixes",
    "Cooperative fast cancellation",
    "Windows Python 3.12/3.13 CI and Linux Mesa/Xvfb real VTK rendering tests",
]
INCOMPLETE_SCOPE = [
    "Fracture-fracture intersection graph",
    "Connected clusters and boundary-spanning paths",
    "Flow/percolation and preferential channels",
    "Block cutting and fragmentation statistics",
    "Mechanical properties",
    "Formal 3DEC/PFC export",
    "Kriging interpolation",
]
REQUIRED_SOURCE_DIRECTORIES = ("src", "tests", "scripts", "benchmarks", "docs", "examples", ".github/workflows")
OPTIONAL_SOURCE_DIRECTORIES = ("resources", "sample_data")
SOURCE_ROOT_FILES = ("LICENSE", "conftest.py", ".gitignore")
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox",
    ".claude", ".vscode", "build", "dist", "coverage_html", "htmlcov", "diagnostics", "crash", "crashes",
    "temp", "tmp",
}
EXCLUDED_NAMES = {
    ".coverage", "coverage.xml", "test_results_core.json", "test_results_gui.json", "test_results_real_vtk.json",
    "junit-core.xml", "junit-gui.xml", "junit-real-vtk.xml",
}
EXCLUDED_SUFFIXES = {
    ".dfnproj", ".dfncs", ".h5", ".hdf", ".hdf5", ".he5", ".pyc", ".pyo", ".pem", ".key",
    ".log", ".dmp", ".crash", ".core", ".tmp", ".bak",
}


def parse_arguments() -> argparse.Namespace:
    """Parse review-package command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="Exact tag-shaped version, for example v0.11.0-M11.1")
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
                raise RuntimeError(
                    f"Invalid review version {candidate!r}; expected "
                    "v<major>.<minor>.<patch>-M<number>[.<phase>]"
                )
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
        "collected": int(summary.get("collected", summary.get("total", 0))),
        "deselected": int(summary.get("deselected", 0)),
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


def _assert_json_matches_junit(label: str, counts: dict[str, int], junit_path: Path) -> None:
    """Reject a CI result when its JSON and JUnit selected-test counts differ."""
    junit = parse_junit_results(junit_path)
    selected = {key: counts[key] for key in ("total", "passed", "failed", "skipped")}
    if selected != junit:
        raise RuntimeError(f"{label} JSON/JUnit statistics disagree: JSON={selected}, JUnit={junit}")


def validate_results(results_dir: Path) -> dict[str, object]:
    """Validate each CI job independently without inventing a cross-job total."""
    windows: dict[str, object] = {}
    for version in WINDOWS_PYTHON_VERSIONS:
        job_dir = results_dir / f"python-{version}"
        missing = [name for name in WINDOWS_REQUIRED_RESULTS if not (job_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing Python {version} Windows CI results: {', '.join(missing)}")
        core = parse_json_results(job_dir / "test_results_core.json")
        gui = parse_json_results(job_dir / "test_results_gui.json")
        _assert_json_matches_junit(f"Python {version} core", core, job_dir / "junit-core.xml")
        _assert_json_matches_junit(f"Python {version} GUI", gui, job_dir / "junit-gui.xml")
        validate_junit_scope(
            job_dir / "junit-core.xml",
            ("tests.unit.", "tests.integration.", "tests.scientific.", "tests.end_to_end."),
            ("tests.gui.",),
        )
        validate_junit_scope(
            job_dir / "junit-gui.xml", ("tests.gui.",), ("tests.unit.", "tests.integration.")
        )
        coverage = parse_coverage(job_dir / "coverage.xml")
        if coverage < 80.0:
            raise RuntimeError(f"Python {version} coverage {coverage:.2f}% is below the required 80%")
        windows[version] = {
            "platform": "windows-latest",
            "python": version,
            "core": core,
            "gui": gui,
            "coverage_percent": coverage,
        }

    render_dir = results_dir / "real-vtk"
    missing = [name for name in REAL_VTK_REQUIRED_RESULTS if not (render_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing Linux real-VTK CI results: {', '.join(missing)}")
    real_vtk = parse_json_results(render_dir / "test_results_real_vtk.json")
    _assert_json_matches_junit("Linux real VTK", real_vtk, render_dir / "junit-real-vtk.xml")
    return {
        "windows": windows,
        "linux_real_vtk": {
            "platform": "ubuntu-latest + Mesa/Xvfb",
            "python": "3.12",
            "tests": real_vtk,
        },
        "counting_note": (
            "Counts are reported per independent CI job. Python matrix runs are not summed into a fictitious total."
        ),
    }


def changed_files(repo_root: Path, commit_sha: str) -> list[str]:
    """Return files changed since the previous published M10 release."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"v0.10.1-M10..{commit_sha}"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        return []
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
    test_jobs: dict[str, object],
) -> Path:
    """Create the complete review directory from validated existing artifacts."""
    milestone = milestone_from_version(version)
    review_dir = output_root / version
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    for version_name in WINDOWS_PYTHON_VERSIONS:
        source_dir = results_dir / f"python-{version_name}"
        target_dir = review_dir / "test-results" / f"windows-python-{version_name}"
        for name in WINDOWS_REQUIRED_RESULTS:
            copy_file(source_dir / name, target_dir / name)
    for name in REAL_VTK_REQUIRED_RESULTS:
        copy_file(results_dir / "real-vtk" / name, review_dir / "test-results" / "linux-real-vtk" / name)

    benchmark_payloads: dict[str, object] = {}
    for stem in M11_BENCHMARK_STEMS:
        for suffix in (".json", ".md"):
            source = repo_root / "benchmarks" / f"{stem}{suffix}"
            if not source.is_file():
                raise RuntimeError(f"Missing required M11 benchmark report: {source}")
            copy_file(source, review_dir / "benchmarks" / source.name)
        try:
            benchmark_payloads[stem] = json.loads(
                (repo_root / "benchmarks" / f"{stem}.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot audit M11 benchmark {stem}: {exc}") from exc

    second_voxelization = benchmark_payloads["m11_second_voxelization"]
    second_cases = second_voxelization.get("cases", [])
    if not second_cases or max(int(case["fracture_count"]) for case in second_cases) < 700_000:
        raise RuntimeError("M11 benchmark does not contain the audited 700k+ fracture case")

    source_status = source_directory_status(repo_root)
    manifest = {
        "project_name": "DFN Cave Studio",
        "version": version,
        "tag": version,
        "milestone": milestone,
        "phase_name": PHASE_NAME,
        "planned_tag": version,
        "release_status": "ready_for_release",
        "commit_sha": commit_sha,
        "baseline_commits": BASELINE_COMMITS,
        "created_at": datetime.now(UTC).isoformat(),
        "packager_python_version": sys.version,
        "project_schema_version": 5,
        "generator_version": "m10-multiscale-1",
        "geometry_format": "columnar-multiscale-v1",
        "threshold_method": "area-weighted-cdf-1",
        "m11_algorithm_version": "m11-second-voxelization-1",
        "m11_geometry_kernel": "analytic-circle-convex-polygon-1",
        "tests_by_ci_job": test_jobs,
        "benchmarks": {
            "reports": [f"benchmarks/{stem}.json" for stem in M11_BENCHMARK_STEMS],
            "second_voxelization": second_voxelization,
            "cloud_visualization": benchmark_payloads["m11_cloud_visualization"],
            "cancellation": benchmark_payloads["m11_cancellation"],
            "cancellation_comparison": benchmark_payloads["m11_second_voxelization_cancel_fix"],
        },
        "completed_scope": COMPLETED_SCOPE,
        "not_implemented_scope": INCOMPLETE_SCOPE,
        "scope_warning": "M11.1 is ready for release; its tag and GitHub Release do not yet exist, and the complete M11 milestone is not complete.",
        "known_issues": ["Historical repository Ruff F/E9 baseline contains 132 findings; it is not newly passing."],
        "source_package_directories": source_status,
        "changed_files": changed_files(repo_root, commit_sha),
        "source_archive": f"dfn-cave-studio-{version}-source.zip",
        "review_archive": f"dfn-cave-studio-{version}-review.zip",
    }
    (review_dir / "REVIEW_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (review_dir / "TEST_SUMMARY.json").write_text(
        json.dumps({"tests_by_ci_job": test_jobs}, indent=2), encoding="utf-8"
    )
    windows = test_jobs["windows"]
    vtk = test_jobs["linux_real_vtk"]
    report_lines = [
        f"# DFN Cave Studio {version} review",
        "",
        f"**Phase:** {PHASE_NAME}",
        f"**Commit baseline:** `{commit_sha}`",
        f"**Milestone identifier:** `{milestone}`",
        "**Status:** READY FOR RELEASE",
        "",
        "## Test jobs (not summed across duplicated Python matrix runs)",
        "",
        "| Job | Core | GUI | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for python_version in WINDOWS_PYTHON_VERSIONS:
        job = windows[python_version]
        report_lines.append(
            f"| Windows Python {python_version} | {job['core']['passed']} | {job['gui']['passed']} | "
            f"{job['coverage_percent']:.2f}% |"
        )
    report_lines.extend(
        [
            "",
            f"Linux Mesa/Xvfb real VTK: {vtk['tests']['passed']}/{vtk['tests']['total']} passed.",
            "",
            "## Completed M11.1 scope",
            "",
            *[f"- {item}" for item in COMPLETED_SCOPE],
            "",
            "## Not implemented in M11.1",
            "",
            *[f"- {item}" for item in INCOMPLETE_SCOPE],
            "",
            "M11.1 is ready for release as a bounded phase. Its tag and GitHub Release have not been created, and it does not claim completion of all M11 work.",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    (review_dir / "MILESTONE_REPORT.md").write_text(report, encoding="utf-8")
    python_312 = windows["3.12"]
    python_313 = windows["3.13"]
    vtk_tests = vtk["tests"]
    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{version} review</title>
<style>body{{font-family:Arial,sans-serif;max-width:960px;margin:2rem auto;line-height:1.5}}code{{background:#eee;padding:.1rem .3rem}}</style>
</head><body><h1>{version} &mdash; {PHASE_NAME}</h1>
<p><strong>Status:</strong> READY FOR RELEASE<br><strong>Planned tag:</strong> <code>{version}</code><br><strong>Commit:</strong> <code>{commit_sha[:7]}</code></p>
<p>Windows Python 3.12: {python_312['core']['passed']} core + {python_312['gui']['passed']} GUI, {python_312['coverage_percent']:.2f}% coverage.<br>
Windows Python 3.13: {python_313['core']['passed']} core + {python_313['gui']['passed']} GUI, {python_313['coverage_percent']:.2f}% coverage.<br>
Linux Mesa/Xvfb: {vtk_tests['passed']}/{vtk_tests['total']} real VTK tests passed.</p>
<p>This release publishes M11.1 only. Connectivity, percolation, block cutting, mechanics, formal 3DEC/PFC export, and Kriging remain planned.</p>
<h2>Review files</h2><ul>
<li><a href="MILESTONE_REPORT.md">MILESTONE_REPORT.md</a></li>
<li><a href="REVIEW_MANIFEST.json">REVIEW_MANIFEST.json</a></li>
<li><a href="TEST_SUMMARY.json">TEST_SUMMARY.json</a></li>
<li><a href="test-results/windows-python-3.12/coverage.xml">Python 3.12 coverage.xml</a></li>
<li><a href="test-results/windows-python-3.13/coverage.xml">Python 3.13 coverage.xml</a></li>
<li><a href="test-results/linux-real-vtk/junit-real-vtk.xml">Linux real-VTK JUnit</a></li>
<li><a href="benchmarks/m11_second_voxelization.md">M11.1 benchmark</a></li>
</ul><p>Planned tag: <code>{version}</code> &middot; <a href="../../index.html">Documentation home</a></p>
</body></html>
"""
    (review_dir / "index.html").write_text(index, encoding="utf-8")
    return review_dir


def include_source_file(path: Path, repo_root: Path) -> bool:
    """Return whether a source-package candidate is safe and in scope."""
    relative = path.relative_to(repo_root)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if any(part.lower().endswith(".zarr") for part in relative.parts):
        return False
    lowered = path.name.lower()
    if lowered.startswith(".env") or "credential" in lowered or "secret" in lowered:
        return False
    if len(relative.parts) >= 2 and relative.parts[:2] == ("docs", "review"):
        return False
    return path.is_file()


def source_directory_status(repo_root: Path) -> dict[str, list[str]]:
    """Validate required source directories and report available optional directories."""
    missing_required = [name for name in REQUIRED_SOURCE_DIRECTORIES if not (repo_root / name).is_dir()]
    if missing_required:
        missing = ", ".join(str(repo_root / name) for name in missing_required)
        raise RuntimeError(f"Missing required source-package directory: {missing}")
    included_optional = [name for name in OPTIONAL_SOURCE_DIRECTORIES if (repo_root / name).is_dir()]
    omitted_optional = [name for name in OPTIONAL_SOURCE_DIRECTORIES if name not in included_optional]
    return {
        "required": list(REQUIRED_SOURCE_DIRECTORIES),
        "included_optional": included_optional,
        "omitted_optional": omitted_optional,
    }


def source_files(repo_root: Path) -> list[Path]:
    """Collect the documented complete source-archive inputs."""
    status = source_directory_status(repo_root)
    files: set[Path] = set()
    for directory in (*REQUIRED_SOURCE_DIRECTORIES, *status["included_optional"]):
        root = repo_root / directory
        files.update(path for path in root.rglob("*") if include_source_file(path, repo_root))
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
        status = source_directory_status(repo_root)
        for directory in (*REQUIRED_SOURCE_DIRECTORIES, *status["included_optional"]):
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
    test_jobs = validate_results(results_dir)
    review_dir = build_review_directory(
        repo_root, results_dir, output_root, version, commit_sha, test_jobs
    )
    review_zip, source_zip = build_archives(repo_root, review_dir, repo_root / "dist", version)
    print(f"Built {review_dir}")
    for python_version, job in test_jobs["windows"].items():
        print(
            f"Windows Python {python_version}: core={job['core']}, gui={job['gui']}, "
            f"coverage={job['coverage_percent']:.2f}%"
        )
    print(f"Linux real VTK: {test_jobs['linux_real_vtk']['tests']}")
    print(f"Archives: {review_zip.name}, {source_zip.name}")


if __name__ == "__main__":
    main()
