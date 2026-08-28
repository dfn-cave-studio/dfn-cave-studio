"""
Project persistence — save, load, and manage DFN Cave Studio project files.

Project files:
  - .dfncs (JSON) — main project file (all data embedded)
  - Future: .dfncs.h5 (HDF5) for large voxel data

The ProjectStore handles:
  - Save / load / save-as
  - Auto-save with configurable interval
  - Project file locking (prevent concurrent edits)
  - Version tracking and backup
  - File format detection

References:
  - AGENTS.md (architecture rules)
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable, Dict

from dfn_cave_studio.models.project import Project

_logger = logging.getLogger(__name__)


# =============================================================================
# Project Store
# =============================================================================


class ProjectStoreError(Exception):
    """Base exception for project persistence errors."""


class ProjectFileCorrupted(ProjectStoreError):
    """Project file is corrupted or unreadable."""


class ProjectVersionTooNew(ProjectStoreError):
    """Project file was saved with a newer software version."""


class ProjectStore:
    """Manages project file I/O and state.

    Usage:
        store = ProjectStore()
        project = store.new_project("My Project")
        store.save("path/to/project.dfncs")
        # ... later ...
        project = store.open("path/to/project.dfncs")
    """

    BACKUP_EXTENSION = ".dfncs.bak"

    def __init__(self):
        self._current_project: Optional[Project] = None
        self._current_path: Optional[Path] = None
        self._last_save_time: Optional[datetime] = None
        self._dirty: bool = False
        self._auto_save_enabled: bool = True
        self._auto_save_interval: int = 300  # seconds
        self._last_auto_save: float = 0.0
        self._save_in_progress: bool = False
        self._large_autosave_limit_bytes: int = 64 * 1024**2

        # Callbacks
        self._on_saved: Optional[Callable[[Path], None]] = None
        self._on_loaded: Optional[Callable[[Project], None]] = None
        self._on_dirty_changed: Optional[Callable[[bool], None]] = None

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def current_project(self) -> Optional[Project]:
        return self._current_project

    @property
    def current_path(self) -> Optional[Path]:
        return self._current_path

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    @property
    def has_project(self) -> bool:
        return self._current_project is not None

    # ── Callbacks ─────────────────────────────────────────────────────────

    def set_on_saved(self, callback: Callable[[Path], None]) -> None:
        self._on_saved = callback

    def set_on_loaded(self, callback: Callable[[Project], None]) -> None:
        self._on_loaded = callback

    def set_on_dirty_changed(self, callback: Callable[[bool], None]) -> None:
        self._on_dirty_changed = callback

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            if self._on_dirty_changed:
                self._on_dirty_changed(True)

    def _mark_clean(self) -> None:
        if self._dirty:
            self._dirty = False
            if self._on_dirty_changed:
                self._on_dirty_changed(False)

    # ── Project Lifecycle ─────────────────────────────────────────────────

    def new_project(self, name: str = "Untitled Project") -> Project:
        """Create a new empty project.

        Args:
            name: Project name.

        Returns:
            New Project instance.
        """
        project = Project()
        project.metadata.name = name
        project.metadata.created_at = datetime.now(timezone.utc)
        project.metadata.modified_at = datetime.now(timezone.utc)

        self._current_project = project
        self._current_path = None
        self._last_save_time = None
        self._mark_dirty()

        return project

    def open(self, path: Path) -> Project:
        """Open a project from a file.

        Args:
            path: Path to .dfncs file.

        Returns:
            Loaded Project instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ProjectFileCorrupted: If the file can't be parsed.
            ProjectVersionTooNew: If the file version exceeds current.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ProjectFileCorrupted(f"Invalid JSON in {path}: {e}") from e
        except Exception as e:
            raise ProjectFileCorrupted(f"Failed to read {path}: {e}") from e

        # Check version compatibility
        schema_version = data.get("schema_version", 1)
        from dfn_cave_studio.models.project import CURRENT_PROJECT_VERSION

        if schema_version > CURRENT_PROJECT_VERSION:
            raise ProjectVersionTooNew(
                f"Project version {schema_version} is newer than current {CURRENT_PROJECT_VERSION}. "
                "Please upgrade DFN Cave Studio."
            )

        try:
            project = Project.from_dict(data)
        except Exception as e:
            raise ProjectFileCorrupted(f"Failed to parse project: {e}") from e

        self._current_project = project
        self._current_path = path
        self._last_save_time = datetime.now(timezone.utc)
        self._mark_clean()

        if self._on_loaded:
            self._on_loaded(project)

        return project

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the current project.

        Args:
            path: Target path. Uses current_path if None.

        Returns:
            The saved file path.

        Raises:
            ValueError: If no current project and no path provided.
        """
        if self._current_project is None:
            raise ValueError("No project to save")
        if self._save_in_progress:
            raise RuntimeError("A project save is already in progress")

        target = Path(path) if path else self._current_path
        if target is None:
            raise ValueError("No save path specified")

        # Atomic save: write to temp, then rename
        target = target.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.suffix.lower() not in {".dfncs", ".dfnproj"}:
            raise ValueError(
                f"Unsupported project extension '{target.suffix}'. " "Use .dfnproj (ZIP) or .dfncs (JSON)."
            )
        if (
            target.suffix.lower() == ".dfncs"
            and getattr(self._current_project, "m10_state", None) is not None
            and self._current_project.m10_state.realizations
        ):
            raise ValueError("M10 geometry requires the compressed .dfnproj format; legacy .dfncs would lose arrays")

        tmp_path = target.with_name(f"{target.name}.tmp")
        self._save_in_progress = True
        try:
            if target.suffix.lower() == ".dfnproj":
                from dfn_cave_studio.persistence.zip_project_store import (
                    ZipProjectStore,
                )

                ZipProjectStore().save(self._current_project, tmp_path)
            else:
                self._current_project.save_to_file(tmp_path)
            # Backup existing file if it exists
            if target.exists():
                backup_path = target.with_name(f"{target.name}.bak")
                shutil.copy2(target, backup_path)
            # Atomic rename
            shutil.move(str(tmp_path), str(target))
        except (OSError, ValueError, TypeError):
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        finally:
            self._save_in_progress = False

        self._current_path = target
        self._last_save_time = datetime.now(timezone.utc)
        self._last_auto_save = time.time()
        self._mark_clean()

        if self._on_saved:
            self._on_saved(target)

        return target

    def save_as(self, path: Path) -> Path:
        """Save project to a new location.

        Args:
            path: New file path.

        Returns:
            The saved file path.
        """
        if self._current_project is None:
            raise ValueError("No project to save")
        return self.save(path)

    def adopt_project(self, project: Project, path: Path | None = None) -> None:
        """Make an externally loaded project the current project and mark it clean.

        Use this when a project was loaded by another store (e.g. ZipProjectStore)
        and should become the active project in this ProjectStore instance.

        Args:
            project: Project instance to adopt.
            path: Optional file path to register as current path.
        """
        self._current_project = project
        if path is not None:
            self._current_path = Path(path)
        self._mark_clean()

    def register_save_path(self, path: Path) -> None:
        """Register a save path after external save (e.g. ZipProjectStore).

        Updates current_path and marks the project clean without re-saving.

        Args:
            path: The path where the project was saved.
        """
        self._current_path = Path(path)
        self._mark_clean()
        self._last_save_time = datetime.now(timezone.utc)
        self._last_auto_save = time.time()

    def mark_dirty(self) -> None:
        """Mark the current project as having unsaved changes.

        Public accessor for _mark_dirty — used by MainWindow when external
        operations (import, settings change) modify the project.
        """
        self._mark_dirty()

    def close(self) -> None:
        """Close the current project without saving."""
        self._current_project = None
        self._current_path = None
        self._mark_clean()

    # ── Auto-save ─────────────────────────────────────────────────────────

    def tick_auto_save(self) -> bool:
        """Check and trigger auto-save if needed.

        Call this periodically (e.g., from a QTimer).

        Returns:
            True if auto-save was performed.
        """
        if not self._auto_save_enabled:
            return False
        if not self._dirty:
            return False
        if self._current_path is None:
            return False  # Never saved yet, can't auto-save
        if self._save_in_progress:
            return False
        m10_state = getattr(self._current_project, "m10_state", None)
        if m10_state is not None:
            geometry_bytes = sum(
                array.nbytes
                for realization in m10_state.realizations
                for array in realization.geometry_arrays.values()
            )
            if geometry_bytes > self._large_autosave_limit_bytes:
                _logger.info(
                    "Skipping synchronous auto-save for large M10 state (%0.1f MiB)",
                    geometry_bytes / 1024**2,
                )
                self._last_auto_save = time.time()
                return False

        elapsed = time.time() - self._last_auto_save
        if elapsed < self._auto_save_interval:
            return False

        try:
            self.save()
            return True
        except (OSError, ValueError, TypeError):
            _logger.exception("Auto-save failed for %s", self._current_path)
            return False

    def configure_auto_save(self, enabled: bool = True, interval_seconds: int = 300) -> None:
        """Configure auto-save settings.

        Args:
            enabled: Enable/disable auto-save.
            interval_seconds: Interval in seconds.
        """
        self._auto_save_enabled = enabled
        self._auto_save_interval = interval_seconds


# =============================================================================
# Recent Projects Manager
# =============================================================================


class RecentProjectsManager:
    """Manages the list of recently opened projects.

    Stores the list in a JSON file in the user's config directory.
    """

    MAX_RECENT = 20

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path.home() / ".dfn-cave-studio"
        self._config_dir = Path(config_dir)
        self._recent_file = self._config_dir / "recent_projects.json"
        self._recent: list[Dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        """Load recent projects from disk."""
        if self._recent_file.exists():
            try:
                with open(self._recent_file, "r", encoding="utf-8") as f:
                    self._recent = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._recent = []

    def _save(self) -> None:
        """Save recent projects to disk."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with open(self._recent_file, "w", encoding="utf-8") as f:
            json.dump(self._recent, f, indent=2, ensure_ascii=False)

    def add(self, path: Path, name: Optional[str] = None) -> None:
        """Add or update a project in the recent list.

        Args:
            path: Project file path.
            name: Project name (from metadata).
        """
        path_str = str(path.resolve())
        name = name or path.stem

        # Remove existing entry
        self._recent = [r for r in self._recent if r.get("path") != path_str]

        # Add to front
        self._recent.insert(
            0,
            {
                "path": path_str,
                "name": name,
                "last_opened": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Trim
        self._recent = self._recent[: self.MAX_RECENT]

        self._save()

    def remove(self, path: Path) -> None:
        """Remove a project from the recent list."""
        path_str = str(path.resolve())
        self._recent = [r for r in self._recent if r.get("path") != path_str]
        self._save()

    def list(self) -> list[Dict[str, str]]:
        """Get the recent projects list.

        Returns:
            List of {path, name, last_opened} dicts.
        """
        # Filter out entries whose files no longer exist
        valid = []
        for entry in self._recent:
            if Path(entry["path"]).exists():
                valid.append(entry)
        self._recent = valid
        return valid

    def clear(self) -> None:
        """Clear all recent projects."""
        self._recent = []
        self._save()

    def most_recent(self) -> Optional[Dict[str, str]]:
        """Get the most recent project, if any."""
        items = self.list()
        return items[0] if items else None
