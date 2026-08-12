"""M9 workflow state controller — tracks M8/M9 completion and invalidation.

Pure logic layer (no Qt dependency).  The workflow panel in the UI
reads state from this controller and renders accordingly.

When upstream data changes, downstream steps are automatically marked
STALE so the UI can warn the user that results must be regenerated.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, ClassVar

_logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    NOT_STARTED = "not_started"
    HAS_ISSUES = "has_issues"  # data present but validation problems exist
    READY = "ready"  # can be executed
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STALE = "stale"  # upstream changed — results invalid


class WorkflowStep:
    """One dependency-aware step in the M8/M9 pipeline."""

    def __init__(self, step_id: str, name: str, description: str = "", depends_on: list[str] | None = None):
        self.step_id = step_id
        self.name = name
        self.description = description
        self.depends_on: list[str] = depends_on or []
        self._status = StepStatus.NOT_STARTED
        self._completed_at: datetime | None = None
        self._metadata: dict[str, Any] = {}

    @property
    def status(self) -> StepStatus:
        return self._status

    def set_status(self, status: StepStatus) -> None:
        self._status = status
        if status == StepStatus.COMPLETED:
            self._completed_at = datetime.now(UTC)

    def invalidate(self) -> None:
        if self._status == StepStatus.COMPLETED:
            self._status = StepStatus.STALE

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "status": self._status.value,
            "depends_on": self.depends_on,
            "completed_at": self._completed_at.isoformat() if self._completed_at else None,
            "metadata": self._metadata,
        }


class WorkflowController:
    """Manages workflow step state and dependency invalidation.

    Usage:
        wf = WorkflowController()
        wf.add_step("import", "Import Data", depends_on=[])
        wf.add_step("clean", "Data Cleaning", depends_on=["import"])
        ...
        wf.complete_step("import")
        wf.complete_step("clean")
        # Later, user re-imports:
        wf.invalidate_from("import")  # → clean, holdout, domains, joint_sets all STALE
    """

    # M9 extends the seven M8 steps. The legacy identifier ``import``
    # is retained so v0.7.0 workflow state can be restored without renumbering.
    DEFAULT_STEPS: ClassVar[list[tuple[str, str, str]]] = [
        ("import", "1. 钻孔数据库", "Maintain independently imported borehole tables"),
        ("clean", "2. 数据质量", "Review formal, excluded, and pending records"),
        ("holdout", "3. Validation Holdout", "Lock calibration and validation boreholes"),
        ("domains", "4. 钻孔结构域", "Define borehole structural-domain intervals"),
        ("joint_sets", "5. 节理组", "Identify joint sets using calibration fractures only"),
        ("bounds", "6. 模型边界", "Define and validate the voxel analysis domain"),
        ("voxel_grid", "7. 体素网格预览与确认", "Preview and confirm voxel and DFN generation domains"),
        ("density", "8. Fracture Density Model", "Compute calibration P10/P32 and select a spatial density model"),
        ("size", "9. Fracture Size Distribution", "Fit measured sizes or define explicit prior assumptions"),
        ("parameter_field", "10. First Voxel Parameter Field", "Build the traceable input parameter voxel field"),
        ("validation", "11. Validation", "Evaluate predictions using held-out boreholes only"),
    ]
    DEPENDENCIES: ClassVar[dict[str, list[str]]] = {
        "import": [],
        "clean": ["import"],
        "holdout": ["clean"],
        "domains": ["clean"],
        "joint_sets": ["clean", "holdout"],
        "bounds": ["import"],
        "voxel_grid": ["bounds"],
        "density": ["clean", "holdout", "domains", "joint_sets"],
        "size": ["density"],
        "parameter_field": ["density", "size", "voxel_grid"],
        "validation": ["parameter_field"],
    }

    def __init__(self):
        self._steps: dict[str, WorkflowStep] = {}
        self._enabled_steps: set[str] = set()
        self._enabled_steps.update(step_id for step_id, _, _ in self.DEFAULT_STEPS)
        self._init_default_steps()
        self._change_listeners: list[Callable] = []

    def _init_default_steps(self) -> None:
        for step_id, name, desc in self.DEFAULT_STEPS:
            self._steps[step_id] = WorkflowStep(
                step_id=step_id,
                name=name,
                description=desc,
                depends_on=list(self.DEPENDENCIES[step_id]),
            )

    # ── Step management ─────────────────────────────────────────────────

    def add_step(self, step_id: str, name: str, description: str = "", depends_on: list[str] | None = None) -> None:
        self._steps[step_id] = WorkflowStep(
            step_id=step_id,
            name=name,
            description=description,
            depends_on=depends_on or [],
        )

    def enable_step(self, step_id: str) -> None:
        self._enabled_steps.add(step_id)

    def is_enabled(self, step_id: str) -> bool:
        return step_id in self._enabled_steps

    def get_step(self, step_id: str) -> WorkflowStep | None:
        return self._steps.get(step_id)

    def get_steps(self) -> list[WorkflowStep]:
        return [self._steps[k] for k, _, _ in self.DEFAULT_STEPS if k in self._steps]

    def get_enabled_steps(self) -> list[WorkflowStep]:
        return [s for s in self.get_steps() if self.is_enabled(s.step_id)]

    # ── Status transitions ──────────────────────────────────────────────

    def set_step_status(self, step_id: str, status: StepStatus) -> None:
        step = self._steps.get(step_id)
        if step is None:
            return
        old = step.status
        step.set_status(status)
        if old != status:
            self._notify_listeners()

    def complete_step(self, step_id: str) -> None:
        self.set_step_status(step_id, StepStatus.COMPLETED)

    def mark_issues(self, step_id: str) -> None:
        self.set_step_status(step_id, StepStatus.HAS_ISSUES)

    def mark_ready(self, step_id: str) -> None:
        self.set_step_status(step_id, StepStatus.READY)

    # ── Invalidation ────────────────────────────────────────────────────

    def invalidate_from(self, step_id: str) -> None:
        """Mark this step and all downstream steps as STALE."""
        step = self._steps.get(step_id)
        if step is None:
            return
        step.invalidate()
        # Find all steps that depend on this one (transitively)
        changed = True
        while changed:
            changed = False
            for s in self._steps.values():
                if s.status == StepStatus.COMPLETED:
                    for dep in s.depends_on:
                        dep_step = self._steps.get(dep)
                        if dep_step and dep_step.status == StepStatus.STALE:
                            s.invalidate()
                            changed = True
        self._notify_listeners()

    def invalidate_all_after(self, step_id: str) -> None:
        """Invalidate everything downstream from step_id."""
        found = False
        for sid, name, desc in self.DEFAULT_STEPS:
            if sid == step_id:
                found = True
                continue
            if found:
                step = self._steps.get(sid)
                if step:
                    step.invalidate()
        self._notify_listeners()

    def invalidate_dependents(self, step_id: str) -> None:
        """Invalidate completed transitive dependents without staling the source."""
        affected = {step_id}
        changed = True
        while changed:
            changed = False
            for candidate_id, candidate in self._steps.items():
                if candidate_id not in affected and any(dependency in affected for dependency in candidate.depends_on):
                    affected.add(candidate_id)
                    changed = True
        for candidate_id in affected - {step_id}:
            self._steps[candidate_id].invalidate()
        self._notify_listeners()

    def invalidate_steps(self, step_ids: list[str]) -> None:
        """Invalidate only the explicitly supplied completed result steps."""
        for step_id in step_ids:
            step = self._steps.get(step_id)
            if step is not None:
                step.invalidate()
        self._notify_listeners()

    # ── Query ───────────────────────────────────────────────────────────

    def is_step_done(self, step_id: str) -> bool:
        step = self._steps.get(step_id)
        return step is not None and step.status == StepStatus.COMPLETED

    def has_stale_results(self) -> bool:
        return any(s.status == StepStatus.STALE for s in self._steps.values())

    def get_active_issues(self) -> list[WorkflowStep]:
        return [s for s in self._steps.values() if s.status == StepStatus.HAS_ISSUES]

    # ── Listeners ───────────────────────────────────────────────────────

    def add_change_listener(self, callback: Callable) -> None:
        self._change_listeners.append(callback)

    def _notify_listeners(self) -> None:
        for cb in self._change_listeners:
            try:
                cb()
            except Exception:
                _logger.warning("Workflow change listener raised an exception", exc_info=True)

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "steps": {sid: s.to_dict() for sid, s in self._steps.items()},
            "enabled_steps": sorted(self._enabled_steps),
        }

    def from_dict(self, data: dict) -> None:
        for sid, sdata in data.get("steps", {}).items():
            step = self._steps.get(sid)
            if step:
                step._status = StepStatus(sdata.get("status", "not_started"))
                if sdata.get("completed_at"):
                    step._completed_at = datetime.fromisoformat(sdata["completed_at"])
                step._metadata = sdata.get("metadata", {})
        self._enabled_steps = set(data.get("enabled_steps", []))
        self._enabled_steps.update(step_id for step_id, _, _ in self.DEFAULT_STEPS)
