"""M7 workflow state controller — tracks step completion and invalidation.

Pure logic layer (no Qt dependency).  The workflow panel in the UI
reads state from this controller and renders accordingly.

When upstream data changes, downstream steps are automatically marked
STALE so the UI can warn the user that results must be regenerated.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import List, Set, Callable, Dict, Any, Optional
from datetime import datetime, timezone

_logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    NOT_STARTED = "not_started"
    HAS_ISSUES = "has_issues"      # data present but validation problems exist
    READY = "ready"                 # can be executed
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STALE = "stale"                 # upstream changed — results invalid


class WorkflowStep:
    """One step in the M7 pipeline."""

    def __init__(self, step_id: str, name: str, description: str = "",
                 depends_on: Optional[List[str]] = None):
        self.step_id = step_id
        self.name = name
        self.description = description
        self.depends_on: List[str] = depends_on or []
        self._status = StepStatus.NOT_STARTED
        self._completed_at: Optional[datetime] = None
        self._metadata: Dict[str, Any] = {}

    @property
    def status(self) -> StepStatus:
        return self._status

    def set_status(self, status: StepStatus) -> None:
        self._status = status
        if status == StepStatus.COMPLETED:
            self._completed_at = datetime.now(timezone.utc)

    def invalidate(self) -> None:
        if self._status == StepStatus.COMPLETED:
            self._status = StepStatus.STALE

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id, "name": self.name,
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

    DEFAULT_STEPS = [
        ("import",      "1. 导入数据",     "Import borehole data from CSV/XLSX files"),
        ("clean",       "2. 数据清洗",     "Check data quality and fix issues"),
        ("holdout",     "3. 划分验证钻孔",  "Split boreholes into calibration/validation"),
        ("domains",     "4. 构造结构域",    "Define structural domains and intervals"),
        ("joint_sets",  "5. 识别节理组",    "Identify joint sets from fracture observations"),
        ("density",     "6. 裂隙密度",      "[后续] P10/P32 spatial estimation"),
        ("size_dist",   "7. 尺寸分布",      "[后续] Fracture size distribution"),
        ("voxel_1",     "8. 第一次体素化",  "[后续] Parameter field voxelization"),
        ("dfn_gen",     "9. 生成DFN",       "[后续] Stochastic DFN generation"),
        ("voxel_2",     "10. 第二次体素化", "[后续] Result field voxelization"),
        ("validation",  "11. 验证",         "[后续] Model validation"),
        ("export",      "12. 导出",         "[后续] Export to 3DEC/PFC/ML"),
    ]

    def __init__(self):
        self._steps: Dict[str, WorkflowStep] = {}
        self._enabled_steps: Set[str] = set()
        # Enable first 5 by default
        self._enabled_steps.update(["import", "clean", "holdout", "domains", "joint_sets"])
        self._init_default_steps()
        self._change_listeners: List[Callable] = []

    def _init_default_steps(self) -> None:
        deps: List[str] = []
        for step_id, name, desc in self.DEFAULT_STEPS:
            self._steps[step_id] = WorkflowStep(
                step_id=step_id, name=name, description=desc,
                depends_on=list(deps),
            )
            deps.append(step_id)

    # ── Step management ─────────────────────────────────────────────────

    def add_step(self, step_id: str, name: str, description: str = "",
                 depends_on: Optional[List[str]] = None) -> None:
        self._steps[step_id] = WorkflowStep(
            step_id=step_id, name=name, description=description,
            depends_on=depends_on or [],
        )

    def enable_step(self, step_id: str) -> None:
        self._enabled_steps.add(step_id)

    def is_enabled(self, step_id: str) -> bool:
        return step_id in self._enabled_steps or step_id in ("import", "clean", "holdout", "domains", "joint_sets")

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        return self._steps.get(step_id)

    def get_steps(self) -> List[WorkflowStep]:
        return [self._steps[k] for k, _, _ in self.DEFAULT_STEPS if k in self._steps]

    def get_enabled_steps(self) -> List[WorkflowStep]:
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
            for sid, s in self._steps.items():
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

    # ── Query ───────────────────────────────────────────────────────────

    def is_step_done(self, step_id: str) -> bool:
        step = self._steps.get(step_id)
        return step is not None and step.status == StepStatus.COMPLETED

    def has_stale_results(self) -> bool:
        return any(s.status == StepStatus.STALE for s in self._steps.values())

    def get_active_issues(self) -> List[WorkflowStep]:
        return [s for s in self._steps.values() if s.status == StepStatus.HAS_ISSUES]

    # ── Listeners ───────────────────────────────────────────────────────

    def add_change_listener(self, callback: Callable) -> None:
        self._change_listeners.append(callback)

    def _notify_listeners(self) -> None:
        for cb in self._change_listeners:
            try:
                cb()
            except Exception:
                _logger.warning("Workflow change listener raised an exception",
                                exc_info=True)

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "steps": {sid: s.to_dict() for sid, s in self._steps.items()},
            "enabled_steps": list(self._enabled_steps),
        }

    def from_dict(self, data: Dict) -> None:
        for sid, sdata in data.get("steps", {}).items():
            step = self._steps.get(sid)
            if step:
                step._status = StepStatus(sdata.get("status", "not_started"))
                if sdata.get("completed_at"):
                    step._completed_at = datetime.fromisoformat(sdata["completed_at"])
                step._metadata = sdata.get("metadata", {})
        self._enabled_steps = set(data.get("enabled_steps", []))
