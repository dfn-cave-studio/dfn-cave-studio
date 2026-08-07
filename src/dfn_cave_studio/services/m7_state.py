"""M7 centralized Project state access — single source of truth for _m7_data.

All M7 dialogs read/write through these functions.  No dialog maintains
its own copy of holdout, workflow, or domain intervals independently.
"""

from __future__ import annotations
from typing import List, Optional, Any

from dfn_cave_studio.models.data_management import DomainInterval
from dfn_cave_studio.services.workflow_controller import WorkflowController

# ── Runtime type map ──────────────────────────────────────────────────
# project._m7_data keys and their expected runtime types:
#   "raw_surveys"          -> pd.DataFrame
#   "raw_fractures"        -> pd.DataFrame
#   "raw_rqd"              -> pd.DataFrame
#   "raw_domain_intervals" -> pd.DataFrame
#   "domain_intervals"     -> list[DomainInterval]
#   "holdout"              -> HoldoutService
#   "workflow"             -> WorkflowController
#   "quality_issues"       -> list[DataQualityIssue]


def _m7(project: Any) -> dict[str, Any]:
    """Get or create the _m7_data dict."""
    d = getattr(project, "_m7_data", None)
    if d is None:
        d = {}
        project._m7_data = d
    return d


def get_domain_intervals(project: Any) -> List[DomainInterval]:
    """Return formal domain_intervals list (always DomainInterval objects).

    Migrates list[dict] → list[DomainInterval] on first access if needed.
    """
    m7 = _m7(project)
    raw = m7.get("domain_intervals", [])
    if not raw:
        return []
    if isinstance(raw[0], DomainInterval):
        return raw
    # Migrate from list[dict]
    converted = []
    for item in raw:
        if isinstance(item, DomainInterval):
            converted.append(item)
        elif isinstance(item, dict):
            converted.append(DomainInterval(**item))
    m7["domain_intervals"] = converted
    return converted


def set_domain_intervals(project: Any, intervals: List[DomainInterval]) -> None:
    m7 = _m7(project)
    m7["domain_intervals"] = list(intervals)


def get_holdout(project: Any) -> Any:
    """Return HoldoutService from project or None."""
    m7 = _m7(project)
    return m7.get("holdout")


def set_holdout(project: Any, holdout: Any) -> None:
    m7 = _m7(project)
    m7["holdout"] = holdout


def get_workflow(project: Any) -> Optional[WorkflowController]:
    m7 = _m7(project)
    return m7.get("workflow")


def set_workflow(project: Any, wf: WorkflowController) -> None:
    m7 = _m7(project)
    m7["workflow"] = wf


def get_raw_surveys(project: Any) -> Any:
    """Return raw surveys DataFrame or None."""
    m7 = _m7(project)
    return m7.get("raw_surveys")


def set_raw_surveys(project: Any, surveys: Any) -> None:
    """Store the imported, unmodified survey table."""
    _m7(project)["raw_surveys"] = surveys


def get_raw_fractures(project: Any) -> Any:
    """Return the complete, unmodified fracture source table."""
    return _m7(project).get("raw_fractures")


def set_raw_fractures(project: Any, fractures: Any) -> None:
    """Store all fracture source rows, including rejected records."""
    _m7(project)["raw_fractures"] = fractures


def get_raw_rqd(project: Any) -> Any:
    m7 = _m7(project)
    return m7.get("raw_rqd")


def set_raw_rqd(project: Any, rqd: Any) -> None:
    """Store the imported, unmodified RQD table."""
    _m7(project)["raw_rqd"] = rqd


def get_raw_domain_intervals(project: Any) -> Any:
    m7 = _m7(project)
    return m7.get("raw_domain_intervals")


def set_raw_domain_intervals(project: Any, intervals: Any) -> None:
    """Store the imported, unmodified domain-interval table."""
    _m7(project)["raw_domain_intervals"] = intervals


def get_cleaned_rqd(project: Any) -> Any:
    """Return the cleaned RQD table, if cleaning has been committed."""
    return _m7(project).get("cleaned_rqd")


def set_cleaned_rqd(project: Any, rqd: Any) -> None:
    """Commit cleaned RQD without changing the raw imported table."""
    _m7(project)["cleaned_rqd"] = rqd


def get_excluded_records(project: Any) -> list[dict[str, Any]]:
    """Return traceable records excluded during cleaning."""
    return _m7(project).get("excluded_records", [])


def set_excluded_records(project: Any, records: list[dict[str, Any]]) -> None:
    """Commit traceable cleaning exclusions."""
    _m7(project)["excluded_records"] = list(records)


def get_quality_issues(project: Any) -> list[Any]:
    m7 = _m7(project)
    return m7.get("quality_issues", [])


def set_quality_issues(project: Any, issues: list[Any]) -> None:
    m7 = _m7(project)
    m7["quality_issues"] = list(issues)


def has_data(project: Any) -> bool:
    """Return True if any M7 data has been imported."""
    m7 = _m7(project)
    return bool(
        m7.get("raw_surveys") is not None
        or m7.get("raw_fractures") is not None
        or m7.get("raw_rqd") is not None
        or m7.get("raw_domain_intervals") is not None
        or m7.get("domain_intervals")
        or m7.get("holdout") is not None
    )
