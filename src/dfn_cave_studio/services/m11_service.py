"""Project service for transactional M11.1 exact second voxelization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dfn_cave_studio.models.m10 import M10Realization
from dfn_cave_studio.models.m11 import M11SecondVoxelizationResult
from dfn_cave_studio.voxel.second_voxelization import (
    M11SecondVoxelizer,
    SecondVoxelizationConfig,
    parameter_field_hash,
)


class M11SecondVoxelizationService:
    """Validate dependencies, compute results, and commit only complete batches."""

    def __init__(self, project: Any) -> None:
        self.project = project

    def compute(
        self,
        realization_id: str,
        *,
        config: SecondVoxelizationConfig | None = None,
        progress: Callable[[int, int, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        commit: bool = True,
    ) -> M11SecondVoxelizationResult:
        """Compute one realization and optionally replace its prior M11.1 result."""
        realization = self._realization(realization_id)
        m9 = self.project.m9_state
        spatial = self.project.spatial_grid_config
        if m9.parameter_field_metadata is None or not m9.parameter_field_arrays:
            raise ValueError("M11 second voxelization requires a complete M9 parameter field")
        if spatial is None:
            raise ValueError("M11 second voxelization requires spatial grid configuration")
        result = M11SecondVoxelizer(
            m9.parameter_field_metadata,
            m9.parameter_field_arrays,
            spatial.generation_domain,
            realization,
            config,
            progress=progress,
            cancelled=cancelled,
        ).compute()
        if cancelled is not None and cancelled():
            raise InterruptedError("M11 second voxelization cancelled")
        if commit:
            self.commit(result)
        return result

    def commit(self, result: M11SecondVoxelizationResult) -> None:
        """Commit one complete result without retaining a partial predecessor."""
        if not result.complete:
            raise ValueError("Cannot commit an incomplete M11 result")
        realization = self._realization(result.source_m10_realization_id)
        if realization.config_hash != result.source_m10_config_hash:
            raise ValueError("M11 result source M10 config hash is stale")
        current = [
            item
            for item in self.project.m11_state.results
            if item.source_m10_realization_id != result.source_m10_realization_id
        ]
        current.append(result)
        current.sort(key=lambda item: item.source_m10_realization_id)
        self.project.m11_state.results = current
        self.project.m11_state.provenance.update(
            {
                "phase": "M11.1 exact second voxelization",
                "connectivity_implemented": False,
                "block_analysis_implemented": False,
            }
        )

    def is_valid(self, result: M11SecondVoxelizationResult) -> bool:
        """Whether a stored result still matches its source M10 realization and field hash."""
        try:
            realization = self._realization(result.source_m10_realization_id)
        except ValueError:
            return False
        metadata = self.project.m9_state.parameter_field_metadata
        if metadata is None:
            return False
        field_hash = parameter_field_hash(metadata, self.project.m9_state.parameter_field_arrays)
        return (
            result.complete
            and result.source_m10_config_hash == realization.config_hash
            and result.source_parameter_field_hash == field_hash
        )

    def _realization(self, realization_id: str) -> M10Realization:
        realization = next(
            (item for item in self.project.m10_state.realizations if item.realization_id == realization_id), None
        )
        if realization is None or not realization.complete:
            raise ValueError(f"Complete M10 realization not found: {realization_id}")
        return realization
