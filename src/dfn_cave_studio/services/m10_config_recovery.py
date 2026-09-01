"""Read-only inspection and explicit copy-on-recovery for invalid M10 configuration metadata."""

from __future__ import annotations

import json
import os
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dfn_cave_studio.models.m10 import M10GenerationConfig
from dfn_cave_studio.services.workflow_controller import StepStatus


@dataclass(frozen=True, slots=True)
class M10ConfigInspection:
    """Raw saved configuration and its validation outcome."""

    source_path: Path
    raw_config: dict[str, Any]
    validation_errors: tuple[str, ...]
    valid: bool


class M10ConfigRecoveryService:
    """Inspect an invalid archive and create a separately named recovery copy."""

    M10_STATE_PATH = "parameters/m10_state.json"
    M11_STATE_PATH = "parameters/m11_state.json"
    WORKFLOW_PATH = "m7/workflow.json"

    @classmethod
    def inspect(cls, source_path: Path) -> M10ConfigInspection:
        """Read only the saved M10 metadata; never load or modify scientific arrays."""
        source = Path(source_path).resolve()
        with zipfile.ZipFile(source, "r") as archive:
            if cls.M10_STATE_PATH not in archive.namelist():
                raise ValueError("The project does not contain M10 configuration metadata")
            state = json.loads(archive.read(cls.M10_STATE_PATH).decode("utf-8"))
        raw_config = state.get("config")
        if not isinstance(raw_config, dict):
            raise TypeError("The saved M10 state does not contain a configuration object")
        try:
            M10GenerationConfig.model_validate(raw_config)
        except ValidationError as exc:
            errors = tuple(
                f"{'.'.join(str(part) for part in item['loc']) or 'config'}: {item['msg']}"
                for item in exc.errors()
            )
            return M10ConfigInspection(source, raw_config, errors, False)
        return M10ConfigInspection(source, raw_config, (), True)

    @classmethod
    def recover_to_new_file(
        cls,
        source_path: Path,
        target_path: Path,
        replacement_config: M10GenerationConfig | dict[str, Any],
        *,
        confirmed: bool,
    ) -> Path:
        """Copy an archive with confirmed replacement metadata and stale scientific workflow status."""
        if not confirmed:
            raise ValueError("M10 configuration recovery requires explicit user confirmation")
        source = Path(source_path).resolve()
        target = Path(target_path).resolve()
        if source == target:
            raise ValueError("Recovery must be saved to a new file; the source project is read-only")
        if target.exists():
            raise FileExistsError("Recovery target already exists; choose a new file name")
        validated = M10GenerationConfig.model_validate(
            replacement_config.model_dump(mode="python")
            if isinstance(replacement_config, M10GenerationConfig)
            else replacement_config
        )
        inspection = cls.inspect(source)
        timestamp = datetime.now(UTC).isoformat()
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(source, "r") as source_archive, zipfile.ZipFile(
                temporary, "w", allowZip64=True
            ) as target_archive:
                for info in source_archive.infolist():
                    if info.filename == cls.M10_STATE_PATH:
                        state = json.loads(source_archive.read(info.filename).decode("utf-8"))
                        state["config"] = validated.model_dump(mode="json")
                        provenance = state.setdefault("provenance", {})
                        provenance["config_recovery"] = {
                            "recovered_at": timestamp,
                            "source_file": str(source),
                            "original_config": inspection.raw_config,
                            "replacement_config": validated.model_dump(mode="json"),
                            "result_config_consistency_confirmed": False,
                            "scientific_arrays_preserved": True,
                            "note": "Configuration metadata recovery is not M10 regeneration.",
                        }
                        target_archive.writestr(info, json.dumps(state, indent=2).encode("utf-8"))
                    elif info.filename == cls.M11_STATE_PATH:
                        state = json.loads(source_archive.read(info.filename).decode("utf-8"))
                        provenance = state.setdefault("provenance", {})
                        provenance["m10_config_consistency"] = "unconfirmed_after_metadata_recovery"
                        target_archive.writestr(info, json.dumps(state, indent=2).encode("utf-8"))
                    elif info.filename == cls.WORKFLOW_PATH:
                        workflow = json.loads(source_archive.read(info.filename).decode("utf-8"))
                        for step_id in ("explicit_dfn", "second_voxelization"):
                            step = workflow.get("steps", {}).get(step_id)
                            if step is not None:
                                step["status"] = StepStatus.STALE.value
                                step.setdefault("metadata", {})["config_consistency"] = (
                                    "unconfirmed_after_metadata_recovery"
                                )
                        target_archive.writestr(info, json.dumps(workflow, indent=2).encode("utf-8"))
                    else:
                        with source_archive.open(info, "r") as reader, target_archive.open(info, "w") as writer:
                            while chunk := reader.read(1024 * 1024):
                                writer.write(chunk)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target
