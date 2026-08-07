"""M7 validation borehole holdout service.

Splits boreholes into calibration and validation sets BEFORE any modelling.
Validation boreholes must not participate in:
  - Domain fitting
  - Joint set clustering / Fisher statistics
  - Density estimation
  - DFN parameter assignment

Supports manual selection, random holdout (fixed seed), and stratified
random holdout (by structural domain).

DO NOT allow validation data to leak into calibration outputs.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Set, Tuple
from datetime import datetime, timezone

import numpy as np

from dfn_cave_studio.models.data_management import (
    ValidationHoldout, HoldoutConfig, HoldoutRole, new_record_id,
)


class HoldoutService:
    """Manage calibration/validation borehole split."""

    def __init__(self):
        self._config = HoldoutConfig()
        self._holdouts: Dict[str, ValidationHoldout] = {}

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def config(self) -> HoldoutConfig:
        return self._config

    @property
    def is_locked(self) -> bool:
        return self._config.locked

    @property
    def calibration_holes(self) -> List[str]:
        return sorted([h.hole_id for h in self._holdouts.values()
                       if h.role == HoldoutRole.CALIBRATION])

    @property
    def validation_holes(self) -> List[str]:
        return sorted([h.hole_id for h in self._holdouts.values()
                       if h.role == HoldoutRole.VALIDATION])

    # ── Selection methods ─────────────────────────────────────────────────

    def select_manual(self, hole_ids: List[str], validation_ids: List[str]) -> None:
        """Manually select validation boreholes.

        Args:
            hole_ids: All borehole IDs.
            validation_ids: IDs to hold out for validation.
        """
        if self._config.locked:
            raise RuntimeError("Holdout is locked. Unlock first.")
        self._holdouts.clear()
        for hid in hole_ids:
            role = HoldoutRole.VALIDATION if hid in validation_ids else HoldoutRole.CALIBRATION
            self._holdouts[hid] = ValidationHoldout(
                hole_id=hid, role=role, selection_method="manual",
                created_at=datetime.now(timezone.utc),
            )
        self._config.method = "manual"
        self._config.holdouts = list(self._holdouts.values())

    def select_random(self, hole_ids: List[str], validation_fraction: float = 0.2,
                      random_seed: int = 42) -> None:
        """Randomly select validation boreholes with fixed seed.

        Args:
            hole_ids: All borehole IDs.
            validation_fraction: Fraction to hold out (0.0–1.0).
            random_seed: Fixed seed for reproducibility.
        """
        if self._config.locked:
            raise RuntimeError("Holdout is locked. Unlock first.")
        if validation_fraction < 0 or validation_fraction > 1:
            raise ValueError("validation_fraction must be in [0, 1]")

        n_total = len(hole_ids)
        n_validation = max(self._config.min_validation_count,
                           int(round(n_total * validation_fraction)))
        n_validation = min(n_validation, n_total - 1)  # keep at least 1 calibration

        rng = np.random.default_rng(random_seed)
        shuffled = list(hole_ids)
        rng.shuffle(shuffled)
        validation_set = set(shuffled[:n_validation])

        self._holdouts.clear()
        for hid in hole_ids:
            role = HoldoutRole.VALIDATION if hid in validation_set else HoldoutRole.CALIBRATION
            self._holdouts[hid] = ValidationHoldout(
                hole_id=hid, role=role, selection_method="random",
                random_seed=random_seed,
                created_at=datetime.now(timezone.utc),
            )
        self._config.method = "random"
        self._config.validation_fraction = validation_fraction
        self._config.random_seed = random_seed
        self._config.holdouts = list(self._holdouts.values())

    def select_stratified(self, hole_ids: List[str], domain_assignments: Dict[str, int],
                          validation_fraction: float = 0.2, random_seed: int = 42) -> None:
        """Stratified random holdout by structural domain.

        Args:
            hole_ids: All borehole IDs.
            domain_assignments: Dict of hole_id → domain_id.
            validation_fraction: Fraction per domain.
            random_seed: Fixed seed.
        """
        if self._config.locked:
            raise RuntimeError("Holdout is locked. Unlock first.")

        rng = np.random.default_rng(random_seed)
        self._holdouts.clear()

        # Group by domain
        by_domain: Dict[int, List[str]] = {}
        for hid in hole_ids:
            did = domain_assignments.get(hid, 0)
            by_domain.setdefault(did, []).append(hid)

        for did, holes in by_domain.items():
            n_val = max(1, int(round(len(holes) * validation_fraction)))
            n_val = min(n_val, len(holes) - 1) if len(holes) > 1 else 0
            shuffled = list(holes)
            rng.shuffle(shuffled)
            validation_set = set(shuffled[:n_val])
            for hid in holes:
                role = HoldoutRole.VALIDATION if hid in validation_set else HoldoutRole.CALIBRATION
                self._holdouts[hid] = ValidationHoldout(
                    hole_id=hid, role=role, selection_method="stratified",
                    random_seed=random_seed, domain_stratum=did,
                    created_at=datetime.now(timezone.utc),
                )

        self._config.method = "stratified"
        self._config.validation_fraction = validation_fraction
        self._config.random_seed = random_seed
        self._config.stratify_by_domain = True
        self._config.holdouts = list(self._holdouts.values())

    # ── Config update ──────────────────────────────────────────────────────

    def update_config(
        self,
        *,
        method: str | None = None,
        validation_fraction: float | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Update holdout configuration fields without direct _config access.

        Args:
            method: Selection method ("manual", "random", "stratified").
            validation_fraction: Fraction of boreholes for validation [0, 1].
            random_seed: Random seed for reproducibility.
        """
        if method is not None:
            self._config.method = method
        if validation_fraction is not None:
            self._config.validation_fraction = validation_fraction
        if random_seed is not None:
            self._config.random_seed = random_seed

    # ── Lock / unlock ─────────────────────────────────────────────────────

    def lock(self) -> None:
        """Lock the holdout split. Downstream results are invalidated on unlock."""
        self._config.locked = True
        for h in self._holdouts.values():
            h.locked = True
            h.locked_at = datetime.now(timezone.utc)

    def unlock(self) -> None:
        """Unlock the holdout split. Downstream results must be invalidated."""
        self._config.locked = False
        for h in self._holdouts.values():
            h.locked = False
            h.unlocked_at = datetime.now(timezone.utc)

    # ── Query ─────────────────────────────────────────────────────────────

    def get_role(self, hole_id: str) -> HoldoutRole:
        """Get the role of a borehole."""
        h = self._holdouts.get(hole_id)
        return h.role if h else HoldoutRole.CALIBRATION

    def is_calibration(self, hole_id: str) -> bool:
        return self.get_role(hole_id) == HoldoutRole.CALIBRATION

    def is_validation(self, hole_id: str) -> bool:
        return self.get_role(hole_id) == HoldoutRole.VALIDATION

    def filter_calibration(self, hole_ids: List[str]) -> List[str]:
        """Return only calibration hole IDs from a list."""
        return [h for h in hole_ids if self.is_calibration(h)]

    def filter_validation(self, hole_ids: List[str]) -> List[str]:
        """Return only validation hole IDs from a list."""
        return [h for h in hole_ids if self.is_validation(h)]

    def to_dict(self) -> Dict:
        return {
            "method": self._config.method,
            "validation_fraction": self._config.validation_fraction,
            "random_seed": self._config.random_seed,
            "locked": self._config.locked,
            "holdouts": [
                {
                    "hole_id": h.hole_id, "role": h.role.value,
                    "selection_method": h.selection_method,
                    "random_seed": h.random_seed,
                    "domain_stratum": h.domain_stratum,
                    "created_at": h.created_at.isoformat() if h.created_at else None,
                }
                for h in self._holdouts.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HoldoutService":
        svc = cls()
        svc._config.method = data.get("method", "manual")
        svc._config.validation_fraction = data.get("validation_fraction", 0.2)
        svc._config.random_seed = data.get("random_seed", 42)
        svc._config.locked = data.get("locked", False)
        for hd in data.get("holdouts", []):
            h = ValidationHoldout(
                hole_id=hd["hole_id"],
                role=HoldoutRole(hd.get("role", "calibration")),
                selection_method=hd.get("selection_method", "manual"),
                random_seed=hd.get("random_seed"),
                domain_stratum=hd.get("domain_stratum"),
            )
            svc._holdouts[h.hole_id] = h
        return svc
