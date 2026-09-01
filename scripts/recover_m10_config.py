"""Inspect invalid M10 metadata and create an explicitly confirmed recovery copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dfn_cave_studio.services.m10_config_recovery import M10ConfigRecoveryService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read an invalid M10 config or save a separately named metadata-recovery copy."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold-mode", choices=["auto", "manual"])
    parser.add_argument("--small-area-share", type=float)
    parser.add_argument("--medium-large-cumulative-share", type=float)
    parser.add_argument("--manual-small-medium-radius", type=float)
    parser.add_argument("--manual-medium-large-radius", type=float)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    inspection = M10ConfigRecoveryService.inspect(args.source)
    print(json.dumps({"valid": inspection.valid, "errors": inspection.validation_errors, "config": inspection.raw_config}, indent=2))
    if args.output is None:
        return 0 if inspection.valid else 2
    if args.threshold_mode is None:
        parser.error("--threshold-mode is required when creating a recovery copy")
    replacement = dict(inspection.raw_config)
    replacement["size_threshold_mode"] = args.threshold_mode
    if args.threshold_mode == "auto":
        if args.small_area_share is None or args.medium_large_cumulative_share is None:
            parser.error("Auto recovery requires both explicit Auto share values")
        replacement["small_area_share"] = args.small_area_share
        replacement["medium_large_cumulative_share"] = args.medium_large_cumulative_share
    else:
        if args.manual_small_medium_radius is None or args.manual_medium_large_radius is None:
            parser.error("Manual recovery requires both explicit radius thresholds")
        replacement["manual_small_medium_radius"] = args.manual_small_medium_radius
        replacement["manual_medium_large_radius"] = args.manual_medium_large_radius
    recovered = M10ConfigRecoveryService.recover_to_new_file(
        args.source, args.output, replacement, confirmed=args.confirm
    )
    print(f"Recovery copy created: {recovered}")
    print("Existing M10/M11 results were preserved but their configuration consistency is unconfirmed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
