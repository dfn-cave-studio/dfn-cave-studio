"""Shared VTK lookup-table safety helpers for finite display geometry."""

from __future__ import annotations

from typing import Any


def make_lookup_table_opaque(actor: Any) -> None:
    """Set every mapper LUT alpha to one without forcing actor opacity."""
    mapper = getattr(actor, "mapper", None)
    if mapper is None:
        get_mapper = getattr(actor, "GetMapper", None)
        mapper = get_mapper() if get_mapper is not None else None
    if mapper is None:
        return
    get_lookup_table = getattr(mapper, "GetLookupTable", None)
    lookup_table = get_lookup_table() if get_lookup_table is not None else None
    if lookup_table is None:
        return
    for index in range(lookup_table.GetNumberOfTableValues()):
        red, green, blue, _ = lookup_table.GetTableValue(index)
        lookup_table.SetTableValue(index, red, green, blue, 1.0)
    for getter_name, setter_name in (
        ("GetNanColor", "SetNanColor"),
        ("GetBelowRangeColor", "SetBelowRangeColor"),
        ("GetAboveRangeColor", "SetAboveRangeColor"),
    ):
        red, green, blue, _ = getattr(lookup_table, getter_name)()
        getattr(lookup_table, setter_name)(red, green, blue, 1.0)
    lookup_table.Modified()
    modified = getattr(mapper, "Modified", None)
    if modified is not None:
        modified()
