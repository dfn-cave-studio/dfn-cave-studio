"""Extract literal Qt presentation strings while preserving TS translations."""

from __future__ import annotations

import ast
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

CONTEXT = "DFNCaveStudio"
ONE_TEXT_ARGUMENT = {
    "QAction",
    "QCheckBox",
    "QDockWidget",
    "QGroupBox",
    "QLabel",
    "QPushButton",
    "QRadioButton",
    "addMenu",
    "setHeaderLabel",
    "setText",
    "setTitle",
    "setToolTip",
    "setStatusTip",
    "setWindowTitle",
}


def _method_name(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")


def _literal(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def extract(root: Path) -> set[str]:
    """Return user-facing literal strings found in the Qt UI layer."""
    messages: set[str] = set()
    for path in sorted((root / "src" / "dfn_cave_studio" / "ui").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _method_name(node)
            if name in ONE_TEXT_ARGUMENT and node.args:
                value = _literal(node.args[0])
                if value:
                    messages.add(value)
            elif name == "addItem" and node.args:
                value = _literal(node.args[0])
                if value:
                    messages.add(value)
            elif name in {"addItems", "setHorizontalHeaderLabels"} and node.args:
                sequence = node.args[0]
                if isinstance(sequence, (ast.List, ast.Tuple)):
                    messages.update(value for item in sequence.elts if (value := _literal(item)))
            elif name == "addRow" and node.args:
                value = _literal(node.args[0])
                if value:
                    messages.add(value)
            elif name in {"critical", "information", "question", "warning", "about"}:
                for argument in node.args[1:3]:
                    value = _literal(argument)
                    if value:
                        messages.add(value)
    return messages


def merge_catalog(path: Path, messages: set[str]) -> None:
    """Merge extracted sources into a TS file without replacing translations."""
    tree = ET.parse(path)
    root = tree.getroot()
    context = next((item for item in root.findall("context") if item.findtext("name") == CONTEXT), None)
    if context is None:
        context = ET.SubElement(root, "context")
        ET.SubElement(context, "name").text = CONTEXT
    existing = {item.findtext("source", "") for item in context.findall("message")}
    for source in sorted(messages - existing, key=str.casefold):
        message = ET.SubElement(context, "message")
        ET.SubElement(message, "source").text = source
        translation = ET.SubElement(message, "translation", {"type": "unfinished"})
        translation.text = ""
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main() -> int:
    """Update all checked-in TS catalogs from the repository UI source."""
    root = Path(__file__).resolve().parents[1]
    catalog_dir = root / "src" / "dfn_cave_studio" / "resources" / "i18n"
    catalogs = sorted(catalog_dir.glob("*.ts"))
    if not catalogs:
        raise RuntimeError(f"No TS catalogs found in {catalog_dir}")
    messages = extract(root)
    for catalog in catalogs:
        merge_catalog(catalog, messages)
    print(f"Extracted {len(messages)} UI source strings into {len(catalogs)} catalog(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
