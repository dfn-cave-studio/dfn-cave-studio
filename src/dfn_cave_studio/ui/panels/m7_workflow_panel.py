"""M7 workflow navigation panel — displays pipeline steps with status."""

from dfn_cave_studio.ui.qt_adapter import (
    Qt, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem, QDockWidget, QFrame,
)
from dfn_cave_studio.services.workflow_controller import (
    WorkflowController, WorkflowStep, StepStatus,
)

STATUS_COLORS = {
    StepStatus.NOT_STARTED: "#999",
    StepStatus.HAS_ISSUES: "#f57c00",
    StepStatus.READY: "#1565c0",
    StepStatus.IN_PROGRESS: "#1976d2",
    StepStatus.COMPLETED: "#2e7d32",
    StepStatus.STALE: "#c62828",
}

STATUS_ICONS = {
    StepStatus.NOT_STARTED: "○",
    StepStatus.HAS_ISSUES: "⚠",
    StepStatus.READY: "▶",
    StepStatus.IN_PROGRESS: "◉",
    StepStatus.COMPLETED: "✓",
    StepStatus.STALE: "✗",
}


class M7WorkflowPanel(QDockWidget):
    """Dockable workflow navigation panel for the M7 pipeline."""

    def __init__(self, controller: WorkflowController, main_window=None, parent=None):
        super().__init__("M7 Workflow", parent)
        self._controller = controller
        self._main_window = main_window
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        title = QLabel("<b>Data Pipeline</b>")
        layout.addWidget(title)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Step", "Status"])
        self._tree.setColumnWidth(0, 220)
        self._tree.setIndentation(0)
        self._tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

        self.setWidget(widget)
        self._refresh()

        if controller:
            controller.add_change_listener(self._refresh)

    def _refresh(self):
        self._tree.clear()
        for step in self._controller.get_steps():
            enabled = self._controller.is_enabled(step.step_id)
            item = QTreeWidgetItem(self._tree)
            icon = STATUS_ICONS.get(step.status, "?")
            label = f"{icon}  {step.name}"
            if not enabled and step.status == StepStatus.NOT_STARTED:
                label = f"🔒  {step.name} [后续]"
                item.setForeground(0, Qt.GlobalColor.gray)
            else:
                color_name = STATUS_COLORS.get(step.status, "#999")
                item.setForeground(0, Qt.GlobalColor(
                    {"#999": 7, "#f57c00": 18, "#1565c0": 13,
                     "#1976d2": 13, "#2e7d32": 13, "#c62828": 1}.get(color_name, 7)
                ))
            item.setText(0, label)
            item.setText(1, step.status.value.replace("_", " ").title())
            item.setData(0, Qt.ItemDataRole.UserRole, step.step_id)

    def _on_item_clicked(self, item, column):
        step_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not step_id:
            return
        enabled = self._controller.is_enabled(step_id)
        if not enabled:
            return
        # Dispatch to main window handlers
        if self._main_window:
            handler = getattr(self._main_window, f"_m7_{step_id}", None)
            if handler:
                handler()
