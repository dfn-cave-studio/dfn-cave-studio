"""Application-wide Qt translation and language preference management."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from dfn_cave_studio.ui.qt_adapter import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QCoreApplication,
    QDockWidget,
    QDialogButtonBox,
    QEvent,
    QGroupBox,
    QLabel,
    QLineEdit,
    QLocale,
    QLibraryInfo,
    QMenu,
    QObject,
    QPushButton,
    QRadioButton,
    QSettings,
    QSignalBlocker,
    QTabWidget,
    QTableWidget,
    QToolButton,
    QToolBar,
    QTreeWidget,
    QTranslator,
    QWidget,
    Signal,
    is_valid_qobject,
)

LANGUAGE_ENGLISH = "en"
LANGUAGE_CHINESE = "zh_CN"
SUPPORTED_LANGUAGES = (LANGUAGE_CHINESE, LANGUAGE_ENGLISH)
TRANSLATION_CONTEXT = "DFNCaveStudio"


def translation_directory() -> Path:
    """Return the packaged Qt translation directory."""
    return Path(__file__).resolve().parent.parent / "resources" / "i18n"


def system_default_language(locale: QLocale | None = None) -> str:
    """Choose Chinese on a Chinese system and English everywhere else."""
    selected = locale or QLocale.system()
    return LANGUAGE_CHINESE if selected.language() == QLocale.Language.Chinese else LANGUAGE_ENGLISH


def tr(source: str) -> str:
    """Translate an application-owned user-facing string with English fallback."""
    translated = QCoreApplication.translate(TRANSLATION_CONTEXT, source)
    return translated or source


class LanguageManager(QObject):
    """Own the active translator and the user-level language preference."""

    language_changed = Signal(str)
    switching_enabled_changed = Signal(bool)

    def __init__(
        self,
        app: QApplication | None = None,
        *,
        settings: QSettings | None = None,
        resource_dir: Path | None = None,
    ) -> None:
        super().__init__(app)
        self._app = app or QApplication.instance()
        if self._app is None:
            raise RuntimeError("LanguageManager requires a QApplication")
        self._settings = settings or QSettings("DFNCaveStudio", "Preferences")
        self._resource_dir = resource_dir or translation_directory()
        self._translator: QTranslator | None = None
        self._qt_translator: QTranslator | None = None
        self._busy_reasons: set[str] = set()
        stored = str(self._settings.value("language", "") or "")
        self._language = stored if stored in SUPPORTED_LANGUAGES else system_default_language()
        self._app.installEventFilter(self)

    @property
    def language(self) -> str:
        """Return the active stable language code."""
        return self._language

    @property
    def switching_enabled(self) -> bool:
        """Return whether language switching is safe at this moment."""
        return not self._busy_reasons

    def initialize(self) -> bool:
        """Install the saved/default language without rewriting the preference."""
        return self.set_language(self._language, persist=False, force=True)

    def set_language(self, language: str, *, persist: bool = True, force: bool = False) -> bool:
        """Install a language and refresh visible widgets.

        Returns ``False`` when switching is busy or a Chinese catalog is missing.
        English never needs a catalog and is the guaranteed fallback.
        """
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported interface language: {language}")
        if not force and not self.switching_enabled:
            return False

        old_translator = self._translator
        if old_translator is not None and language != self._language:
            # Restore authoritative English sources while the old catalog is
            # still available, so dynamic text is not mistaken for a source.
            self.retranslate_open_windows(LANGUAGE_ENGLISH)
        if old_translator is not None:
            self._app.removeTranslator(old_translator)
            self._translator = None
        if self._qt_translator is not None:
            self._app.removeTranslator(self._qt_translator)
            self._qt_translator = None

        applied = language
        if language == LANGUAGE_CHINESE:
            translator = QTranslator(self)
            catalog = self._resource_dir / "dfn_cave_studio_zh_CN.qm"
            if catalog.is_file() and translator.load(str(catalog)):
                self._app.installTranslator(translator)
                self._translator = translator
                qt_translator = QTranslator(self)
                qt_catalog = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)) / "qtbase_zh_CN.qm"
                if qt_catalog.is_file() and qt_translator.load(str(qt_catalog)):
                    self._app.installTranslator(qt_translator)
                    self._qt_translator = qt_translator
            else:
                applied = LANGUAGE_ENGLISH

        self._language = applied
        if persist:
            self._settings.setValue("language", applied)
            self._settings.sync()
        self.retranslate_open_windows()
        self.language_changed.emit(applied)
        return applied == language

    def set_busy(self, reason: str, busy: bool) -> None:
        """Disable switching while a named critical operation is running."""
        before = self.switching_enabled
        if busy:
            self._busy_reasons.add(reason)
        else:
            self._busy_reasons.discard(reason)
        if before != self.switching_enabled:
            self.switching_enabled_changed.emit(self.switching_enabled)

    @contextmanager
    def busy(self, reason: str) -> Iterator[None]:
        """Temporarily prevent language changes during a critical operation."""
        self.set_busy(reason, True)
        try:
            yield
        finally:
            self.set_busy(reason, False)

    def retranslate_open_windows(self, language: str | None = None) -> None:
        """Retranslate existing windows without rebuilding application state."""
        # Snapshot first: translating a widget can dispatch Qt events which alter
        # the top-level collection.  A wrapper may also outlive its C++ QObject
        # while deferred deletion is being drained, so validate before touching
        # any Qt property or walking its child tree.
        for widget in tuple(QApplication.topLevelWidgets()):
            if not is_valid_qobject(widget):
                continue
            if bool(widget.property("i18n_window_closed")):
                continue
            retranslate_widget_tree(widget, language or self._language)

    def dispose(self) -> None:
        """Remove translators and the event filter owned by this manager."""
        if self._translator is not None:
            self._app.removeTranslator(self._translator)
            self._translator = None
        if self._qt_translator is not None:
            self._app.removeTranslator(self._qt_translator)
            self._qt_translator = None
        self._app.removeEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API
        """Translate newly shown windows and widgets created after startup."""
        if not is_valid_qobject(watched):
            return False
        if event.type() == QEvent.Type.Close and isinstance(watched, QWidget) and watched.isWindow():
            watched.setProperty("i18n_window_closed", True)
        elif event.type() == QEvent.Type.Show and isinstance(watched, QWidget):
            if watched.isWindow():
                watched.setProperty("i18n_window_closed", False)
            retranslate_widget_tree(watched, self._language)
        return super().eventFilter(watched, event)


def _source(obj: QObject, key: str, current: str) -> str:
    property_name = f"i18n_source_{key}"
    source = obj.property(property_name)
    if source is None or current not in {str(source), tr(str(source))}:
        source = current
        obj.setProperty(property_name, source)
    return str(source)


def _translated(source: str, language: str) -> str:
    return source if language == LANGUAGE_ENGLISH else tr(source)


def retranslate_widget_tree(root: QWidget, language: str) -> None:
    """Update presentation text while preserving widget values and signals."""
    if not is_valid_qobject(root):
        return
    objects: list[QObject] = [root, *root.findChildren(QObject)]
    for obj in objects:
        if not is_valid_qobject(obj):
            continue
        with QSignalBlocker(obj):
            if isinstance(obj, QWidget) and not isinstance(obj, (QDockWidget, QToolBar, QMenu)) and obj.windowTitle():
                source = _source(obj, "window_title", obj.windowTitle())
                obj.setWindowTitle(_translated(source, language))
            if isinstance(obj, (QLabel, QPushButton, QCheckBox, QRadioButton, QToolButton, QGroupBox)) and not (
                isinstance(obj, QPushButton) and isinstance(obj.parent(), QDialogButtonBox)
            ):
                source = _source(obj, "text", obj.text() if hasattr(obj, "text") else obj.title())
                if isinstance(obj, QGroupBox):
                    obj.setTitle(_translated(source, language))
                else:
                    obj.setText(_translated(source, language))
            if isinstance(obj, QDockWidget):
                source = _source(obj, "dock_title", obj.windowTitle())
                obj.setWindowTitle(_translated(source, language))
            if isinstance(obj, QToolBar):
                source = _source(obj, "toolbar_title", obj.windowTitle())
                obj.setWindowTitle(_translated(source, language))
            if isinstance(obj, QMenu):
                source = _source(obj, "menu_title", obj.title())
                obj.setTitle(_translated(source, language))
            if isinstance(obj, QAction) and obj.menu() is None:
                source = _source(obj, "action_text", obj.text())
                obj.setText(_translated(source, language))
                if obj.toolTip():
                    tip = _source(obj, "action_tooltip", obj.toolTip())
                    obj.setToolTip(_translated(tip, language))
                    obj.setStatusTip(_translated(tip, language))
            if isinstance(obj, QComboBox):
                sources = obj.property("i18n_item_sources")
                current_texts = [obj.itemText(index) for index in range(obj.count())]
                sources_are_current = isinstance(sources, list) and len(sources) == obj.count() and all(
                    current in {str(source), tr(str(source))}
                    for current, source in zip(current_texts, sources, strict=True)
                )
                if not sources_are_current:
                    sources = [obj.itemText(index) for index in range(obj.count())]
                    obj.setProperty("i18n_item_sources", sources)
                selected = obj.currentData()
                index_before = obj.currentIndex()
                for index, source in enumerate(sources):
                    obj.setItemText(index, _translated(str(source), language))
                data_index = obj.findData(selected) if selected is not None else -1
                obj.setCurrentIndex(data_index if data_index >= 0 else index_before)
            if isinstance(obj, QLineEdit) and obj.placeholderText():
                source = _source(obj, "placeholder", obj.placeholderText())
                obj.setPlaceholderText(_translated(source, language))
            if isinstance(obj, QTableWidget):
                for axis, count, getter in (
                    ("h", obj.columnCount(), obj.horizontalHeaderItem),
                    ("v", obj.rowCount(), obj.verticalHeaderItem),
                ):
                    for index in range(count):
                        item = getter(index)
                        if item is None:
                            continue
                        key = f"i18n_source_{axis}_{index}"
                        source = obj.property(key)
                        if source is None:
                            source = item.text()
                            obj.setProperty(key, source)
                        item.setText(_translated(str(source), language))
            if isinstance(obj, QTabWidget):
                for index in range(obj.count()):
                    key = f"i18n_source_tab_{index}"
                    source = obj.property(key)
                    if source is None:
                        source = obj.tabText(index)
                        obj.setProperty(key, source)
                    obj.setTabText(index, _translated(str(source), language))
            if isinstance(obj, QTreeWidget):
                header = obj.headerItem()
                for column in range(obj.columnCount()):
                    source = header.data(column, 0x0100)
                    if source is None:
                        source = header.text(column)
                        header.setData(column, 0x0100, source)
                    header.setText(column, _translated(str(source), language))
                stack = [obj.topLevelItem(index) for index in range(obj.topLevelItemCount())]
                while stack:
                    item = stack.pop()
                    for column in range(obj.columnCount()):
                        source = item.data(column, 0x0100)
                        if source is None:
                            source = item.text(column)
                            item.setData(column, 0x0100, source)
                        item.setText(column, _translated(str(source), language))
                    stack.extend(item.child(index) for index in range(item.childCount()))


_manager: LanguageManager | None = None


def install_language_manager(app: QApplication | None = None) -> LanguageManager:
    """Create and initialize the process-wide language manager once."""
    global _manager
    if _manager is None:
        _manager = LanguageManager(app)
        _manager.initialize()
    return _manager


def language_manager() -> LanguageManager:
    """Return the initialized process-wide language manager."""
    if _manager is None:
        return install_language_manager()
    return _manager
