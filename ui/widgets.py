"""
ui/widgets.py
=============
Переиспользуемые мелкие виджеты интерфейса.

Все виджеты стилизованы под интерфейс наводчика-оператора Т-90М:
монохромный зелёный на чёрном, шрифт Courier New, военный HUD.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import QLabel, QWidget, QFrame, QSizePolicy


class SectionLabel(QLabel):
    """Заголовок раздела с засечками — стиль военного HUD."""

    def __init__(self, text: str, parent=None):
        super().__init__(f"◈ {text}", parent)
        self.setStyleSheet(
            "font-size: 9px; color: #3a7a28; letter-spacing: 2px; "
            "background: transparent; padding: 2px 0;"
        )


class ValueLabel(QLabel):
    """Большой информационный лейбл (тип техники, FPS и т.д.)."""

    def __init__(
        self,
        text: str = "—",
        font_size: int = 20,
        color: str = "#8acc6a",
        parent=None,
    ):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"font-size: {font_size}px; color: {color}; "
            f"font-weight: bold; letter-spacing: 2px; background: transparent;"
        )
        self.setWordWrap(True)


class HLine(QFrame):
    """Горизонтальный разделитель."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFixedHeight(1)
        self.setStyleSheet("border: none; border-top: 1px solid #1a3010; background: transparent;")


class TacticalButton(QWidget):
    """
    Кнопка в стиле тактического дисплея — с мигающим индикатором.
    Используется для критичных действий (скриншот, остановка).
    """

    clicked = __import__("PyQt6.QtCore", fromlist=["pyqtSignal"]).pyqtSignal()

    def __init__(self, text: str, accent_color: str = "#8acc6a", parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QPushButton, QHBoxLayout
        self._btn = QPushButton(text)
        self._btn.clicked.connect(self.clicked)
        self._btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #0a1410;
                color: {accent_color};
                border: 1px solid {accent_color};
                border-radius: 2px;
                padding: 6px 14px;
                font-size: 11px;
                letter-spacing: 1px;
                font-family: 'Courier New', monospace;
            }}
            QPushButton:hover {{
                background-color: #162e0e;
                color: #ccffaa;
            }}
            QPushButton:pressed {{ background-color: #244818; }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._btn)

    def setEnabled(self, enabled: bool) -> None:  # type: ignore[override]
        self._btn.setEnabled(enabled)
        super().setEnabled(enabled)


class AlertIndicator(QWidget):
    """
    Индикатор обнаружения цели (требование 3.2.5).

    Зелёный  — целей нет.
    Красный  — цель обнаружена (мигает).
    """

    _GREEN  = QColor(20, 200, 60)
    _RED    = QColor(220, 30, 30)
    _DIM_R  = QColor(80, 10, 10)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(52, 52)
        self._has_target = False
        self._blink      = True

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._toggle_blink)
        self._timer.start(480)

    def set_target(self, has_target: bool) -> None:
        if has_target != self._has_target:
            self._has_target = has_target
            self.update()

    def _toggle_blink(self) -> None:
        self._blink = not self._blink
        if self._has_target:
            self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._has_target:
            color = self._RED if self._blink else self._DIM_R
        else:
            color = self._GREEN

        # Внешнее кольцо
        p.setPen(QPen(color.darker(150), 2))
        p.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        p.drawEllipse(4, 4, 44, 44)

        # Основной круг
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawEllipse(10, 10, 32, 32)

        # Блик (отражение)
        highlight = QColor(255, 255, 255, 60)
        p.setBrush(QBrush(highlight))
        p.drawEllipse(14, 13, 10, 7)

        p.end()


class FPSBadge(QLabel):
    """
    Счётчик FPS (требование 3.2.3).
    Меняет цвет при падении ниже TARGET_FPS.
    """

    def __init__(self, parent=None):
        super().__init__("FPS: --", parent)
        from config import TARGET_FPS
        self._target = TARGET_FPS
        self._update_style("#8acc6a")

    def set_fps(self, fps: float) -> None:
        self.setText(f"FPS: {fps:05.1f}")
        color = "#8acc6a" if fps >= self._target else "#ff6a2a"
        self._update_style(color)

    def _update_style(self, color: str) -> None:
        self.setStyleSheet(
            f"font-size: 14px; color: {color}; font-weight: bold; "
            f"background: transparent; letter-spacing: 1px;"
        )


class LogWidget(__import__("PyQt6.QtWidgets", fromlist=["QTextEdit"]).QTextEdit):
    """
    Журнал событий с автопрокруткой.
    Хранит не более MAX_LINES строк чтобы не расти в памяти.
    """

    MAX_LINES = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFixedHeight(72)

    def append_event(self, msg: str) -> None:
        import datetime
        ts   = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        self.append(line)

        # Ограничиваем количество строк
        doc = self.document()
        while doc.blockCount() > self.MAX_LINES:
            cursor = self.textCursor()
            cursor.movePosition(
                cursor.MoveOperation.Start
            )
            cursor.select(cursor.SelectionType.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

        # Автопрокрутка вниз
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())