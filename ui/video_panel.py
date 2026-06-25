"""
ui/video_panel.py
=================
Левая часть интерфейса: дисплей видео + управление источником.

Новое: кнопка ПАУЗА / ПРОДОЛЖИТЬ (п. 8).
Требование 3.2.1: основная рабочая область (побольше, п. 16).
"""

from __future__ import annotations

import cv2
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QFrame, QSizePolicy, QFileDialog,
)

from ui.widgets import FPSBadge

_ALL_FILTER = (
    "Все медиафайлы (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv *.wmv);;"
    "Изображения (*.jpg *.jpeg *.png *.bmp);;"
    "Видео (*.mp4 *.avi *.mov *.mkv *.wmv *.ts)"
)


class VideoPanel(QWidget):
    """
    Панель видео. Испускает сигналы — сама ничего не знает о модели.
    """

    source_opened  = pyqtSignal(str)   # выбран источник
    stream_stop    = pyqtSignal()      # нажат стоп
    pause_toggled  = pyqtSignal()      # нажата пауза

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._paused = False
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 4, 0)
        root.setSpacing(4)

        # Верхняя строка: FPS + время
        fps_row = QHBoxLayout()
        fps_row.setContentsMargins(0, 0, 0, 0)
        self.fps_badge = FPSBadge()
        self._time_lbl = QLabel()
        self._time_lbl.setStyleSheet(
            "font-size:11px;color:#3a6a28;background:transparent;"
        )
        self._time_lbl.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        fps_row.addWidget(self.fps_badge)
        fps_row.addStretch()
        fps_row.addWidget(self._time_lbl)
        root.addLayout(fps_row)

        # Основной дисплей (п. 16 — занимает всё доступное пространство)
        self.display = VideoDisplay()
        root.addWidget(self.display, 1)

        # Панель управления
        root.addWidget(self._build_source_bar())

    def _build_source_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(42)
        bar.setStyleSheet(
            "QFrame{border:1px solid #1a3010;background:#040804;border-radius:2px;}"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)

        lbl = QLabel("SRC:")
        lbl.setStyleSheet("color:#3a6a28;font-size:10px;background:transparent;")
        lbl.setFixedWidth(28)

        self._src_input = QLineEdit()
        self._src_input.setPlaceholderText(
            "Путь к файлу / rtsp:// / http:// / 0 (камера)"
        )
        self._src_input.setFixedHeight(26)
        self._src_input.returnPressed.connect(self._on_stream)

        # Кнопка паузы
        self._btn_pause = QPushButton("[ ПАУЗА ]")
        self._btn_pause.setFixedHeight(26)
        self._btn_pause.clicked.connect(self._on_pause)

        lay.addWidget(lbl)
        lay.addWidget(self._src_input, 1)
        lay.addWidget(self._mkbtn("ФАЙЛ",   self._on_file))
        lay.addWidget(self._mkbtn("КАМЕРА", self._on_webcam))
        lay.addWidget(self._mkbtn("ПОТОК",  self._on_stream))
        lay.addWidget(self._btn_pause)
        lay.addWidget(self._mkbtn("СТОП",   self.stream_stop))

        return bar

    @staticmethod
    def _mkbtn(text: str, slot) -> QPushButton:
        b = QPushButton(f"[ {text} ]")
        b.clicked.connect(slot)
        b.setFixedHeight(26)
        return b

    # ── Слоты ────────────────────────────────────────────────────────────

    def _on_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть медиафайл", "", _ALL_FILTER)
        if path:
            self._src_input.setText(path)
            self.source_opened.emit(path)

    def _on_webcam(self) -> None:
        self._src_input.setText("0")
        self.source_opened.emit("0")

    def _on_stream(self) -> None:
        src = self._src_input.text().strip()
        if src:
            self.source_opened.emit(src)

    def _on_pause(self) -> None:
        self._paused = not self._paused
        self._btn_pause.setText("[ ПРОДОЛЖИТЬ ]" if self._paused else "[ ПАУЗА ]")
        self.pause_toggled.emit()

    def set_paused_state(self, paused: bool) -> None:
        """Синхронизирует текст кнопки с реальным состоянием."""
        self._paused = paused
        self._btn_pause.setText("[ ПРОДОЛЖИТЬ ]" if paused else "[ ПАУЗА ]")

    # ── Публичные методы ─────────────────────────────────────────────────

    def show_frame(self, frame: np.ndarray) -> None:
        self.display.set_frame(frame)

    def set_time(self, text: str) -> None:
        self._time_lbl.setText(text)

    def show_no_signal(self) -> None:
        self.display.clear()


class VideoDisplay(QLabel):
    """Виджет отображения кадра с масштабированием."""

    _PLACEHOLDER = (
        "[ НЕТ СИГНАЛА ]\n\n"
        "Загрузите видеофайл, изображение\n"
        "или введите URL трансляции"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(
            "QLabel{"
            "background-color:#020402;"
            "border:2px solid #1a3a10;"
            "color:#1a4010;"
            "font-size:13px;"
            "font-family:'Courier New',monospace;}"
        )
        self.setText(self._PLACEHOLDER)
        self._cache: QPixmap | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        h, w, ch = frame.shape
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._cache = QPixmap.fromImage(img)
        self._refresh()

    def clear(self) -> None:
        self._cache = None
        self.setText(self._PLACEHOLDER)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._cache is None:
            return
        self.setPixmap(
            self._cache.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
