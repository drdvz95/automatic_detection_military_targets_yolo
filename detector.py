"""
detector.py
===========
Бизнес-логика обнаружения объектов.

Не импортирует ничего из ui/ — разделение ответственности.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QMutex, QMutexLocker, pyqtSignal

from config import CONFIDENCE_THRESHOLD, FPS_HISTORY_LEN, THREAT_PRIORITY, PROCESS_EVERY_N_FRAMES


# ---------------------------------------------------------------------------
# Датакласс детекции
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """Иммутабельное описание одной обнаруженной цели."""
    cls_name: str
    cls_id:   int
    conf:     float
    x1: int; y1: int
    x2: int; y2: int

    @property
    def bbox_w(self) -> int:
        return self.x2 - self.x1

    @property
    def bbox_h(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def threat_level(self) -> int:
        return THREAT_PRIORITY.get(self.cls_name, 0)

    @property
    def area(self) -> int:
        return self.bbox_w * self.bbox_h


# ---------------------------------------------------------------------------
# Загрузка модели
# ---------------------------------------------------------------------------

class ModelLoader:

    @staticmethod
    def load(weights_path: str | Path):
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics не установлен.\nВыполни: pip install ultralytics"
            ) from exc

        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Файл весов не найден: {path.resolve()}\n"
                "Обучи модель: python train.py"
            )
        return YOLO(str(path))

    @staticmethod
    def is_available() -> bool:
        try:
            import ultralytics  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False


# ---------------------------------------------------------------------------
# Инференс
# ---------------------------------------------------------------------------

def run_inference(model, frame: np.ndarray, conf: float) -> list[Detection]:
    """Запускает YOLO. При ошибке возвращает []."""
    try:
        results = model.predict(source=frame, conf=conf, verbose=False, stream=False)
    except Exception:
        return []

    if not results or results[0].boxes is None:
        return []

    names = model.names
    dets: list[Detection] = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id   = int(box.cls[0])
        cls_name = names.get(cls_id, f"cls_{cls_id}")
        dets.append(Detection(
            cls_name=cls_name, cls_id=cls_id,
            conf=float(box.conf[0]),
            x1=x1, y1=y1, x2=x2, y2=y2,
        ))

    dets.sort(key=lambda d: d.threat_level, reverse=True)
    return dets


# ---------------------------------------------------------------------------
# Рабочий поток
# ---------------------------------------------------------------------------

class DetectionWorker(QThread):
    """
    QThread: захват видео + инференс YOLO.

    Поддерживает:
      - pause() / resume() без перезапуска потока
      - пропуск кадров (PROCESS_EVERY_N_FRAMES) для повышения FPS
      - любой источник: файл, RTSP, веб-камера, изображение
    """

    frame_ready    = pyqtSignal(np.ndarray, list, float)
    error_occurred = pyqtSignal(str)
    stream_ended   = pyqtSignal()

    _IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, source, model, conf: float = CONFIDENCE_THRESHOLD, parent=None):
        super().__init__(parent)
        self.source = source
        self.model  = model
        self.conf   = conf

        self._running  = True
        self._paused   = False
        self._mutex    = QMutex()
        self._fps_buf: list[float] = []
        self._skip_ctr = 0

        # Последние детекции — показываем на пропущенных кадрах
        self._last_dets: list[Detection] = []

    # ── Пауза ────────────────────────────────────────────────────────────

    def pause(self) -> None:
        with QMutexLocker(self._mutex):
            self._paused = True

    def resume(self) -> None:
        with QMutexLocker(self._mutex):
            self._paused = False

    def toggle_pause(self) -> bool:
        """Переключает паузу. Возвращает True если теперь на паузе."""
        with QMutexLocker(self._mutex):
            self._paused = not self._paused
            return self._paused

    @property
    def is_paused(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._paused

    def stop(self) -> None:
        self._running = False
        self.wait(4000)

    # ── Основной цикл ─────────────────────────────────────────────────────

    def run(self) -> None:
        src = self.source

        # Одиночное изображение
        if isinstance(src, (str, Path)):
            if Path(str(src)).suffix.lower() in self._IMAGE_EXT:
                self._process_image(str(src))
                return

        cap = self._open_capture(src)
        if cap is None:
            return

        while self._running:
            # Пауза — ждём не блокируя поток навсегда
            if self.is_paused:
                self.msleep(50)
                continue

            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                self.stream_ended.emit()
                break

            # Пропуск кадров для повышения FPS
            self._skip_ctr += 1
            if self._skip_ctr >= PROCESS_EVERY_N_FRAMES:
                self._skip_ctr = 0
                if self.model is not None:
                    self._last_dets = run_inference(self.model, frame, self.conf)

            fps = self._update_fps(time.perf_counter() - t0)
            self.frame_ready.emit(frame, list(self._last_dets), fps)

        cap.release()

    # ── Вспомогательные ──────────────────────────────────────────────────

    def _process_image(self, path: str) -> None:
        frame = cv2.imread(path)
        if frame is None:
            self.error_occurred.emit(f"Не удалось открыть: {path}")
            return
        dets = run_inference(self.model, frame, self.conf) if self.model else []
        self.frame_ready.emit(frame, dets, 0.0)
        self.stream_ended.emit()

    def _open_capture(self, source) -> Optional[cv2.VideoCapture]:
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(str(source))

        if not cap.isOpened():
            self.error_occurred.emit(f"Не удалось открыть источник: {source}")
            return None

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def _update_fps(self, elapsed: float) -> float:
        self._fps_buf.append(1.0 / max(elapsed, 1e-6))
        if len(self._fps_buf) > FPS_HISTORY_LEN:
            self._fps_buf.pop(0)
        return sum(self._fps_buf) / len(self._fps_buf)
