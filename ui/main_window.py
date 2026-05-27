"""
ui/main_window.py
=================
Main window — orchestrates all components.

Fixes vs previous revision:
  - _log_widget renamed to avoid conflict with _log() method
  - update_detections() call updated (no longer passes unique_classes)
  - English UI text throughout
  - TXT tactical report on screenshot
  - Pause support
"""

from __future__ import annotations

import datetime
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QFileDialog, QSplitter,
)

from annotator import Annotator
from config import DEFAULT_WEIGHTS, SCREENSHOTS_DIR, QSS_THEME
from detector import Detection, DetectionWorker, ModelLoader
from ui.info_panel import InfoPanel
from ui.video_panel import VideoPanel
from ui.widgets import LogWidget, SectionLabel


class MainWindow(QMainWindow):
    """Main application window. Style: T-90M gunner-operator interface."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "ARMORED VEHICLE DETECTION SYSTEM"
        )
        self.setMinimumSize(1280, 780)
        self.resize(1520, 900)

        # State
        self._model:         object | None          = None
        self._worker:        DetectionWorker | None = None
        self._annotator:     Annotator              = Annotator()
        self._current_frame: np.ndarray | None      = None
        self._current_dets:  list[Detection]        = []
        self._unique_classes: set[str]              = set()

        self.setStyleSheet(QSS_THEME)
        self._build_ui()

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

        if DEFAULT_WEIGHTS.exists():
            self._load_model(str(DEFAULT_WEIGHTS))

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(5)

        root.addWidget(self._build_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self._video_panel = VideoPanel()
        self._info_panel  = InfoPanel()

        splitter.addWidget(self._video_panel)
        splitter.addWidget(self._info_panel)
        splitter.setSizes([1140, 360])
        splitter.setHandleWidth(3)
        root.addWidget(splitter, 1)

        root.addWidget(self._build_log_bar())

        # Signals
        self._video_panel.source_opened.connect(self._on_source_opened)
        self._video_panel.stream_stop.connect(self._stop_worker)
        self._video_panel.pause_toggled.connect(self._on_pause_toggled)
        self._info_panel.screenshot_requested.connect(self._take_screenshot)
        self._info_panel.model_load_requested.connect(self._load_model_dialog)

    def _build_header(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(38)
        bar.setStyleSheet(
            "QFrame{border:1px solid #244818;background:#040a04;border-radius:2px;}"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)

        self._clock_lbl = QLabel()
        self._clock_lbl.setStyleSheet(
            "font-size:12px;color:#4a8a38;font-weight:bold;background:transparent;"
        )
        self._clock_lbl.setFixedWidth(170)

        title = QLabel(
            "◈  ARMORED VEHICLE DETECTION AND CLASSIFICATION SYSTEM  //  T-90M GUNNER  ◈"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size:11px;color:#4a8a38;letter-spacing:2px;background:transparent;"
        )

        ver = QLabel("v1.0.0")
        ver.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        ver.setStyleSheet("font-size:10px;color:#2a5a1c;background:transparent;")
        ver.setFixedWidth(55)

        lay.addWidget(self._clock_lbl)
        lay.addWidget(title, 1)
        lay.addWidget(ver)
        return bar

    def _build_log_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(82)
        bar.setStyleSheet(
            "QFrame{border:1px solid #1a3010;background:#020402;border-radius:2px;}"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(10)

        hdr = SectionLabel("EVENT LOG")
        hdr.setFixedWidth(100)
        hdr.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Use a distinct attribute name to avoid shadowing the _log() method
        self._log_widget = LogWidget()

        lay.addWidget(hdr)
        lay.addWidget(self._log_widget, 1)
        return bar

    # ── Slots ─────────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_source_opened(self, source: str) -> None:
        self._stop_worker()
        self._log(f"Source: {source}")

        worker = DetectionWorker(source, self._model)
        worker.frame_ready.connect(self._on_frame)
        worker.error_occurred.connect(self._on_error)
        worker.stream_ended.connect(self._on_stream_ended)
        worker.start()
        self._worker = worker

    @pyqtSlot(np.ndarray, list, float)
    def _on_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        fps: float,
    ) -> None:
        self._current_frame = frame
        self._current_dets  = detections

        if detections:
            for d in detections:
                self._unique_classes.add(d.cls_name)

        annotated = self._annotator.draw(frame, detections)
        self._video_panel.show_frame(annotated)
        self._video_panel.fps_badge.set_fps(fps)

        # Updated call — no unique_classes argument
        self._info_panel.update_detections(detections)

        if detections:
            names = ", ".join(d.cls_name.upper() for d in detections[:3])
            if len(detections) > 3:
                names += f" +{len(detections) - 3}"
            self._log(f"Targets [{len(detections)}]: {names}")

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._log(f"ERROR: {msg}")
        self._video_panel.show_no_signal()

    @pyqtSlot()
    def _on_stream_ended(self) -> None:
        self._log("Stream ended")

    @pyqtSlot()
    def _on_pause_toggled(self) -> None:
        if self._worker:
            paused = self._worker.toggle_pause()
            self._video_panel.set_paused_state(paused)
            self._log("Paused" if paused else "Resumed")

    # ── Model ─────────────────────────────────────────────────────────────

    def _load_model_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO weights", ".", "Weights (*.pt)"
        )
        if path:
            self._load_model(path)

    def _load_model(self, path: str) -> None:
        try:
            self._model = ModelLoader.load(path)
            name = Path(path).name
            self._info_panel.set_model_status(f"OK  {name}", ok=True)
            self._log(f"Model loaded: {name}")
            if self._worker and self._worker.isRunning():
                self._worker.model = self._model
        except Exception as exc:
            self._info_panel.set_model_status("LOAD ERROR", ok=False)
            self._log(f"Model error: {exc}")

    # ── Screenshot + TXT report ───────────────────────────────────────────

    @pyqtSlot()
    def _take_screenshot(self) -> None:
        if self._current_frame is None:
            self._log("No frame available for screenshot")
            return

        annotated = self._annotator.draw(self._current_frame, self._current_dets)
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = SCREENSHOTS_DIR / f"detection_{ts}.jpg"
        txt_path = SCREENSHOTS_DIR / f"detection_{ts}.txt"

        cv2.imwrite(str(img_path), annotated)
        self._write_report(txt_path, ts)
        self._log(f"Screenshot: {img_path.name}  |  Report: {txt_path.name}")

    def _write_report(self, path: Path, ts: str) -> None:
        """Tactical text report saved alongside each screenshot."""
        dets   = self._current_dets
        counts = Counter(d.cls_name for d in dets)
        dt     = datetime.datetime.now()

        lines = [
            "=" * 60,
            "  TACTICAL REPORT  —  ARMORED VEHICLE DETECTION SYSTEM",
            "=" * 60,
            f"  Date / Time   : {dt.strftime('%d.%m.%Y  %H:%M:%S')}",
            f"  Callsign      : GUNNER-1",
            f"  Platform      : T-90M",
            f"  Image file    : detection_{ts}.jpg",
            "-" * 60,
            f"  TOTAL TARGETS IN FRAME: {len(dets)}",
            "-" * 60,
        ]

        if dets:
            lines.append("  TARGET COMPOSITION:")
            for cls_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                from config import AMMO_TABLE, AMMO_DEFAULT, THREAT_PRIORITY
                ammo  = AMMO_TABLE.get(cls_name, AMMO_DEFAULT)
                prio  = THREAT_PRIORITY.get(cls_name, 0)
                lines.append(
                    f"    {cls_name.upper():<22} x{cnt}"
                    f"   ammo: {ammo.get('primary','—'):<10}"
                    f"   priority: {prio}/10"
                )

            lines.append("-" * 60)
            lines.append("  DETAILED LIST:")
            for i, d in enumerate(dets, 1):
                lines.append(
                    f"    [{i:02d}]  {d.cls_name.upper():<22}"
                    f"  conf={d.conf:.0%}"
                    f"  bbox=({d.x1},{d.y1})-({d.x2},{d.y2})"
                )
        else:
            lines.append("  No targets detected in frame.")

        lines += [
            "-" * 60,
            f"  Unique target types (session): {len(self._unique_classes)}",
            "=" * 60,
            "  DOCUMENT GENERATED AUTOMATICALLY",
            "=" * 60,
        ]

        path.write_text("\n".join(lines), encoding="utf-8")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _stop_worker(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker = None

    def _update_clock(self) -> None:
        now = datetime.datetime.now()
        self._clock_lbl.setText(now.strftime("%H:%M:%S    %d.%m.%Y"))
        self._video_panel.set_time(now.strftime("%H:%M:%S"))

    def _log(self, msg: str) -> None:
        """Append a timestamped message to the event log widget."""
        self._log_widget.append_event(msg)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_worker()
        event.accept()