"""
ui/info_panel.py
================
Правая информационная панель.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QFrame, QSizePolicy,
)

from config import AMMO_DEFAULT, AMMO_IMAGES, AMMO_IMAGES_DIR, AMMO_TABLE
from detector import Detection
from ui.widgets import AlertIndicator, HLine, SectionLabel


class InfoPanel(QWidget):
    """Правая панель — идентификация цели, боеприпасы, индикаторы."""

    screenshot_requested = pyqtSignal()
    model_load_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._build()

    # ── Построение ─────────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 0, 0, 0)
        root.setSpacing(5)

        root.addWidget(self._build_alert_block())
        root.addWidget(self._build_target_block())
        root.addWidget(self._build_ammo_block())
        root.addWidget(self._build_counters_block())
        root.addWidget(self._build_model_block())

        # Кнопка скриншота (требование 3.2.4)
        self._btn_screenshot = QPushButton("◉  SAVE SCREENSHOT WITH ANALYTICS")
        self._btn_screenshot.setFixedHeight(36)
        self._btn_screenshot.setStyleSheet("""
            QPushButton {
                background: #0a1a0a; color: #8acc6a;
                border: 1px solid #4a8a3a; border-radius: 2px;
                font-size: 11px; letter-spacing: 1px;
                font-family: 'Courier New', monospace;
            }
            QPushButton:hover { background: #122418; border-color: #ccffaa; color: #ccffaa; }
            QPushButton:pressed { background: #244818; }
        """)
        self._btn_screenshot.clicked.connect(self.screenshot_requested)
        root.addWidget(self._btn_screenshot)
        root.addStretch()

    def _panel(self) -> QFrame:
        f = QFrame()
        f.setProperty("role", "panel")
        f.setStyleSheet(
            "QFrame[role='panel']{"
            "border:1px solid #1a3010;"
            "background-color:#040804;"
            "border-radius:1px;}"
        )
        return f

    # ── Блоки ────────────────────────────────────────────────────────────

    def _build_alert_block(self) -> QWidget:
        """Индикатор тревоги (требование 3.2.5)."""
        w   = self._panel()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(12)

        self.indicator = AlertIndicator()

        right = QVBoxLayout()
        right.setSpacing(2)

        self._alert_status = QLabel("NO TARGETS")
        self._alert_status.setStyleSheet(
            "font-size:15px;color:#4acc3a;font-weight:bold;"
            "letter-spacing:2px;background:transparent;"
        )
        self._alert_sub = QLabel("Observation zone clear")
        self._alert_sub.setStyleSheet(
            "font-size:9px;color:#3a6a28;background:transparent;"
        )
        right.addWidget(self._alert_status)
        right.addWidget(self._alert_sub)

        lay.addWidget(self.indicator)
        lay.addLayout(right)
        lay.addStretch()
        return w

    def _build_target_block(self) -> QWidget:
        """Идентификация цели."""
        w   = self._panel()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(5)

        lay.addWidget(SectionLabel("TARGET IDENTIFICATION"))

        self._cls_label = QLabel("—")
        self._cls_label.setStyleSheet(
            "font-size:24px;color:#8acc6a;font-weight:bold;"
            "letter-spacing:2px;background:transparent;"
        )
        lay.addWidget(self._cls_label)

        # Шкала уверенности
        conf_row = QHBoxLayout()
        conf_row.setSpacing(6)
        lbl = QLabel("CONF:")
        lbl.setStyleSheet("font-size:9px;color:#3a6a28;background:transparent;")
        lbl.setFixedWidth(36)

        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setValue(0)
        self._conf_bar.setTextVisible(False)
        self._conf_bar.setFixedHeight(5)

        self._conf_val = QLabel("0%")
        self._conf_val.setStyleSheet(
            "font-size:9px;color:#6aaa4a;min-width:30px;background:transparent;"
        )
        conf_row.addWidget(lbl)
        conf_row.addWidget(self._conf_bar, 1)
        conf_row.addWidget(self._conf_val)
        lay.addLayout(conf_row)

        return w

    def _build_ammo_block(self) -> QWidget:
        """Рекомендуемые боеприпасы + фото (требования 3.2.2, 3.1.3)."""
        w   = self._panel()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        lay.addWidget(SectionLabel("AMMUNITION"))

        self._ammo_primary = QLabel("—")
        self._ammo_primary.setStyleSheet(
            "font-size:16px;color:#ccff99;font-weight:bold;"
            "background:transparent;letter-spacing:1px;"
        )
        lay.addWidget(self._ammo_primary)

        self._ammo_note = QLabel("Target not identified")
        self._ammo_note.setStyleSheet(
            "font-size:10px;color:#4a7a30;background:transparent;"
        )
        self._ammo_note.setWordWrap(True)
        lay.addWidget(self._ammo_note)

        lay.addWidget(HLine())

        # Фото боеприпаса
        self._ammo_img_lbl = QLabel()
        self._ammo_img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ammo_img_lbl.setFixedHeight(90)
        self._ammo_img_lbl.setStyleSheet(
            "background:#020402;border:1px solid #1a3010;"
        )
        self._ammo_img_lbl.setText("[ ammo photo ]")
        lay.addWidget(self._ammo_img_lbl)

        hint = QLabel("assets/ammo/  ←  place photos here")
        hint.setStyleSheet("font-size:8px;color:#244818;background:transparent;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(hint)

        return w

    def _build_counters_block(self) -> QWidget:
        """Счетчик целей — только 'в кадре' (счетчик типов удален)."""
        w   = self._panel()
        w.setFixedHeight(56)
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 6, 12, 6)

        self._cnt_now = self._counter(lay, "IN FRAME")
        return w

    def _counter(self, parent_layout, title: str) -> QLabel:
        col = QVBoxLayout()
        col.setSpacing(1)
        t = QLabel(title)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet("font-size:8px;color:#3a6a28;background:transparent;")
        v = QLabel("0")
        v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.setStyleSheet(
            "font-size:20px;color:#8acc6a;font-weight:bold;background:transparent;"
        )
        col.addWidget(t)
        col.addWidget(v)
        parent_layout.addLayout(col)
        return v

    def _build_model_block(self) -> QWidget:
        """Статус модели."""
        w   = self._panel()
        w.setFixedHeight(52)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(4)

        lay.addWidget(SectionLabel("YOLO MODEL"))

        row = QHBoxLayout()
        self._model_status = QLabel("NOT LOADED")
        self._model_status.setStyleSheet(
            "font-size:10px;color:#cc5a2a;background:transparent;"
        )
        btn = QPushButton("[ LOAD ]")
        btn.setFixedHeight(22)
        btn.clicked.connect(self.model_load_requested)
        row.addWidget(self._model_status, 1)
        row.addWidget(btn)
        lay.addLayout(row)
        return w

    # ── Обновление данных ───────────────────────────────────────────────────────

    def update_detections(self, detections: list[Detection]) -> None:
        has = len(detections) > 0
        self.indicator.set_target(has)

        if has:
            self._alert_status.setText(f"TARGET ACQUIRED  ×{len(detections)}")
            self._alert_status.setStyleSheet(
                "font-size:14px;color:#ff4a4a;font-weight:bold;"
                "letter-spacing:2px;background:transparent;"
            )
            self._alert_sub.setText("Identify and engage")
        else:
            self._alert_status.setText("NO TARGETS")
            self._alert_status.setStyleSheet(
                "font-size:15px;color:#4acc3a;font-weight:bold;"
                "letter-spacing:2px;background:transparent;"
            )
            self._alert_sub.setText("Observation zone clear")

        self._cnt_now.setText(str(len(detections)))

        if detections:
            best = detections[0]   # отсортировано по уровню угрозы в detector.py
            self._cls_label.setText(best.cls_name.upper())
            conf_pct = int(best.conf * 100)
            self._conf_bar.setValue(conf_pct)
            self._conf_val.setText(f"{conf_pct}%")

            ammo = AMMO_TABLE.get(best.cls_name, AMMO_DEFAULT)
            primary = ammo.get("primary", "—")
            self._ammo_primary.setText(primary)
            self._ammo_note.setText(ammo.get("note", ""))
            self._load_ammo_image(primary)
        else:
            self._cls_label.setText("—")
            self._conf_bar.setValue(0)
            self._conf_val.setText("0%")
            self._ammo_primary.setText(AMMO_DEFAULT["primary"])
            self._ammo_note.setText(AMMO_DEFAULT["note"])
            self._ammo_img_lbl.setPixmap(QPixmap())
            self._ammo_img_lbl.setText("[ ammo photo ]")

    def _load_ammo_image(self, ammo_name: str) -> None:
        img_file = AMMO_IMAGES.get(ammo_name, "")
        if not img_file:
            self._ammo_img_lbl.setPixmap(QPixmap())
            self._ammo_img_lbl.setText("[ no photo ]")
            return

        path = AMMO_IMAGES_DIR / img_file
        if not path.exists():
            self._ammo_img_lbl.setPixmap(QPixmap())
            self._ammo_img_lbl.setText(f"[ {img_file} not found ]")
            return

        pix = QPixmap(str(path))
        if pix.isNull():
            self._ammo_img_lbl.setText("[ load error ]")
            return

        self._ammo_img_lbl.setPixmap(
            pix.scaled(
                self._ammo_img_lbl.width() - 4,
                self._ammo_img_lbl.height() - 4,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_model_status(self, text: str, ok: bool = False) -> None:
        color = "#8acc6a" if ok else "#cc5a2a"
        self._model_status.setText(text)
        self._model_status.setStyleSheet(
            f"font-size:10px;color:{color};background:transparent;"
        )
