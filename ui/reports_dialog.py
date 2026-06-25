"""
ui/reports_dialog.py
=====================
In-app viewer for saved screenshots + tactical TXT reports.

Why this exists:
  When the app runs inside a Docker container, there is no guarantee that
  a host file manager is reachable to "open the folder" — the container
  has its own filesystem namespace. Rendering the report list and content
  *inside* the Qt application sidesteps that entirely: as long as
  SCREENSHOTS_DIR is correctly volume-mounted (see launch script), this
  dialog will show exactly what's on disk, identically whether the app
  is run directly or via `docker run`.

Layout:
  Left  — list of all detection_*.jpg / detection_*.txt pairs, newest first
  Right — image preview (top) + report text (bottom)

A "Show in file manager" button is also included as a convenience for
non-Docker runs; it silently does nothing useful inside a container
(xdg-open has no host file manager to hand off to), so it is not relied
upon as the primary way to view reports.
"""

from __future__ import annotations

import re
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QLabel, QPushButton, QTextEdit, QSplitter, QWidget, QMessageBox,
)

from config import SCREENSHOTS_DIR

_TS_RE = re.compile(r"detection_(\d{8}_\d{6})")


class ReportsDialog(QDialog):
    """Browse every screenshot + TXT report saved by the app."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAVED REPORTS")
        self.setMinimumSize(900, 560)
        self.setStyleSheet(
            "QDialog{background:#06090a;}"
            "QListWidget{background:#040804;color:#8acc6a;"
            "border:1px solid #1a3010;font-family:'Courier New',monospace;font-size:11px;}"
            "QListWidget::item{padding:5px;}"
            "QListWidget::item:selected{background:#1a3010;color:#ccffaa;}"
            "QTextEdit{background:#040604;color:#8acc6a;border:1px solid #1a3010;"
            "font-family:'Courier New',monospace;font-size:11px;}"
            "QLabel{color:#8acc6a;background:transparent;}"
        )
        self._build()
        self._reload()

    # ── Build ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top_bar = QHBoxLayout()
        path_lbl = QLabel(f"Folder: {SCREENSHOTS_DIR.resolve()}")
        path_lbl.setStyleSheet("font-size:10px;color:#3a6a28;")
        btn_refresh = QPushButton("[ REFRESH ]")
        btn_refresh.clicked.connect(self._reload)
        btn_open_fm = QPushButton("[ SHOW IN FILE MANAGER ]")
        btn_open_fm.clicked.connect(self._open_in_file_manager)
        top_bar.addWidget(path_lbl, 1)
        top_bar.addWidget(btn_refresh)
        top_bar.addWidget(btn_open_fm)
        root.addLayout(top_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._list = QListWidget()
        self._list.setFixedWidth(260)
        self._list.currentItemChanged.connect(self._on_select)
        splitter.addWidget(self._list)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(6)

        self._preview = QLabel("Select a report on the left")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(260)
        self._preview.setStyleSheet(
            "background:#020402;border:1px solid #1a3010;color:#1a4010;"
        )
        rlay.addWidget(self._preview, 1)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFixedHeight(220)
        rlay.addWidget(self._text)

        splitter.addWidget(right)
        splitter.setSizes([260, 600])
        root.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        btn_delete = QPushButton("[ DELETE SELECTED ]")
        btn_delete.setStyleSheet("color:#cc5a2a;")
        btn_delete.clicked.connect(self._delete_selected)
        btn_close = QPushButton("[ CLOSE ]")
        btn_close.clicked.connect(self.accept)
        bottom.addWidget(btn_delete)
        bottom.addStretch()
        bottom.addWidget(btn_close)
        root.addLayout(bottom)

    # ── Data ─────────────────────────────────────────────────────────────

    def _reload(self) -> None:
        self._list.clear()
        SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

        jpgs = sorted(
            SCREENSHOTS_DIR.glob("detection_*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not jpgs:
            item = QListWidgetItem("(no reports saved yet)")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._list.addItem(item)
            self._preview.setText("No reports yet.\nUse SAVE SCREENSHOT WITH ANALYTICS first.")
            self._text.clear()
            return

        for jpg in jpgs:
            m = _TS_RE.search(jpg.stem)
            label = m.group(1) if m else jpg.stem
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, jpg)
            self._list.addItem(item)

        self._list.setCurrentRow(0)

    def _on_select(self, current: QListWidgetItem, _previous) -> None:
        if current is None:
            return
        jpg: Path | None = current.data(Qt.ItemDataRole.UserRole)
        if jpg is None:
            return

        pix = QPixmap(str(jpg))
        if not pix.isNull():
            self._preview.setPixmap(
                pix.scaled(
                    self._preview.width() - 4,
                    self._preview.height() - 4,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._preview.setText("[ image load error ]")

        txt_path = jpg.with_suffix(".txt")
        if txt_path.exists():
            self._text.setPlainText(txt_path.read_text(encoding="utf-8"))
        else:
            self._text.setPlainText("(no matching .txt report found)")

    # ── Actions ──────────────────────────────────────────────────────────

    def _open_in_file_manager(self) -> None:
        """
        Best-effort convenience action. Inside a Docker container this
        cannot actually hand off to a host file manager process, so it
        is informational only when it fails — the dialog above remains
        the reliable way to view reports in every run mode.
        """
        ok = QDesktopServices.openUrl(QUrl.fromLocalFile(str(SCREENSHOTS_DIR.resolve())))
        if not ok:
            QMessageBox.information(
                self,
                "File manager unavailable",
                "Could not hand off to a file manager from inside this "
                "environment. Use the list on the left to browse reports instead.",
            )

    def _delete_selected(self) -> None:
        current = self._list.currentItem()
        if current is None:
            return
        jpg: Path | None = current.data(Qt.ItemDataRole.UserRole)
        if jpg is None:
            return

        reply = QMessageBox.question(
            self, "Delete report",
            f"Delete {jpg.name} and its .txt report?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        jpg.unlink(missing_ok=True)
        jpg.with_suffix(".txt").unlink(missing_ok=True)
        self._reload()
