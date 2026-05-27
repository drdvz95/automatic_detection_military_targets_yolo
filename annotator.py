"""
annotator.py
============
All OpenCV drawing logic — frame annotation.

No Qt imports. Takes a numpy BGR array, returns a numpy BGR array.
Stateless drawing functions + cached scanline overlay.

Key behaviour:
  - Strike zone markers (×) are shown ONLY when bbox area >= ZONE_MIN_BBOX_AREA.
    This creates a natural distance gate: distant targets get no markers,
    close/medium targets get them. Tune ZONE_MIN_BBOX_AREA in config.py.
  - No bbox fill — outline only (corner brackets + dashed rect).
  - Simple white + crosshair at screen centre.
  - Scanline overlay for optical sight effect.
  - No vignette.

──────────────────────────────────────────────────────────────────
TUNING CONSTANTS (change in this file):

  CROSSHAIR_ARM  = 40   — half-length of crosshair lines (pixels)
  CROSSHAIR_GAP  = 12   — dead-zone radius around centre (pixels)
  ZONE_CROSS_SZ  = 7    — half-size of strike zone × marker (pixels)

Distance gate lives in config.py → ZONE_MIN_BBOX_AREA
──────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from config import CLASS_COLORS_BGR, VULNERABILITY_ZONES, ZONE_MIN_BBOX_AREA
from detector import Detection

# ---------------------------------------------------------------------------
# Module-level tuning constants
# ---------------------------------------------------------------------------
CROSSHAIR_ARM  = 40    # ← crosshair line half-length (pixels)
CROSSHAIR_GAP  = 12    # ← gap from centre to line start (pixels)
ZONE_CROSS_SZ  = 7     # ← × marker half-size (pixels)


class Annotator:
    """
    Frame annotator. Instantiated once, reused every frame.
    Caches the scanline overlay per frame resolution.
    """

    def __init__(self) -> None:
        self._scanline_cache: dict[tuple[int, int], np.ndarray] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        detections: Sequence[Detection],
    ) -> np.ndarray:
        """
        Apply all visual layers and return an annotated copy.

        Layer order:
          1. Detection boxes + strike zones (distance-gated)
          2. Central crosshair
          3. Corner reticle brackets
          4. Scanlines
        """
        out = frame.copy()

        for det in detections:
            self._draw_detection(out, det)

        self._draw_crosshair(out)
        self._draw_reticle_border(out)
        out = self._apply_scanlines(out)

        return out

    # ── Detection ─────────────────────────────────────────────────────────

    def _draw_detection(self, img: np.ndarray, det: Detection) -> None:
        """
        Draw one detection:
          - Corner brackets (no fill)
          - Dashed outline
          - Class label + confidence
          - Strike zone × markers (only if close enough)
        """
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        color = CLASS_COLORS_BGR.get(det.cls_name, CLASS_COLORS_BGR["default"])
        bw, bh = det.bbox_w, det.bbox_h

        # Dashed outline (subtle)
        self._dashed_rect(img, x1, y1, x2, y2, color)

        # Corner brackets — primary bbox visual
        corner = max(14, min(bw, bh) // 6)
        for (ox, oy, sx, sy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                   (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(img, (ox, oy), (ox + sx * corner, oy),     color, 2, cv2.LINE_AA)
            cv2.line(img, (ox, oy), (ox, oy + sy * corner),     color, 2, cv2.LINE_AA)

        # Label
        self._draw_label(img, det, x1, y1, color)

        # ── Distance gate ─────────────────────────────────────────────────
        # Only draw strike markers when the target is large enough in frame,
        # i.e. close enough to matter tactically.
        if det.area >= ZONE_MIN_BBOX_AREA:
            zones = VULNERABILITY_ZONES.get(det.cls_name, [])
            for zone in zones:
                cx = int(x1 + zone["rx"] * bw)
                cy = int(y1 + zone["ry"] * bh)
                self._draw_zone_marker(img, cx, cy)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _dashed_rect(
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: tuple,
        dash: int = 8,
        gap:  int = 7,
    ) -> None:
        def seg(pt1, pt2):
            lx = pt2[0] - pt1[0]
            ly = pt2[1] - pt1[1]
            ln = math.hypot(lx, ly)
            if ln == 0:
                return
            ux, uy = lx / ln, ly / ln
            pos, draw = 0.0, True
            while pos < ln:
                end = min(pos + (dash if draw else gap), ln)
                if draw:
                    p1 = (int(pt1[0] + ux * pos), int(pt1[1] + uy * pos))
                    p2 = (int(pt1[0] + ux * end), int(pt1[1] + uy * end))
                    cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
                pos, draw = end, not draw

        seg((x1, y1), (x2, y1))
        seg((x2, y1), (x2, y2))
        seg((x2, y2), (x1, y2))
        seg((x1, y2), (x1, y1))

    @staticmethod
    def _draw_label(
        img: np.ndarray,
        det: Detection,
        x1: int, y1: int,
        color: tuple,
    ) -> None:
        font  = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.50
        thick = 1
        tag   = f"{det.cls_name.upper()}  {det.conf:.0%}"
        (tw, th), bl = cv2.getTextSize(tag, font, scale, thick)
        pad = 4
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1 - th - bl - pad * 2), (x1 + tw + pad * 2, y1), color, -1)
        cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
        cv2.putText(img, tag, (x1 + pad, y1 - bl - pad // 2),
                    font, scale, (0, 0, 0), thick, cv2.LINE_AA)

    @staticmethod
    def _draw_zone_marker(img: np.ndarray, cx: int, cy: int) -> None:
        """
        Small red × at strike zone centre.
        No rings, no labels — just the cross.

        Tune size via ZONE_CROSS_SZ at the top of this file.
        """
        COLOR = (30, 30, 220)   # BGR → red
        THICK = 2
        d = int(ZONE_CROSS_SZ * 0.707)
        cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), COLOR, THICK, cv2.LINE_AA)
        cv2.line(img, (cx + d, cy - d), (cx - d, cy + d), COLOR, THICK, cv2.LINE_AA)

    # ── Crosshair ─────────────────────────────────────────────────────────

    @staticmethod
    def _draw_crosshair(img: np.ndarray) -> None:
        """
        Simple white + crosshair at frame centre with a centre gap.

        Tune via module constants CROSSHAIR_ARM / CROSSHAIR_GAP.
        """
        h, w   = img.shape[:2]
        cx, cy = w // 2, h // 2
        C      = (255, 255, 255)  # white
        T      = 1

        cv2.line(img, (cx - CROSSHAIR_ARM, cy), (cx - CROSSHAIR_GAP, cy), C, T, cv2.LINE_AA)
        cv2.line(img, (cx + CROSSHAIR_GAP, cy), (cx + CROSSHAIR_ARM, cy), C, T, cv2.LINE_AA)
        cv2.line(img, (cx, cy - CROSSHAIR_ARM), (cx, cy - CROSSHAIR_GAP), C, T, cv2.LINE_AA)
        cv2.line(img, (cx, cy + CROSSHAIR_GAP), (cx, cy + CROSSHAIR_ARM), C, T, cv2.LINE_AA)

    # ── Reticle border ────────────────────────────────────────────────────

    @staticmethod
    def _draw_reticle_border(img: np.ndarray) -> None:
        """Corner brackets simulating the optical sight frame."""
        h, w   = img.shape[:2]
        color  = (40, 100, 40)
        length = 55
        thick  = 2
        pad    = 18

        for (x, y, sx, sy) in [
            (pad,     pad,      1,  1),
            (w - pad, pad,     -1,  1),
            (pad,     h - pad,  1, -1),
            (w - pad, h - pad, -1, -1),
        ]:
            cv2.line(img, (x, y), (x + sx * length, y),        color, thick, cv2.LINE_AA)
            cv2.line(img, (x, y), (x, y + sy * length),        color, thick, cv2.LINE_AA)

    # ── Scanlines ─────────────────────────────────────────────────────────

    def _apply_scanlines(self, img: np.ndarray) -> np.ndarray:
        """Horizontal scanlines simulating electro-optical sight effect."""
        h, w = img.shape[:2]
        key  = (h, w)
        if key not in self._scanline_cache:
            ov = np.zeros((h, w, 3), dtype=np.uint8)
            for y in range(0, h, 3):
                cv2.line(ov, (0, y), (w, y), (0, 0, 0), 1)
            self._scanline_cache[key] = ov
        return cv2.addWeighted(img, 1.0, self._scanline_cache[key], 0.09, 0)