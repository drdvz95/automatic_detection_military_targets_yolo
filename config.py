"""
config.py
=========
Central configuration for the armored vehicle detection system.

To add a new vehicle class:
  1. CLASS_COLORS_BGR      — bbox color
  2. VULNERABILITY_ZONES   — strike zones
  3. AMMO_TABLE            — recommended ammunition
  4. THREAT_PRIORITY       — threat level
  No other files need to be changed.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
ASSETS_DIR      = BASE_DIR / "assets"
AMMO_IMAGES_DIR = ASSETS_DIR / "ammo"   # put ammo photos here
DEFAULT_WEIGHTS = BASE_DIR / "weights" / "best.pt"

# ---------------------------------------------------------------------------
# Detection parameters
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD   = 0.10
FPS_HISTORY_LEN        = 30
TARGET_FPS             = 15

# ---------------------------------------------------------------------------
# FPS boost
# ---------------------------------------------------------------------------
# Process every N-th frame (1 = every frame, 2 = every other → ~+40% FPS).
# Change here if you need more FPS at the cost of slightly delayed detection.
PROCESS_EVERY_N_FRAMES = 2

# ---------------------------------------------------------------------------
# Zone marker distance threshold
# ---------------------------------------------------------------------------
# Strike zone markers (×) are shown only when the target bbox area
# exceeds this pixel threshold — i.e. the target is close enough.
#
# How to tune:
#   - Small value  → markers appear even on distant/small targets
#   - Large value  → markers only appear when target is very close
#
# Default 4000 px² ≈ bbox of ~63×63 px, which roughly corresponds to
# a tank at medium engagement range on a 1080p feed.
#
# CHANGE HERE:
ZONE_MIN_BBOX_AREA = 4000

# ---------------------------------------------------------------------------
# Bbox colors (BGR for OpenCV) — unique color per class
# ---------------------------------------------------------------------------
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "car":              (160, 160, 160),   # grey
    "truck":            (180, 130, 80),    # steel
    "military_truck":   (40,  170, 80),    # army green
    "armored_vehicle":  (40,  180, 220),   # cyan
    "tank":             (50,  50,  230),   # blue
    "artillery":        (180, 40,  210),   # purple
    "person":           (40,  220, 120),   # green
    "explosion":        (30,  30,  255),   # red
    "default":          (80,  200, 80),
}

# ---------------------------------------------------------------------------
# Vulnerability zones
# ---------------------------------------------------------------------------
# Positioning logic based on real vehicle layouts:
#
#   TANK (T-72/80/90 — rear engine):
#     Engine bay (MTO) → rx=0.78 (rear, not at edge)
#     Ammo rack  (BK)  → rx=0.44, ry=0.28 (turret, upper)
#
#   ARMORED_VEHICLE (BTR-80 — front engine):
#     Engine bay → rx=0.22 (nose)
#     Ammo rack  → rx=0.55, ry=0.30 (turret)
#
#   MILITARY_TRUCK:
#     Engine only → rx=0.17 (front hood)
#
#   ARTILLERY:
#     Engine bay → rx=0.78 (rear)
#     Breech     → rx=0.44, ry=0.26 (gun breech)
#
#   PERSON: centre of mass
#
# All rx,ry kept in [0.15 .. 0.85] — never at the very edge of bbox.
# Markers only show when bbox area >= ZONE_MIN_BBOX_AREA (distance gate).

VULNERABILITY_ZONES: dict[str, list[dict]] = {

    "tank": [
        {"name": "Engine", "rx": 0.78, "ry": 0.62},
        {"name": "Ammo",   "rx": 0.44, "ry": 0.28},
    ],

    "armored_vehicle": [
        {"name": "Engine", "rx": 0.22, "ry": 0.58},
        {"name": "Ammo",   "rx": 0.55, "ry": 0.30},
    ],

    "military_truck": [
        {"name": "Engine", "rx": 0.17, "ry": 0.52},
    ],

    "artillery": [
        {"name": "Engine", "rx": 0.78, "ry": 0.64},
        {"name": "Breech", "rx": 0.44, "ry": 0.26},
    ],

    "person": [
        {"name": "CoM", "rx": 0.50, "ry": 0.42},
    ],

    # No strike zones — observation only
    "car":       [],
    "truck":     [],
    "explosion": [],
}

# ---------------------------------------------------------------------------
# Ammunition table — short designations only
# ---------------------------------------------------------------------------
AMMO_TABLE: dict[str, dict[str, str]] = {
    "tank":             {"primary": "3BM60",       "note": "Front: 3BM60. Side/rear: 3BK18M."},
    "armored_vehicle":  {"primary": "3OF26",       "note": "Light armour. HE-FRAG sufficient."},
    "military_truck":   {"primary": "3OF26",       "note": "Engine bay or cargo."},
    "artillery":        {"primary": "3OF26",       "note": "Suppress before it fires."},
    "car":              {"primary": "Observation", "note": "Identify before engaging."},
    "truck":            {"primary": "3OF26",       "note": "Logistics target."},
    "person":           {"primary": "3OF26",       "note": "Identify affiliation first."},
    "explosion":        {"primary": "— DANGER —",  "note": "Keep distance. Hold fire."},
}

AMMO_DEFAULT: dict[str, str] = {
    "primary": "—",
    "note":    "Target not identified",
}

# ---------------------------------------------------------------------------
# Ammo photos (place JPG files in assets/ammo/)
# ---------------------------------------------------------------------------
# Filename must match exactly (case-sensitive on Linux).
AMMO_IMAGES: dict[str, str] = {
    "3BM60":       "3bm60.jpg",
    "3OF26":       "3of26.jpg",
    "3BK18M":      "3bk18m.jpg",
    "Observation": "",
    "— DANGER —":  "",
}

# ---------------------------------------------------------------------------
# Threat priority (used for sorting when multiple targets in frame)
# ---------------------------------------------------------------------------
THREAT_PRIORITY: dict[str, int] = {
    "tank":             10,
    "artillery":         9,
    "armored_vehicle":   8,
    "military_truck":    6,
    "explosion":         7,
    "person":            5,
    "truck":             3,
    "car":               1,
}

# ---------------------------------------------------------------------------
# Qt stylesheet — T-90M gunner interface theme
# ---------------------------------------------------------------------------
QSS_THEME = """
QMainWindow, QDialog { background-color: #06090a; }
QWidget {
    background-color: #06090a;
    color: #c8e8b0;
    font-family: 'Courier New', 'Courier', 'DejaVu Sans Mono', monospace;
    font-size: 11px;
}
QPushButton {
    background-color: #0a1410;
    color: #8acc6a;
    border: 1px solid #244818;
    border-radius: 2px;
    padding: 5px 14px;
    letter-spacing: 1px;
    font-size: 11px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover  { background-color: #122418; border-color: #8acc6a; color: #ccffaa; }
QPushButton:pressed { background-color: #244818; }
QPushButton:disabled { color: #1a3010; border-color: #0e1a0c; }
QLineEdit {
    background-color: #080e0c;
    color: #8acc6a;
    border: 1px solid #1a3010;
    border-radius: 2px;
    padding: 3px 8px;
    font-family: 'Courier New', monospace;
    selection-background-color: #244818;
}
QLineEdit:focus { border-color: #8acc6a; }
QTextEdit {
    background-color: #040604;
    color: #3a7a28;
    border: 1px solid #102010;
    font-family: 'Courier New', monospace;
    font-size: 10px;
}
QScrollBar:vertical { background: #080e0c; width: 6px; border: none; margin: 0; }
QScrollBar::handle:vertical { background: #244818; border-radius: 3px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QProgressBar {
    background-color: #040604;
    border: 1px solid #1a3010;
    border-radius: 1px;
    color: transparent;
}
QProgressBar::chunk { background-color: #8acc6a; }
QSplitter::handle { background-color: #1a3010; width: 2px; height: 2px; }
QToolTip {
    background-color: #0a1410;
    color: #8acc6a;
    border: 1px solid #244818;
    font-size: 10px;
    padding: 4px;
}
QFrame[role="panel"] { border: 1px solid #1a3010; background-color: #040804; border-radius: 1px; }
"""