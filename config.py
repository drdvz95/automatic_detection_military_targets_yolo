"""
config.py
=========
Центральная конфигурация системы обнаружения бронетехники.

Чтобы добавить новый класс техники:
  1. CLASS_COLORS_BGR      — цвет bbox (ограничивающей рамки)
  2. VULNERABILITY_ZONES   — зоны поражения
  3. AMMO_TABLE            — рекомендуемые боеприпасы
  4. THREAT_PRIORITY       — уровень угрозы
  Никакие другие файлы изменять не нужно.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
ASSETS_DIR      = BASE_DIR / "assets"
AMMO_IMAGES_DIR = ASSETS_DIR / "ammo"   # поместите сюда фотографии боеприпасов
DEFAULT_WEIGHTS = BASE_DIR / "weights" / "best.pt"

# ---------------------------------------------------------------------------
# Параметры обнаружения
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD   = 0.08
FPS_HISTORY_LEN        = 30
TARGET_FPS             = 15

# ---------------------------------------------------------------------------
# Увеличение FPS
# ---------------------------------------------------------------------------
# Обрабатывать каждый N-й кадр (1 = каждый кадр, 2 = через один → ~+40% FPS).
# Измените это значение, если вам нужно больше FPS ценой небольшой задержки обнаружения.
PROCESS_EVERY_N_FRAMES = 2

# ---------------------------------------------------------------------------
# Порог расстояния для маркеров зон
# ---------------------------------------------------------------------------
# Маркеры зон поражения (×) отображаются только тогда, когда площадь bbox цели
# превышает этот пиксельный порог — т.е. цель находится достаточно близко.
#
# Как настраивать:
#   - Малое значение  → маркеры появляются даже на далеких/мелких целях
#   - Большое значение  → маркеры появляются только когда цель очень близко
#
# По умолчанию 4000 px² ≈ bbox размером ~63×63 px, что примерно соответствует
# танку на средней дистанции боя при разрешении 1080p.
#
# ИЗМЕНЯТЬ ЗДЕСЬ:
ZONE_MIN_BBOX_AREA = 4000

# ---------------------------------------------------------------------------
# Цвета ограничивающих рамок (BGR для OpenCV) — уникальный цвет для каждого класса
# ---------------------------------------------------------------------------
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "car":              (160, 160, 160),   # серый
    "truck":            (180, 130, 80),    # стальной
    "military_truck":   (40,  170, 80),    # армейский зеленый
    "armored_vehicle":  (40,  180, 220),   # голубой
    "tank":             (50,  50,  230),   # синий
    "artillery":        (180, 40,  210),   # фиолетовый
    "person":           (40,  220, 120),   # зеленый
    "explosion":        (30,  30,  255),   # красный
    "default":          (80,  200, 80),
}

# ---------------------------------------------------------------------------
# Зоны уязвимости
# ---------------------------------------------------------------------------
# Логика позиционирования основана на компоновке реальных машин:
#
#   ТАНК (Т-72/80/90 — заднее расположение двигателя):
#     МТО (моторно-трансмиссионное отделение) → rx=0.78 (сзади, не с самого края)
#     БК (боекомплект) → rx=0.44, ry=0.28 (башня, сверху)
#
#   БРОНЕТЕХНИКА (БТР-80 — переднее расположение двигателя):
#     МТО → rx=0.22 (нос)
#     БК  → rx=0.55, ry=0.30 (башня)
#
#   ВОЕННЫЙ ГРУЗОВИК:
#     Только двигатель → rx=0.17 (передний капот)
#
#   АРТИЛЛЕРИЯ:
#     МТО → rx=0.78 (сзади)
#     Казенник → rx=0.44, ry=0.26 (казенная часть орудия)
#
#   ЧЕЛОВЕК: центр масс
#
# Все rx, ry находятся в пределах [0.15 .. 0.85] — никогда не на самом краю bbox.
# Маркеры отображаются только когда площадь bbox >= ZONE_MIN_BBOX_AREA (ограничение по дистанции).

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

    # Нет зон поражения — только наблюдение
    "car":       [],
    "truck":     [],
    "explosion": [],
}

# ---------------------------------------------------------------------------
# Таблица боеприпасов — только краткие обозначения
# ---------------------------------------------------------------------------
AMMO_TABLE: dict[str, dict[str, str]] = {
    "tank":             {"primary": "3BM60",       "note": "В лоб: 3БМ60. В борт/корму: 3БК18М."},
    "armored_vehicle":  {"primary": "3OF26",       "note": "Легкая броня. ОФС достаточно."},
    "military_truck":   {"primary": "3OF26",       "note": "МТО или груз."},
    "artillery":        {"primary": "3OF26",       "note": "Подавить до открытия огня."},
    "car":              {"primary": "Observation", "note": "Идентифицировать перед поражением."},
    "truck":            {"primary": "3OF26",       "note": "Цель логистики."},
    "person":           {"primary": "3OF26",       "note": "Сначала определить принадлежность."},
    "explosion":        {"primary": "— ОПАСНОСТЬ —",  "note": "Держать дистанцию. Не стрелять."},
}

AMMO_DEFAULT: dict[str, str] = {
    "primary": "—",
    "note":    "Цель не идентифицирована",
}

# ---------------------------------------------------------------------------
# Фотографии боеприпасов (помещайте файлы JPG в assets/ammo/)
# ---------------------------------------------------------------------------
# Имя файла должно совпадать в точности (с учетом регистра в Linux).
AMMO_IMAGES: dict[str, str] = {
    "3BM60":       "3bm60.jpg",
    "3OF26":       "3of26.jpg",
    "3BK18M":      "3bk18m.jpg",
    "Observation": "",
    "— DANGER —":  "",
}

# ---------------------------------------------------------------------------
# Приоритет угрозы (используется для сортировки, когда в кадре несколько целей)
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
# Таблица стилей Qt — тема интерфейса наводчика Т-90М
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
