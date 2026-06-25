#!/bin/bash
# install.sh
# ===========
# Единоразовая установка приложения "Автоматическое распознавание
# военных целей" на Ubuntu. Собирает Docker-образ, делает launch-скрипт
# исполняемым и добавляет ярлык в меню приложений и на рабочий стол.
#
# Запуск (один раз, после git clone):
#   chmod +x install.sh
#   ./install.sh

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo " Установка: Автоматическое распознавание военных целей"
echo "=================================================="
echo "Каталог проекта: $PROJECT_DIR"
echo ""

# ── Проверка наличия Docker ──────────────────────────────────────────────
if ! command -v docker &> /dev/null; then
    echo "Docker не найден. Устанавливаю..."
    sudo apt update
    sudo apt install -y docker.io
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    echo ""
    echo "ВНИМАНИЕ: Docker только что установлен и твой пользователь"
    echo "добавлен в группу docker. Нужно перелогиниться (или выполнить"
    echo "'newgrp docker' в текущем терминале), затем запустить этот"
    echo "скрипт повторно."
    exit 0
fi

# ── Проверка наличия Git LFS (нужен для весов модели) ────────────────────
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS не найден. Устанавливаю..."
    sudo apt update
    sudo apt install -y git-lfs
    git lfs install
fi

# Если веса модели не подтянулись автоматически при клонировании —
# подтягиваем явно. Без этого weights/best.pt будет лёгким
# текстовым указателем на ~130 байт, а не настоящей моделью.
if [ -f "$PROJECT_DIR/weights/best.pt" ]; then
    SIZE=$(stat -c%s "$PROJECT_DIR/weights/best.pt" 2>/dev/null || echo 0)
    if [ "$SIZE" -lt 100000 ]; then
        echo "Веса модели не материализованы, подтягиваю через Git LFS..."
        (cd "$PROJECT_DIR" && git lfs pull)
    fi
fi

# ── Сборка Docker-образа ──────────────────────────────────────────────────
echo ""
echo "Собираю Docker-образ (может занять несколько минут)..."
docker build -t military-detector "$PROJECT_DIR"

# ── Права на запуск ────────────────────────────────────────────────────────
chmod +x "$PROJECT_DIR/launch-military-detector.sh"

# ── Ярлык в меню приложений и на рабочем столе ────────────────────────────
DESKTOP_FILE="$HOME/.local/share/applications/military-detector.desktop"
mkdir -p "$HOME/.local/share/applications"

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Military Detector
Comment=Автоматическое распознавание военных целей
Exec=/bin/bash $PROJECT_DIR/launch-military-detector.sh
Icon=$PROJECT_DIR/assets/icon.png
Terminal=false
Categories=Utility;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"

# Копия на рабочий стол — не у всех ~/.local/share/applications сразу
# подхватывается лаунчером, а на рабочем столе ярлык виден сразу.
if [ -d "$HOME/Desktop" ]; then
    cp "$DESKTOP_FILE" "$HOME/Desktop/military-detector.desktop"
    chmod +x "$HOME/Desktop/military-detector.desktop"
fi

echo ""
echo "=================================================="
echo " Установка завершена."
echo "=================================================="
echo "Запустить приложение можно:"
echo "  - через меню приложений: поиск 'Military Detector'"
echo "  - ярлыком на рабочем столе"
echo "  - вручную: ./launch-military-detector.sh"
echo "=================================================="
