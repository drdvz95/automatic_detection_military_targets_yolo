#!/bin/bash
# launch-military-detector.sh
# =============================
# Запуск приложения внутри Docker-контейнера с выводом GUI на экран хоста.
#
# Путь к проекту определяется автоматически (через расположение этого
# скрипта), поэтому файл можно скопировать на любую машину после
# git clone — не нужно вручную прописывать свой логин или путь.

set -e

# Абсолютный путь к папке, где лежит сам этот скрипт = корень проекта.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Папки для скриншотов/отчётов и вывода создаём заранее на хосте.
# Если не сделать этого, Docker создаст их сам от имени root внутри
# контейнера, и потом обычный пользователь не сможет в них писать/читать.
mkdir -p "$PROJECT_DIR/screenshots"
mkdir -p "$PROJECT_DIR/output"

# Разрешаем контейнеру рисовать в X11-сессию хоста — без этого PyQt6
# окно просто не появится на экране.
xhost +local:docker

docker run --rm \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PROJECT_DIR/assets:/app/assets" \
  -v "$PROJECT_DIR/output:/app/output" \
  -v "$PROJECT_DIR/screenshots:/app/screenshots" \
  -v /usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libEGL.so.1 \
  -v /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0:/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 \
  -v /usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0:/usr/lib/x86_64-linux-gnu/libxcb-cursor.so.0 \
  --device /dev/dri \
  military-detector

# Закрываем доступ обратно — не оставляем X11-сессию открытой для
# любого локального процесса после завершения работы программы.
xhost -local:docker
