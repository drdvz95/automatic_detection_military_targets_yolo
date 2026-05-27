"""
main.py
=======
Точка входа. Инициализирует QApplication и запускает главное окно.

Запуск:
    python main.py
"""

import sys

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from config import QSS_THEME
from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Military Detection System")
    app.setApplicationVersion("1.0.0")
    app.setStyleSheet(QSS_THEME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()