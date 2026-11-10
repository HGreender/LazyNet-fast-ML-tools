#!/usr/bin/env python3
"""
Smoke-тест: прогоняет проверки fast_ml_tools.check с логированием в файл.

Сама логика проверок — в пакете (fast_ml_tools/check.py), здесь только:
    - запись логов в logs/test_installation.log (папка в .gitignore)
    - точка входа для pytest

Запуск:
    python tests/test_installation.py            # скриптовый режим
    python tests/test_installation.py --verbose  # с полными трейсбеками
    pytest tests/test_installation.py            # режим pytest

loguru опционален — без него используется стандартный logging.

Код выхода: 0 — все проверки пройдены, 1 — есть ошибки.
"""

import sys
from pathlib import Path

from fast_ml_tools.check import main as check_main, run_checks, setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "test_installation.log"


def test_installation():
    """Точка входа для pytest: все smoke-проверки должны пройти."""
    setup_logging(file_sink=LOG_FILE)
    passed, total = run_checks(verbose="--verbose" in sys.argv)
    assert passed == total, f"провалено {total - passed} из {total} проверок"


def main():
    return check_main(sys.argv[1:], file_sink=LOG_FILE)


if __name__ == "__main__":
    sys.exit(main())
