"""
Smoke-проверка установки fast-ml-tools.

Запуск (после pip install fast-ml-tools):
    fast-ml-check            # быстрая проверка
    fast-ml-check --verbose  # с полными трейсбеками ошибок
    python -m fast_ml_tools.check

Логирование: loguru, если установлен, иначе стандартный logging.

Что проверяется:
    1.  Импорт пакета, версия
    2.  Публичный API: LazyNet, LazyNetOptuna
    3.  Подмодули: preprocessing, visualization, ml.*, logging
    4.  Конструктор LazyNet() без аргументов
    5.  Фабрики моделей: тип архитектуры и число входных каналов
        (в т.ч. регрессионная проверка: unetpp* — UnetPlusPlus,
         *_1_channel — 1 канал)
    6.  Функции потерь: создание + вычисление на случайных тензорах
    7.  Метрики сегментации
    8.  Аугментация: применение к тестовой паре image+mask
    9.  Forward-прогон модели на CPU
    10. Функциональная проверка preprocessing.get_mask_bbox

Важно: при первом создании моделей скачиваются ImageNet-веса
efficientnet-b4 (~60 МБ) — нужен интернет или уже заполненный кэш timm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- Логирование: loguru, если доступен, иначе стандартный logging ---
try:
    from loguru import logger

    _HAS_LOGURU = True
except ImportError:
    import logging

    _HAS_LOGURU = False

    class _FallbackLogger:
        """Минимальная имитация API loguru (если loguru не установлен)."""

        def __init__(self) -> None:
            self._log = logging.getLogger("fast_ml_tools.check")
            self._log.setLevel(logging.DEBUG)
            self._log.propagate = False
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                  datefmt="%H:%M:%S")
            )
            self._log.addHandler(handler)
            self._file_sinks: set[str] = set()

        def remove(self) -> None:
            pass

        def add(self, sink, level="INFO", format=None, rotation=None, encoding=None):
            if isinstance(sink, (str, Path)):
                key = str(sink)
                if key in self._file_sinks:
                    return
                self._file_sinks.add(key)
                file_handler = logging.FileHandler(key, encoding=encoding or "utf-8")
                file_handler.setLevel(getattr(logging, str(level).upper(), logging.INFO))
                file_handler.setFormatter(
                    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
                )
                self._log.addHandler(file_handler)
            # консольный sink (sys.stdout) уже настроен в __init__

        def debug(self, message, *args, **kwargs):
            self._log.debug(message)

        def info(self, message, *args, **kwargs):
            self._log.info(message)

        def warning(self, message, *args, **kwargs):
            self._log.warning(message)

        def success(self, message, *args, **kwargs):
            self._log.info(message)

        def error(self, message, *args, **kwargs):
            self._log.error(message)

        def exception(self, message, *args, **kwargs):
            self._log.exception(message)

    logger = _FallbackLogger()


def setup_logging(file_sink: str | Path | None = None) -> None:
    """Настроить логирование: консоль (всегда) + опционально файл."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    if file_sink is not None:
        Path(file_sink).parent.mkdir(parents=True, exist_ok=True)
        logger.add(file_sink, level="DEBUG", rotation="1 MB", encoding="utf-8")
    if not _HAS_LOGURU:
        logger.warning("loguru не установлен — используется стандартный logging "
                       "(pip install loguru для цветного вывода)")


# ---------------------------------------------------------------------------
# 1. Импорт пакета
# ---------------------------------------------------------------------------
def _check_import_package() -> str:
    import fast_ml_tools
    assert fast_ml_tools.__version__, "пустая версия"
    return f"version={fast_ml_tools.__version__}"


# ---------------------------------------------------------------------------
# 2. Публичный API
# ---------------------------------------------------------------------------
def _check_public_api() -> None:
    from fast_ml_tools import LazyNet, LazyNetOptuna
    assert callable(LazyNet) and callable(LazyNetOptuna)


# ---------------------------------------------------------------------------
# 3. Подмодули
# ---------------------------------------------------------------------------
def _check_submodules() -> None:
    from fast_ml_tools import preprocessing, visualization
    from fast_ml_tools.ml import models, losses, metrics, datasets, augmentations, utils, trainer
    from fast_ml_tools.logging import EpochsLogger
    from fast_ml_tools.ml.datasets import (MultiDirsDataset, InferenceDataset,
                                           create_weighted_sampler,
                                           create_train_val_datasets_from_multiple_dirs)
    from fast_ml_tools.ml.utils import (get_device,
                                        preprocess_image_array_for_model,
                                        postprocess_prediction)
    from fast_ml_tools.ml.trainer import Trainer


# ---------------------------------------------------------------------------
# 4. Конструктор LazyNet без аргументов
# ---------------------------------------------------------------------------
def _check_lazynet_ctor() -> str:
    from fast_ml_tools import LazyNet
    net = LazyNet()
    assert net.device is not None
    return f"device={net.device}"


# ---------------------------------------------------------------------------
# 5. Фабрики моделей
# ---------------------------------------------------------------------------
# Ожидаемые (type, in_channels). Регрессионная проверка:
# раньше unetpp* возвращали обычный Unet, а *_1_channel — 3 канала.
EXPECTED_MODELS = {
    "efficientnetb4_unet": ("Unet", 3),
    "efficientnetb4_unet_1_channel": ("Unet", 1),
    "efficientnetb4_unetpp": ("UnetPlusPlus", 3),
    "efficientnetb4_unetpp_1_channel": ("UnetPlusPlus", 1),
}


def _check_model_factories() -> str:
    import fast_ml_tools.ml.models.create_model as cm
    parts = []
    for name, (expected_type, expected_ch) in EXPECTED_MODELS.items():
        model = getattr(cm, name)()
        actual_type = type(model).__name__
        actual_ch = model.encoder._conv_stem.in_channels
        assert actual_type == expected_type, \
            f"{name}: ожидался {expected_type}, получен {actual_type}"
        assert actual_ch == expected_ch, \
            f"{name}: ожидалось {expected_ch} каналов, получено {actual_ch}"
        parts.append(f"{name}={actual_type}/{actual_ch}ch")
    return ", ".join(parts)




# ---------------------------------------------------------------------------
# 6. Функции потерь
# ---------------------------------------------------------------------------
def _check_losses() -> str:
    import torch
    from fast_ml_tools.ml.losses import (FocalLoss, DiceBCELoss, DiceFocalLoss,
                                         IoUFocalLoss, LovaszBCELoss)
    logits = torch.randn(2, 1, 32, 32)
    targets = (torch.rand(2, 1, 32, 32) > 0.5).float()
    for loss_cls in (FocalLoss, DiceBCELoss, DiceFocalLoss, IoUFocalLoss, LovaszBCELoss):
        loss = loss_cls()(logits, targets)
        assert torch.isfinite(loss), f"{loss_cls.__name__} вернул не-конечную потерю"
    return "5/5 finite"


# ---------------------------------------------------------------------------
# 7. Метрики сегментации
# ---------------------------------------------------------------------------
def _check_metrics() -> str:
    from fast_ml_tools.ml.metrics import get_segmentation_metrics
    metrics = get_segmentation_metrics()
    assert len(metrics) == 3, f"ожидалось 3 метрики, получено {len(metrics)}"
    return f"{len(metrics)} метрик"


# ---------------------------------------------------------------------------
# 8. Аугментация
# ---------------------------------------------------------------------------
def _check_augmentations() -> str:
    import numpy as np
    from fast_ml_tools.ml.augmentations import get_imagenet_encoder_augmentation
    aug = get_imagenet_encoder_augmentation(phase="train", size=(64, 64))
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    out = aug(image=image, mask=mask)
    assert tuple(out["image"].shape) == (3, 64, 64), f"неверный shape: {out['image'].shape}"
    return "image (3, 64, 64)"


# ---------------------------------------------------------------------------
# 9. Forward-прогон
# ---------------------------------------------------------------------------
def _check_forward() -> str:
    import torch
    from fast_ml_tools.ml.models import efficientnetb4_unet
    model = efficientnetb4_unet()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 64, 64))
    assert tuple(out.shape) == (1, 1, 64, 64), f"неверный shape: {out.shape}"
    return f"out={tuple(out.shape)}"


# ---------------------------------------------------------------------------
# 10. Препроцессинг
# ---------------------------------------------------------------------------
def _check_bbox() -> str:
    import numpy as np
    from fast_ml_tools.preprocessing import get_mask_bbox
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    bbox = get_mask_bbox(mask)
    assert bbox == (3, 2, 6, 4), f"неверный bbox: {bbox}"
    return f"bbox={bbox}"


CHECKS: list[tuple[str, callable]] = [
    ("Импорт fast_ml_tools", _check_import_package),
    ("Публичный API: LazyNet, LazyNetOptuna", _check_public_api),
    ("Подмодули (preprocessing, visualization, ml.*, logging)", _check_submodules),
    ("LazyNet() без аргументов", _check_lazynet_ctor),
    ("Фабрики моделей: архитектура и входные каналы", _check_model_factories),
    ("Функции потерь: создание + вычисление", _check_losses),
    ("Метрики сегментации", _check_metrics),
    ("Аугментация: применение к тестовой паре image+mask", _check_augmentations),
    ("Forward-прогон efficientnetb4_unet на CPU", _check_forward),
    ("preprocessing.get_mask_bbox", _check_bbox),
]


def run_checks(verbose: bool = False) -> tuple[int, int]:
    """Выполняет все проверки. Возвращает (passed, total)."""
    results = []
    for name, fn in CHECKS:
        try:
            detail = fn()
            results.append((name, True, detail))
            msg = name + (f" — {detail}" if detail else "")
            logger.success(msg)
        except Exception as e:
            results.append((name, False, str(e)))
            if verbose:
                logger.exception(f"{name} — {e}")
            else:
                logger.error(f"{name} — {e}")
    passed = sum(1 for _, ok, _ in results if ok)
    return passed, len(results)


def main(argv: list[str] | None = None, file_sink: str | Path | None = None) -> int:
    """Точка входа: консольная команда fast-ml-check."""
    parser = argparse.ArgumentParser(
        prog="fast-ml-check",
        description="Smoke-проверка установки fast-ml-tools",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="полные трейсбеки ошибок")
    args = parser.parse_args(argv)

    setup_logging(file_sink=file_sink)
    logger.info("Smoke-тест установки fast-ml-tools")
    passed, total = run_checks(verbose=args.verbose)
    failed = total - passed
    if failed:
        logger.error(f"Итог: {passed}/{total} проверок пройдено, {failed} ошибок")
        return 1
    logger.success(f"Итог: {passed}/{total} проверок пройдено — библиотека установлена корректно")
    return 0


if __name__ == "__main__":
    sys.exit(main())
