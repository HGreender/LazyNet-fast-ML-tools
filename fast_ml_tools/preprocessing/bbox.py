'''Функции для создания bounding box'ов'''

import numpy as np


def get_mask_bbox(heart_mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Вычисляет минимальный ограничивающий прямоугольник (bounding box)
    вокруг размеченной области по бинарной маске.

    Args:
        heart_mask: Бинарная маска сердца (2D numpy array)

    Returns:
        Tuple (x_min, y_min, x_max, y_max)

    Raises:
        ValueError: Если маска пустая
    """
    # Находим координаты ненулевых пикселей
    rows, cols = np.where(heart_mask > 0)

    if len(rows) == 0:
        raise ValueError("Маска сердца пустая")

    # Вычисляем границы
    y_min = int(np.min(rows))
    y_max = int(np.max(rows))
    x_min = int(np.min(cols))
    x_max = int(np.max(cols))

    return (x_min, y_min, x_max, y_max)


def expand_bbox(
        bbox: tuple[int, int, int, int],
        img_width: int,
        img_height: int,
        expand_width: int = 0,
        expand_height: int = 0
) -> tuple[int, int, int, int]:
    """
    Расширяет bounding box на заданное количество пикселей по ширине и высоте.

    Args:
        bbox: Исходный bounding box (x_min, y_min, x_max, y_max)
        img_width: Ширина изображения (для ограничения границ)
        img_height: Высота изображения (для ограничения границ)
        expand_width: Расширение по ширине в пикселях (добавляется с каждой стороны)
        expand_height: Расширение по высоте в пикселях (добавляется с каждой стороны)

    Returns:
        Tuple (x_min, y_min, x_max, y_max) - расширенный bounding box
    """
    x_min, y_min, x_max, y_max = bbox

    # Расширяем и ограничиваем границами изображения
    x_min_expanded = max(0, x_min - expand_width)
    y_min_expanded = max(0, y_min - expand_height)
    x_max_expanded = min(img_width - 1, x_max + expand_width)
    y_max_expanded = min(img_height - 1, y_max + expand_height)

    return (x_min_expanded, y_min_expanded, x_max_expanded, y_max_expanded)


def get_expanded_heart_bbox(
        heart_mask: np.ndarray,
        expand_width: int = 0,
        expand_height: int = 0
) -> tuple[int, int, int, int]:
    """
    Комбинирует поиск bounding box сердца и его расширение.

    Args:
        heart_mask: Бинарная маска сердца (2D numpy array)
        expand_width: Расширение по ширине в пикселях (с каждой стороны)
        expand_height: Расширение по высоте в пикселях (с каждой стороны)

    Returns:
        Tuple (x_min, y_min, x_max, y_max) - расширенный bounding box

    Raises:
        ValueError: Если маска пустая
    """
    # Получаем исходный bbox
    bbox = get_mask_bbox(heart_mask)

    # Расширяем его
    height, width = heart_mask.shape
    expanded_bbox = expand_bbox(bbox, width, height, expand_width, expand_height)

    return expanded_bbox


def bbox_to_yolo_format(
        bbox: tuple[int, int, int, int],
        img_width: int,
        img_height: int
) -> tuple[float, float, float, float]:
    """
    Конвертирует bounding box в формат YOLO.

    Args:
        bbox: (x_min, y_min, x_max, y_max)
        img_width: Ширина изображения
        img_height: Высота изображения

    Returns:
        Tuple (x_center, y_center, width, height) в нормализованном формате [0-1]
    """
    x_min, y_min, x_max, y_max = bbox

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    return (
        x_center / img_width,
        y_center / img_height,
        width / img_width,
        height / img_height
    )
