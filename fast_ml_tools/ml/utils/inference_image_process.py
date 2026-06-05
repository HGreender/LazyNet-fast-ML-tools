import cv2
import torch
import numpy as np


def preprocess_image_for_model(
        image_path: str,
        augmentation_fn,
        device: torch.device
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Читает изображение, применяет аугментацию (валидационную) и возвращает тензор.

    Args:
        image_path: Путь к изображению.
        augmentation_fn: Функция аугментации (например, из albumentations).
        device: Устройство (cuda/cpu).

    Returns:
        image_rgb: Исходное изображение в RGB (numpy) для визуализации.
        input_tensor: Тензор, готовый к подаче в модель [1, C, H, W].
    """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    # Конвертация BGR -> RGB
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # Применение аугментаций (обычно это ресайз и нормализация)
    augmented = augmentation_fn(image=image_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    return image_rgb, input_tensor


def postprocess_prediction(
        output_tensor: torch.Tensor,
        threshold: float = 0.5
) -> np.ndarray:
    """
    Преобразует выход модели в бинарную маску.

    Args:
        output_tensor: Выход модели [B, C, H, W] или [B, H, W].
        threshold: Порог отсечения.

    Returns:
        binary_mask: Numpy array типа uint8 (H, W).
    """
    # Убираем батч-димензию и каналы, если они есть
    pred_mask = output_tensor.squeeze().cpu().numpy()

    # Бинаризация
    binary_mask = (pred_mask > threshold).astype(np.uint8)

    return binary_mask
