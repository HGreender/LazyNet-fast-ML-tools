import cv2
import torch
import numpy as np


def preprocess_image_for_model(
        image_path: str,
        augmentation_fn,
        device: torch.device
) -> tuple[np.ndarray, torch.Tensor, tuple[int, int]]:
    """
    Читает изображение, применяет аугментацию и возвращает тензор + оригинальные размеры.
    """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    original_size = (image_rgb.shape[1], image_rgb.shape[0])  # (Width, Height)

    augmented = augmentation_fn(image=image_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    return image_rgb, input_tensor, original_size


def postprocess_prediction(
        output_tensor: torch.Tensor,
        threshold: float = 0.5,
        original_size: tuple[int, int] = None
) -> np.ndarray:
    """
    Преобразует выход модели в бинарную маску и ресайзит её обратно.
    """
    pred_mask = output_tensor.squeeze().cpu().numpy()
    binary_mask = (pred_mask > threshold).astype(np.uint8)

    # Если маска 2D (H, W), добавляем канал для корректного ресайза в OpenCV, если нужно
    # Но cv2.resize отлично работает с 2D массивами

    if original_size is not None:
        # original_size это (W, H)
        binary_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)

    return binary_mask
