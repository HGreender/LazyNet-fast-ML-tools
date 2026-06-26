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
        original_size: tuple[int, int] = None,
        return_probabilities: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    probabilities = torch.sigmoid(output_tensor)
    pred_mask = probabilities.squeeze().cpu().numpy()

    if original_size is not None:
        prob_resized = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_LINEAR)
        binary_mask = (prob_resized > threshold).astype(np.uint8)
    else:
        binary_mask = (pred_mask > threshold).astype(np.uint8)
        prob_resized = pred_mask

    if return_probabilities:
        return binary_mask, prob_resized
    return binary_mask
