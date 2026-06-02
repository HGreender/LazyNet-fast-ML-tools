import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


from tools.ml.augmentations import get_imagenet_encoder_augmentation

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def show_n_augmentations(image_path, mask_path, n_samples=3, size = (512, 512)):
    """
    Показывает оригинал и аугментированные пары Image + Mask.
    """
    # 1. Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Загрузка маски
    # Маску читаем в grayscale (0 - фон, 255 - объект)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # Если маска бинарная, убедимся, что значения 0 и 1 (или 0 и 255)
    # Albumentations лучше работает с масками, где классы обозначены целыми числами

    # 3. Подготовка трансформации
    transform = get_imagenet_encoder_augmentation(phase='train')

    # 4. Ресайз оригинала для сравнения
    orig_resized_img = cv2.resize(image_rgb, (size[1], size[0]))
    orig_resized_mask = cv2.resize(mask, (size[1], size[0]))

    # Параметры для денормализации
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    # Создаем фигуру: 2 ряда (верх - картинка, низ - маска)
    # Столбцов: 1 (оригинал) + n_samples (аугментации)
    fig, axes = plt.subplots(2, n_samples + 1, figsize=(4 * (n_samples + 1), 8))

    # === ЛЕВАЯ КОЛОНКА: ОРИГИНАЛ ===
    axes[0, 0].imshow(orig_resized_img)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[1, 0].imshow(orig_resized_mask, cmap='gray')
    axes[1, 0].set_title("Original Mask")
    axes[1, 0].axis('off')

    # === ОСТАЛЬНЫЕ КОЛОНКИ: АУГМЕНТАЦИИ ===
    for i in range(1, n_samples + 1):
        # Применяем аугментацию к паре image + mask
        augmented = transform(image=image_rgb, mask=mask)

        tensor_img = augmented['image']
        tensor_mask = augmented['mask']  # Маска тоже стала тензором (1, H, W) или (H, W)

        # --- Обработка Картинки (Денормализация) ---
        img_np = tensor_img.numpy()
        img_np = img_np * std[:, None, None] + mean[:, None, None]
        img_np = np.clip(img_np, 0, 1)
        img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)

        # --- Обработка Маски ---
        # Маска после ToTensorV2 имеет форму (1, H, W) если была 2D, или (H, W)
        # Приводим к 2D numpy для отображения
        if isinstance(tensor_mask, torch.Tensor):
            mask_np = tensor_mask.squeeze().numpy()  # Убираем лишние размерности
        else:
            mask_np = tensor_mask

        # Отрисовка
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"Aug Image #{i}")
        axes[0, i].axis('off')

        # Маску рисуем в оттенках серого.
        # Если маска бинарная (0/1), можно наложить цветом для наглядности:
        axes[1, i].imshow(mask_np, cmap='gray')
        axes[1, i].set_title(f"Aug Mask #{i}")
        axes[1, i].axis('off')

        # Опционально: Наложить маску цветом поверх картинки (для проверки совпадения)
        # Создаем цветную маску (зеленую полупрозрачную)
        colored_mask = np.zeros_like(img_np)
        colored_mask[mask_np > 0.5] = [0, 1, 0]  # Зеленый цвет

        axes[0, i].imshow(colored_mask, alpha=0.4)  # Накладываем сверху

    plt.tight_layout()
    plt.show()
