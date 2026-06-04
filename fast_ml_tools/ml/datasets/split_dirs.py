import os
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ImageMaskDataset(Dataset):
    def __init__(self, img_dir: str, mask_dir: str, img_names: list, augmentation=None):
        """
        Датасет для работы с парами изображение-маска.

        Args:
            img_dir: Путь к папке с изображениями
            mask_dir: Путь к папке с масками
            img_names: Список имён файлов изображений для этого датасета
            augmentation: Функция аугментации (опционально)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = img_names
        self.augmentation = augmentation

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Формируем путь к маске
        name_without_ext = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, name_without_ext + '.png')

        # Чтение изображения
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Чтение маски
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        # Нормализация маски к 0/1 (float32)
        mask = (mask > 127).astype(np.float32)

        # Аугментация
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Преобразование в Tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Убедимся, что у маски есть размерность канала (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask


def create_train_val_datasets(
        img_dir: str,
        mask_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        seed: int = 42,
        augmentation=None
) -> Tuple[ImageMaskDataset, ImageMaskDataset]:
    """
    Создаёт тренировочный и валидационный датасеты из папок с изображениями и масками.

    Args:
        img_dir: Путь к папке с оригинальными изображениями
        mask_dir: Путь к папке с масками
        train_ratio: Доля данных для тренировки (по умолчанию 0.8)
        val_ratio: Доля данных для валидации (по умолчанию 0.2)
        seed: Seed для воспроизводимости разбиения
        augmentation: Функция аугментации (применяется только к train)

    Returns:
        Кортеж (train_dataset, val_dataset)
    """
    assert train_ratio + val_ratio == 1.0, "train_ratio + val_ratio должно равняться 1.0"

    # Получаем список всех изображений
    all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Фильтруем: оставляем только те, для которых есть соответствующая маска
    valid_img_names = []
    missing_count = 0

    for img_name in all_imgs:
        name_without_ext = os.path.splitext(img_name)[0]
        mask_name = name_without_ext + '.png'
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            valid_img_names.append(img_name)
        else:
            missing_count += 1

    print(f"Found {len(valid_img_names)} valid pairs. "
          f"Skipped {missing_count} images due to missing masks.")

    if len(valid_img_names) == 0:
        raise ValueError("No valid image-mask pairs found! Check paths and file extensions.")

    # Перемешиваем и сплитим
    random.seed(seed)
    random.shuffle(valid_img_names)

    split_idx = int(len(valid_img_names) * train_ratio)
    train_names = valid_img_names[:split_idx]
    val_names = valid_img_names[split_idx:]

    print(f"Train samples: {len(train_names)}, Val samples: {len(val_names)}")

    # Создаём датасеты
    train_dataset = ImageMaskDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_names=train_names,
        augmentation=augmentation
    )

    val_dataset = ImageMaskDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        img_names=val_names,
        augmentation=None  # Валидация без аугментации
    )

    return train_dataset, val_dataset

# # Пример
# train_dataset, val_dataset = create_train_val_datasets(
#     img_dir="/path/to/images",
#     mask_dir="/path/to/masks",
#     train_ratio=0.8,
#     val_ratio=0.2,
#     seed=42
# )
