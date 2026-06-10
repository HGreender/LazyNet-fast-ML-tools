import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiDirsDataset(Dataset):
    """
    Датасет для работы с несколькими папками изображений и масок.
    Собирает все валидные пары изображение-маска из всех указанных папок.
    """

    def __init__(self, img_dirs: list[str], mask_dirs: list[str], augmentation=None):
        """
        Args:
            img_dirs: Список путей к папкам с изображениями
            mask_dirs: Список путей к папкам с масками (должен быть той же длины, что и img_dirs)
            augmentation: Функция аугментации (опционально)
        """
        if len(img_dirs) != len(mask_dirs):
            raise ValueError(f"Количество папок с изображениями ({len(img_dirs)}) "
                             f"не совпадает с количеством папок с масками ({len(mask_dirs)})")

        self.augmentation = augmentation
        self.samples = []  # Список кортежей (img_path, mask_path)

        # Собираем все валидные пары из всех папок
        for img_dir, mask_dir in zip(img_dirs, mask_dirs):
            if not os.path.exists(img_dir):
                print(f"Предупреждение: Папка {img_dir} не существует, пропускаем")
                continue
            if not os.path.exists(mask_dir):
                print(f"Предупреждение: Папка {mask_dir} не существует, пропускаем")
                continue

            # Получаем список всех изображений в текущей папке
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
            all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

            for img_name in all_imgs:
                name_without_ext = os.path.splitext(img_name)[0]
                mask_name = name_without_ext + '.png'
                mask_path = os.path.join(mask_dir, mask_name)

                if os.path.exists(mask_path):
                    img_path = os.path.join(img_dir, img_name)
                    self.samples.append((img_path, mask_path))

        print(f"MultiDirsDataset: найдено {len(self.samples)} валидных пар изображение-маска")

        if len(self.samples) == 0:
            raise ValueError("Не найдено ни одной валидной пары изображение-маска!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

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
            # HWC -> CHW и нормализация к [0, 1]
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor):
            # Если аугментация вернула tensor, убедимся что он float
            if image.dtype == torch.uint8:
                image = image.float() / 255.0

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Убедимся, что у маски есть размерность канала (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask


class MultiDirsDatasetFromSamples(Dataset):
    """
    Датасет, созданный из готового списка пар (img_path, mask_path).
    Используется внутри create_train_val_datasets_from_multiple_dirs.
    """

    def __init__(self, samples: list[tuple[str, str]], augmentation=None):
        """
        Args:
            samples: Список кортежей (img_path, mask_path)
            augmentation: Функция аугментации (опционально)
        """
        self.samples = samples
        self.augmentation = augmentation
        print(f"MultiDirsDatasetFromSamples: создано {len(self.samples)} сэмплов")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

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
            # HWC -> CHW и нормализация к [0, 1]
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
        elif isinstance(image, torch.Tensor):
            # Если аугментация вернула tensor, убедимся что он float
            if image.dtype == torch.uint8:
                image = image.float() / 255.0

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Убедимся, что у маски есть размерность канала (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask


def create_train_val_datasets_from_multiple_dirs(
        img_dirs: list[str],
        mask_dirs: list[str],
        train_ratio: float = 0.8,
        seed: int = 42,
        train_augmentation=None,
        val_augmentation=None
) -> tuple[MultiDirsDatasetFromSamples, MultiDirsDatasetFromSamples]:
    """
    Создаёт тренировочный и валидационный датасеты из нескольких папок с изображениями и масками.

    Args:
        img_dirs: Список путей к папкам с оригинальными изображениями
        mask_dirs: Список путей к папкам с масками
        train_ratio: Доля данных для тренировки (по умолчанию 0.8)
        seed: Seed для воспроизводимости разбиения
        train_augmentation: Функция аугментации для train-датасета
        val_augmentation: Функция аугментации для val-датасета

    Returns:
        Кортеж (train_dataset, val_dataset)
    """
    if len(img_dirs) != len(mask_dirs):
        raise ValueError(f"Количество папок с изображениями ({len(img_dirs)}) "
                         f"не совпадает с количеством папок с масками ({len(mask_dirs)})")

    # Собираем все валидные пары из всех папок
    all_samples = []  # Список кортежей (img_path, mask_path)
    missing_count = 0

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')

    for img_dir, mask_dir in zip(img_dirs, mask_dirs):
        if not os.path.exists(img_dir):
            print(f"Предупреждение: Папка {img_dir} не существует, пропускаем")
            continue
        if not os.path.exists(mask_dir):
            print(f"Предупреждение: Папка {mask_dir} не существует, пропускаем")
            continue

        # Получаем список всех изображений в текущей папке
        all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]

        for img_name in all_imgs:
            name_without_ext = os.path.splitext(img_name)[0]
            mask_name = name_without_ext + '.png'
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                img_path = os.path.join(img_dir, img_name)
                all_samples.append((img_path, mask_path))
            else:
                missing_count += 1

    print(f"Found {len(all_samples)} valid pairs across all directories. "
          f"Skipped {missing_count} images due to missing masks.")

    if len(all_samples) == 0:
        raise ValueError("No valid image-mask pairs found! Check paths and file extensions.")

    # Перемешиваем и сплитим
    random.seed(seed)
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    train_dataset = MultiDirsDatasetFromSamples(
        samples=train_samples,
        augmentation=train_augmentation
    )

    val_dataset = MultiDirsDatasetFromSamples(
        samples=val_samples,
        augmentation=val_augmentation
    )

    return train_dataset, val_dataset
