import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DirsDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augmentation=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augmentation = augmentation

        # 1. Получаем список всех изображений
        all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # 2. Фильтруем: оставляем только те, для которых есть соответствующая маска
        self.valid_img_names = []
        missing_count = 0

        for img_name in all_imgs:
            # Формируем ожидаемое имя маски (универсально меняем расширение на .png)
            name_without_ext = os.path.splitext(img_name)[0]
            mask_name = name_without_ext + '.png'
            mask_path = os.path.join(mask_dir, mask_name)

            if os.path.exists(mask_path):
                self.valid_img_names.append(img_name)
            else:
                missing_count += 1

        print(f"Dataset initialized: {len(self.valid_img_names)} valid pairs found. "
              f"Skipped {missing_count} images due to missing masks.")

        if len(self.valid_img_names) == 0:
            raise ValueError("No valid image-mask pairs found! Check paths and file extensions.")

    def __len__(self):
        return len(self.valid_img_names)

    def __getitem__(self, idx):
        img_name = self.valid_img_names[idx]
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
            # Теоретически сюда мы не должны попасть благодаря фильтрации в __init__,
            # но на случай битого файла:
            raise FileNotFoundError(f"Could not read mask (corrupted?): {mask_path}")

        # Нормализация маски к 0/1 (float32)
        mask = (mask > 127).astype(np.float32)

        # Аугментация
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Преобразование в Tensor и добавление размерности канала для маски [1, H, W]
        # ToTensorV2 обычно делает это сам, но если mask все еще numpy:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        # Убедимся, что у маски есть размерность канала (1, H, W)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask
