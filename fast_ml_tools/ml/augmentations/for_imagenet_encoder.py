import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_manga_panel_bubble_augmentation(phase='train', size=(512, 512)):
    """
    Аугментации для сегментации панелей и пузырей в манге.
    Автоматически конвертирует цветные изображения в Grayscale.
    """
    MEAN_GRAY = 0.485 
    STD_GRAY = 0.229
    
    if phase == 'train':
        return A.Compose([
            A.ToGray(num_output_channels=1, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=0.05,
                scale=(0.8, 1.2),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_CONSTANT,
                fill=255, # Заполняем белым (фон манги)
                fill_mask=0, # Заполняем чёрным (фон маски)
                p=0.7
            ),
            A.OneOf([
                # Разная освещённость при сканировании
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                # Усиление локальных контрастов (помогает выделить бледные линии панелей)
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                # Имитация "грязной" бумаги или шума сканера
                A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            ], p=0.8),
            A.Resize(size[0], size[1]),
            A.Normalize(mean=[MEAN_GRAY], std=[STD_GRAY]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.ToGray(num_output_channels=1, p=1.0),
            A.Resize(size[0], size[1]),
            A.Normalize(mean=[MEAN_GRAY], std=[STD_GRAY]),
            ToTensorV2(),
        ])


def get_grayscale_imagenet_augmentation(phase='train', size=(512, 512)):
    """Аугментации для 1-канальных (Grayscale) медицинских снимков"""
    # Используем mean/std только для первого канала ImageNet или средние значения
    MEAN_GRAY = 0.485 
    STD_GRAY = 0.229
    
    if phase == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.Resize(size[0], size[1]),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1)),
                A.GaussianBlur(blur_limit=(3, 5)),
            ], p=0.2),
            A.Normalize(mean=[MEAN_GRAY], std=[STD_GRAY]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean=[MEAN_GRAY], std=[STD_GRAY]),
            ToTensorV2(),
        ])


def get_imagenet_encoder_augmentation(
        phase='train',
        size=(512, 512)
):
    """Аугментации, важные для медицинских снимков"""
    if phase == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=0.05,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.Resize(size[0], size[1]),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1)),
                A.GaussianBlur(blur_limit=(3, 5)),
            ], p=0.2),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            ], p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])


def get_lungs_imagenet_encoder_augmentation(
        phase='train',
        size=(512, 512)
):
    """Агрессивные аугментации, важные для медицинских снимков лёгких"""
    if phase == 'train':
        return A.Compose([
            A.Resize(size[0], size[1]),

            A.HorizontalFlip(p=0.5),

            A.Affine(
                translate_percent=(-0.35, 0.35),
                rotate=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.5
            ),

            A.OneOf([
                A.RandomCrop(height=350, width=350),
                A.RandomSizedCrop(
                    min_max_height=(350, 500),  # ← диапазон высоты кропа
                    size=(512, 512),  # ← целевой размер после ресайза
                    w2h_ratio=1.0,  # ← ГАРАНТИРУЕТ квадратный кроп (сохраняет пропорции)
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST
                )
            ], p=0.5),

            A.Resize(size[0], size[1]),

            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1)),
                A.GaussianBlur(blur_limit=(3, 7)),
            ], p=0.3),

            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            ], p=0.4),

            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(size[0], size[1]),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])