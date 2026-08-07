import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        ]),
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
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Affine(
                translate_percent=(-0.3, 0.3),
                scale=(0.5, 2.5),
                rotate=(-25, 25),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.5
            ),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.OneOf([
                A.Crop(x_min=0, y_min=0, x_max=256, y_max=512, p=1.0),  # Левая половина
                A.Crop(x_min=256, y_min=0, x_max=512, y_max=512, p=1.0),  # Правая половина
                A.CenterCrop(height=350, width=350, p=1.0),  # Центральный фрагмент
                A.RandomSizedCrop(
                    min_max_height=(250, 480),
                    size=(size[0], size[1]),  # ← ОБЯЗАТЕЛЬНЫЙ параметр
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                )
            ], p=0.34),
            A.Resize(size[0], size[1]),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1)),
                A.GaussianBlur(blur_limit=(3, 7)),
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