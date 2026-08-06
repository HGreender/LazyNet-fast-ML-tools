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
                A.GaussNoise(var_limit=(10.0, 50.0)),
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
