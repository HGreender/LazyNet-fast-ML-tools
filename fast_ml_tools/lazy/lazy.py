import os

import torch
from torch.utils.data import DataLoader

from fast_ml_tools.ml.models import efficientnetb4_unet, efficientnetb4_unetpp
from fast_ml_tools.ml.trainer import Trainer
from fast_ml_tools.ml.augmentations import get_imagenet_encoder_augmentation
from fast_ml_tools.ml.datasets import DirsDataset
from fast_ml_tools.ml.losses import DiceBCELoss
from fast_ml_tools.visualization import plot_epochs_data


MODEL_FACTORIES = {
    'efficientnetb4_unet': efficientnetb4_unet,
    'efficientnetb4_unet++': efficientnetb4_unetpp,
}


def _get_device(num: int = 0):
    """Получение устройства (CUDA/CPU)"""
    print(f"Доступно карт: {torch.cuda.device_count()}")
    if num >= torch.cuda.device_count() or num < 0:
        num = 0
        print("Выбрана дефолтная видеокарта")

    device = torch.device(f"cuda:{num}" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    return device


class LazyNet:
    def __init__(
            self,
            model_name: str,
            train_img_dir: str, train_mask_dir: str,
            val_img_dir: str, val_mask_dir: str,
            epochs: int = 100, batch_size: int = 1, lr: float = 1e-4, patience: int = 5,
            device_id: int = 0, num_workers: int = 0,
            classes: list = ['target_class']
    ):
        self.device = _get_device(device_id)

        if model_name not in MODEL_FACTORIES:
            raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {list(MODEL_FACTORIES.keys())}")
        self.model = MODEL_FACTORIES[model_name](classes=classes)

        self.epochs = epochs

        self.train_augmentation = get_imagenet_encoder_augmentation()
        self.val_augmentation = get_imagenet_encoder_augmentation()

        self.train_dataset = DirsDataset(train_img_dir, train_mask_dir, augmentation=self.train_augmentation)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        self.val_dataset = DirsDataset(val_img_dir, val_mask_dir, augmentation=self.val_augmentation)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=patience
        )

        self.trainer = Trainer(
            self.model,
            self.train_loader,
            self.val_loader,
            self.optimizer,
            self.loss_fn,
            self.scheduler,
            self.device
        )

        self.train_logs = None

    def train(
            self,
            save_model_path: str = './models/best_model.pth',
            save_logs_path: str = './logs/training_logs.json',
    ):
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_logs_path), exist_ok=True)

        self.train_logs = self.trainer.fit(
            self.epochs,
            save_path=save_model_path,
            log_path=save_logs_path
        )

    def draw_logs(self):
        plot_epochs_data(self.train_logs)
