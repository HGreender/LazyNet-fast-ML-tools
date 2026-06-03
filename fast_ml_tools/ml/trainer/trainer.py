import os

import cv2
import torch
import numpy as np
from tqdm import tqdm

from fast_ml_tools.logging import EpochsLogger
from fast_ml_tools.visualization import show_segmentation

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, scheduler=None, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device

        # Логгер для отслеживания метрик
        self.logger = EpochsLogger()

        # Состояние тренировки
        self.best_loss = float('inf')
        self.current_epoch = 0

    def train_epoch(self):
        """Одна эпоха обучения"""
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.loss_fn(outputs, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return running_loss / len(self.train_loader.dataset)

    def validate(self):
        """Валидация модели"""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                running_loss += loss.item() * images.size(0)

        return running_loss / len(self.val_loader.dataset)

    def save_best_model(self, path='./models/best_model.pth'):
        """Сохранение лучшей модели"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved best model to {path}")

    def fit(self, epochs, save_path='./models/best_model.pth', log_path='./logs/training_logs.json'):
        """Основной цикл тренировки"""
        print(f"Start training on {self.device} for {epochs} epochs")

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Обучение и валидация
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Шаг планировщика
            if self.scheduler:
                self.scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            # Сохранение лучшей модели
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_best_model(save_path)

            # Логирование
            self.logger.add_and_save_logs_json(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                best_loss=self.best_loss,
                path=log_path
            )

        print(f"Training finished. Best Val Loss: {self.best_loss:.4f}")
        return self.logger.epoch_logs

    def load_logs_and_continue(self, log_path='./logs/training_logs.json'):
        """Загрузка предыдущих логов (опционально, если нужно продолжить тренировку)"""
        if self.logger.load_logs_json(log_path):
            print("Previous logs loaded.")
            # Здесь можно добавить логику восстановления состояния, если нужно


class InferenceEngine:
    def __init__(self, model_class, model_path, device='cuda'):
        self.device = device
        self.model = model_class().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        print(f"📦 Model loaded from {model_path}")

    def predict(self, image_path, augmentation_fn, threshold=0.5):
        """Предсказание маски для одного изображения"""
        # Preprocessing
        image_cv = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        augmented = augmentation_fn(image=image_rgb)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            pred_mask = self.model(input_tensor)

        # Post-processing
        pred_mask = pred_mask.squeeze().cpu().numpy()
        binary_mask = (pred_mask > threshold).astype(np.uint8)

        return image_rgb, binary_mask

    def visualize(self, image_rgb, binary_mask, save_path=None):
        """Визуализация результатов"""
        show_segmentation(image_rgb, binary_mask, save_path)

    def run(self, image_path, augmentation_fn, save_viz_path=None):
        """Полный пайплайн инференса"""
        image_rgb, binary_mask = self.predict(image_path, augmentation_fn)
        self.visualize(image_rgb, binary_mask, save_path=save_viz_path)
        return binary_mask