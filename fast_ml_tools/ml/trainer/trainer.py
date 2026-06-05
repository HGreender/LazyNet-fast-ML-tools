import os

import cv2
import torch
import numpy as np
from tqdm import tqdm

from fast_ml_tools.logging import EpochsLogger
from fast_ml_tools.visualization import show_segmentation

class Trainer:
    def __init__(
            self,
            model, train_loader, val_loader,
            optimizer, loss_fn, metrics: list = None, scheduler=None,
            device='cuda', verbose=True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.verbose = verbose
        self.metrics = metrics or []

        self.logger = EpochsLogger()

        self.best_loss = float('inf')
        self.current_epoch = 0

    def train_epoch(self):
        """Одна эпоха обучения"""
        self.model.train()
        running_loss = 0.0

        if self.verbose:
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        else:
            pbar = self.train_loader
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.loss_fn(outputs, masks)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            if self.verbose:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return running_loss / len(self.train_loader.dataset)

    def validate(self):
        """Валидация модели"""
        self.model.eval()
        running_loss = 0.0
        metric_results = {metric.name: 0.0 for metric in self.metrics}
        num_batches = 0

        with torch.no_grad():
            if self.verbose:
                pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            else:
                pbar = self.val_loader
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, masks)
                running_loss += loss.item() * images.size(0)

                for metric in self.metrics:
                    metric_value = metric(outputs, masks)
                    metric_results[metric.name] += metric_value

                num_batches += 1

        avg_loss = running_loss / len(self.val_loader.dataset)

        avg_metrics = {}
        if num_batches > 0:
            for name, total in metric_results.items():
                avg_metrics[name] = total / num_batches

        return avg_loss, avg_metrics

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
            val_loss, val_metrics = self.validate()

            # Шаг планировщика
            if self.scheduler:
                self.scheduler.step(val_loss)

            metrics_str = " | ".join([f"{name}: {value:.4f}" for name, value in val_metrics.items()])
            print(f"Epoch {epoch + 1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | {metrics_str}")

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
                metrics=val_metrics,
                path=log_path
            )

        print(f"Training finished. Best Val Loss: {self.best_loss:.4f}")
        return self.logger.epoch_logs

    def load_logs_and_continue(self, log_path='./logs/training_logs.json'):
        """Загрузка предыдущих логов (опционально, если нужно продолжить тренировку)"""
        if self.logger.load_logs_json(log_path):
            print("Previous logs loaded.")
            # Здесь можно добавить логику восстановления состояния, если нужно
