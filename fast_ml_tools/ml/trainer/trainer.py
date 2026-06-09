import os
import torch
from tqdm import tqdm
from fast_ml_tools.logging import EpochsLogger


class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            early_stopping_threshold,
            metrics: list = None,
            scheduler=None,
            device='cuda',
            verbose=True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.early_stopping_threshold = early_stopping_threshold
        self.scheduler = scheduler
        self.device = device
        self.verbose = verbose
        self.metrics = metrics or []

        # Перемещаем метрики на нужное устройство
        for metric in self.metrics:
            metric.to(self.device)

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

        # Сбрасываем состояние всех метрик перед новой фазой валидации
        for metric in self.metrics:
            metric.reset()

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

                # 1. Подготавливаем Target (маски)
                # DiceScore(input_format="index") требует LongTensor со значениями классов (0, 1...)
                target_indices = masks.long()
                # Если маска имеет форму [B, 1, H, W], убираем размерность канала до [B, H, W]
                if target_indices.dim() == 4 and target_indices.shape[1] == 1:
                    target_indices = target_indices.squeeze(1)

                # 2. Подготавливаем Preds (предсказания)
                # Сначала получаем вероятности (если outputs - это логиты)
                probs = torch.sigmoid(outputs) if outputs.min() < 0 else outputs

                # Бинаризуем по порогу 0.5, получаем 0 или 1
                pred_indices = (probs > 0.5).long()
                # Также убираем размерность канала, если она есть [B, 1, H, W] -> [B, H, W]
                if pred_indices.dim() == 4 and pred_indices.shape[1] == 1:
                    pred_indices = pred_indices.squeeze(1)

                # Обновляем метрики
                for metric in self.metrics:
                    # Для DiceScore с input_format="index" подаем подготовленные индексы
                    if isinstance(metric, __import__('torchmetrics.segmentation', fromlist=['DiceScore']).DiceScore):
                        metric.update(pred_indices, target_indices)
                    else:
                        # Для BinaryJaccardIndex и других метрик, которые могут принимать сырые данные
                        # оставляем оригинальные tensors или адаптируем под них при необходимости
                        metric.update(outputs, masks)

                num_batches += 1

        avg_loss = running_loss / len(self.val_loader.dataset)

        avg_metrics = {}
        for metric in self.metrics:
            try:
                val = metric.compute()
                avg_metrics[metric.__class__.__name__] = val.item() if val.numel() == 1 else val
            except Exception as e:
                print(f"Ошибка при вычислении метрики {metric.__class__.__name__}: {e}")
                avg_metrics[metric.__class__.__name__] = 0.0

        return avg_loss, avg_metrics

    def save_best_model(self, path='./models/best_model.pth'):
        """Сохранение лучшей модели"""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Saved best model to {path}")

    def save_checkpoint(self, path, extra_state=None):
        """Сохранение полного состояния тренировки"""
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, path)
        if self.verbose:
            print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        """Загрузка состояния для продолжения обучения"""
        if not os.path.exists(path):
            return False

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.verbose:
            print(f"Loaded checkpoint from {path}. Resuming from epoch {self.current_epoch + 1}")
        return True

    def fit(
            self,
            epochs,
            save_path='./models/best_model.pth',
            log_path='./logs/training_logs.json',
            save_checkpoint_path = None,
            resume_from = None
    ):
        """Основной цикл тренировки"""
        if resume_from:
            self.load_checkpoint(resume_from)
            if self.logger.load_logs_json(log_path):
                if self.logger.epoch_logs.epochs:
                    last_logged_epoch = self.logger.epoch_logs.epochs[-1]
                    start_epoch = max(self.current_epoch, last_logged_epoch + 1)
                else:
                    start_epoch = self.current_epoch
            else:
                start_epoch = self.current_epoch
            print(f"Resuming training. Start from epoch {start_epoch}")
        else:
            start_epoch = self.current_epoch

        print(f"Start training on {self.device} | Epochs: {start_epoch + 1} -> {epochs}")

        early_stopping_counter = 0

        for epoch in range(start_epoch, epochs):
            if early_stopping_counter >= self.early_stopping_threshold:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {self.best_loss:.4f}")
                break

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
                if save_checkpoint_path:
                    self.save_checkpoint(save_checkpoint_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.early_stopping_threshold:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best Val Loss: {self.best_loss:.4f}")
                    break

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
