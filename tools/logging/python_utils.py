import json
import os

from tools.models import EpochsData

class EpochsLogger:
    def __init__(self):
        self.epoch_logs = EpochsData()

    def add_logs(self, epoch, train_loss, val_loss, best_loss):
        """Сохранение логов обучения в JSON"""
        self.epoch_logs.best_loss = best_loss
        self.epoch_logs.epochs.append(epoch)
        self.epoch_logs.train_losses.append(train_loss)
        self.epoch_logs.val_losses.append(val_loss)

    def save_logs_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logs_data = {
            'logs': {
                'train_losses': [float(x) for x in self.epoch_logs.train_losses],
                'val_losses': [float(x) for x in self.epoch_logs.val_losses],
                'epochs': [int(x) for x in self.epoch_logs.epochs]
            },
            'best_val_loss': float(self.epoch_logs.best_loss)
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False)

    def add_and_save_logs_json(
            self,
            epoch, train_loss, val_loss, best_loss,
            path='./logs/training_logs.json'
    ):
        """Сохранение логов обучения в JSON"""
        self.add_logs(epoch, train_loss, val_loss, best_loss)
        self.save_logs_json(path)

    def load_logs_json(self, path='./logs/training_logs.json'):
        """Загрузка логов из JSON файла"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logs = data.get('logs', {})
            self.epoch_logs.epochs = logs.get('epochs', [])
            self.epoch_logs.train_losses = logs.get('train_losses', [])
            self.epoch_logs.val_losses = logs.get('val_losses', [])
            self.epoch_logs.best_loss = data.get('best_val_loss', 1.0)

            print(f"Загружено {len(self.epoch_logs.epochs)} эпох из {path}")
            return True

        except FileNotFoundError:
            print(f"Файл логов не найден: {path}")
            return False
        except json.JSONDecodeError:
            print(f"Ошибка чтения JSON: {path}")
            return False
        except Exception as e:
            print(f"Неожиданная ошибка при загрузке: {e}")
            return False
