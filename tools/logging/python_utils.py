import json
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
