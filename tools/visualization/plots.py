import json

import numpy as np
import matplotlib.pyplot as plt

from tools.models import EpochsData


def plot_epochs_data(
        epoch_data: EpochsData = None,
        logs_path='./logs/training_logs.json'
):
    """Визуализация графиков обучения из JSON"""
    if epoch_data is None:
        try:
            with open(logs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print("Файл логов не найден. Пропуск визуализации.")
            return

        logs = data.get('logs', {})
        epochs = logs.get('epochs', [])
        train_losses = logs.get('train_losses', [])
        val_losses = logs.get('val_losses', [])
    else:
        epochs = epoch_data.epochs
        train_losses = epoch_data.train_losses
        val_losses = epoch_data.val_losses

    if not epochs:
        print("Нет данных для построения графика")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

    best_idx = np.argmin(val_losses)
    print(f"✅ Лучший epoch: {epochs[best_idx]}")
    print(f"📉 Минимальный val_loss: {val_losses[best_idx]:.4f}")
