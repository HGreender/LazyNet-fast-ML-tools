import numpy as np
import matplotlib.pyplot as plt

from fast_ml_tools.models import EpochsData
from fast_ml_tools.logging import EpochsLogger


def plot_epochs_data(
        epoch_data: EpochsData = None,
        logs_path='./logs/training_logs.json',
        show_metrics: bool = True,
        metric_names: list[str] = None
):
    """
    Визуализация графиков обучения на одном изображении с двумя осями Y.

    Args:
        epoch_data: Объект с данными логов. Если None, загружается из файла.
        logs_path: Путь к JSON файлу с логами.
        show_metrics: Если True, добавляет метрики на график (правая ось Y).
        metric_names: Список конкретных имен метрик для отображения.
                      Если None, отображаются все найденные метрики.
    """
    epoch_logger = EpochsLogger()
    if epoch_logger.load_logs_json(logs_path):
        epoch_data = epoch_logger.epoch_logs
    elif epoch_data is None:
        print("Не удалось загрузить логи для визуализации")
        return


    epochs = epoch_data.epochs
    train_losses = epoch_data.train_losses
    val_losses = epoch_data.val_losses
    metrics_list = epoch_data.metrics

    if not epochs:
        print("Нет данных для построения графика")
        return

    # 2. Настройка фигуры и первой оси (Loss)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Графики потерь (Левая ось)
    line_train, = ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    line_val, = ax1.plot(epochs, val_losses, 'r--', label='Validation Loss', linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color='black', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # 3. Добавление метрик (Правая ось), если нужно
    if show_metrics and metrics_list:
        ax2 = ax1.twinx()  # Создаем вторую ось, разделяющую ту же X

        # Определяем какие метрики рисовать
        available_metric_names = list(metrics_list[0].keys()) if metrics_list else []

        if specific_names := metric_names:
            names_to_plot = []
            for name in specific_names:
                # Ищем соответствие в доступных ключах (которые уже в нижнем регистре из-за load_logs_json)
                if name.lower() in available_metric_names:
                    names_to_plot.append(name)  # Оставляем оригинальное имя для легенды
        else:
            names_to_plot = available_metric_names  # Тут уже ключи из словаря (в нижнем регистре)

        if names_to_plot:
            colors = plt.cm.tab10(np.linspace(0, 1, len(names_to_plot)))

            for i, metric_name in enumerate(names_to_plot):
                search_key = metric_name.lower()
                metric_values = [epoch_metrics.get(search_key, np.nan) for epoch_metrics in metrics_list]
                ax2.plot(epochs, metric_values, marker='o', linestyle='-',
                         color=colors[i], label=metric_name, linewidth=2, markersize=4)

            ax2.set_ylabel('Metrics', color='black', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='black')

    # 4. Объединение легенд с обеих осей
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    if show_metrics and metrics_list and names_to_plot:
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=10)
    else:
        ax1.legend(loc='best', fontsize=10)

    plt.title('Training History: Losses & Metrics', fontsize=14)
    plt.tight_layout()
    plt.show()

    # 5. Вывод статистики в консоль
    best_idx = np.argmin(val_losses)
    print(f"--- Best Loss ---")
    print(f"Best Epoch: {epochs[best_idx]}")
    print(f"Min Val Loss: {val_losses[best_idx]:.4f}")

    if show_metrics and metrics_list and names_to_plot:
        print(f"\n--- Best Metrics ---")
        for metric_name in names_to_plot:
            metric_values = [epoch_metrics.get(metric_name, np.nan) for epoch_metrics in metrics_list]
            valid_values = [v for v in metric_values if not np.isnan(v)]
            if valid_values:
                best_val = max(valid_values)
                best_epoch_idx = metric_values.index(best_val)
                print(f"Max {metric_name}: {best_val:.4f} at Epoch {epochs[best_epoch_idx]}")
