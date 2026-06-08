from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.segmentation import DiceScore


def get_segmentation_metrics(metric_names: list = None):
    """
    Возвращает список метрик torchmetrics по именам.

    Args:
        metric_names: Список имен метрик. Поддерживаются: 'IoU', 'Dice'.
                      Если None, возвращает IoU и Dice.

    Returns:
        List[torchmetrics.Metric]: Список объектов метрик.
    """
    if metric_names is None:
        metric_names = ["IoU", "Dice"]

    metrics_map = {
        # BinaryJaccardIndex - это то же самое, что и IoU для бинарного случая
        "IoU": BinaryJaccardIndex(threshold=0.5),
        # BinaryDice рассчитывает коэффициент Дайса
        "Dice": DiceScore(
            num_classes=2,           # Фон (0) и Объект (1)
            average="macro",         # Усреднение по классам (рекомендуется)
            input_format="one-hot"   # Ожидает тензоры формы [B, C, H, W]
        )
    }

    selected_metrics = []
    for name in metric_names:
        if name in metrics_map:
            # Клонируем метрику, чтобы каждый экземпляр имел свое состояние
            selected_metrics.append(metrics_map[name].clone())
        else:
            print(f"Предупреждение: Метрика '{name}' не найдена. Доступны: {list(metrics_map.keys())}")

    return selected_metrics
