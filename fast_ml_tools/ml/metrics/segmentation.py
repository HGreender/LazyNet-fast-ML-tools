import torch


class BaseMetric:
    """Базовый класс для метрик"""

    def __init__(self, name: str = "metric"):
        self.name = name

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        raise NotImplementedError


class IoUMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        super().__init__(name="IoU")
        self.threshold = threshold
        self.eps = eps

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        # preds: [B, C, H, W] или [B, H, W]
        # targets: [B, C, H, W] или [B, H, W]

        # Бинаризация предсказаний
        preds_binary = (preds > self.threshold).float()

        # Intersection over Union
        intersection = (preds_binary * targets).sum(dim=(1, 2, 3))
        union = preds_binary.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + self.eps) / (union + self.eps)
        return iou.mean().item()


class DiceMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        super().__init__(name="Dice")
        self.threshold = threshold
        self.eps = eps

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        preds_binary = (preds > self.threshold).float()

        intersection = (preds_binary * targets).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + self.eps) / (
                    preds_binary.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.eps)

        return dice.mean().item()


# Удобные фабричные функции для быстрого импорта
def get_segmentation_metrics(metric_names: list = None):
    """Возвращает список метрик по именам"""
    if metric_names is None:
        metric_names = ["IoU", "Dice"]

    metrics_map = {
        "IoU": IoUMetric(),
        "Dice": DiceMetric(),
    }

    return [metrics_map[name] for name in metric_names if name in metrics_map]
