import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss для бинарной сегментации.

        Args:
            alpha: Балансирующий параметр для компенсации дисбаланса классов
            gamma: Фокусирующий параметр, который уменьшает вклад легко классифицируемых примеров
            reduction: Способ агрегации потерь ('mean', 'sum', или 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, masks):
        """
        Args:
            outputs: Логиты модели (до применения сигмоиды), shape [B, C, H, W]
            masks: Ground truth маски, shape [B, C, H, W]
        """
        # Применяем сигмоиду к логитам
        probs = torch.sigmoid(outputs)

        # BCE loss поэлементно
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs, masks, reduction='none'
        )

        # Вычисляем фокусирующий коэффициент
        pt = torch.where(masks == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        # Комбинируем с alpha для балансировки классов
        alpha_factor = torch.where(masks == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * focal_weight

        # Итоговая потеря
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
