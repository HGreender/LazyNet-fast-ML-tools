from .focal import FocalLoss
import torch.nn as nn
import segmentation_models_pytorch as smp


class IoUFocalLoss(nn.Module):
    def __init__(self, iou_weight=0.5, focal_weight=0.5,
                 alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Комбинированная функция потерь: IoU Loss + Focal Loss для бинарной сегментации.

        Args:
            iou_weight: Вес IoU Loss в итоговой сумме
            focal_weight: Вес Focal Loss в итоговой сумме
            alpha: Балансирующий параметр для компенсации дисбаланса классов в Focal Loss
            gamma: Фокусирующий параметр в Focal Loss (уменьшает вклад легко классифицируемых примеров)
            reduction: Способ агрегации потерь в Focal Loss ('mean', 'sum', или 'none')
        """
        super().__init__()
        self.iou = smp.losses.JaccardLoss(mode='binary')  # Jaccard = IoU
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.iou_weight = iou_weight
        self.focal_weight = focal_weight

    def forward(self, outputs, masks):
        """
        Args:
            outputs: Логиты модели (до применения сигмоиды), shape [B, C, H, W]
            masks: Ground truth маски, shape [B, C, H, W]

        Returns:
            Взвешенная сумма IoU Loss и Focal Loss
        """
        iou_loss = self.iou(outputs, masks)
        focal_loss = self.focal(outputs, masks)

        return self.iou_weight * iou_loss + self.focal_weight * focal_loss
