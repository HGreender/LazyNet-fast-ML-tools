import torch.nn as nn
import segmentation_models_pytorch as smp


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss() # Работает с логитами (activation=None в модели)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, outputs, masks):
        dice_loss = self.dice(outputs, masks)
        bce_loss = self.bce(outputs, masks)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss
