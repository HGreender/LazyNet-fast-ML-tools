import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class LovaszBCELoss(nn.Module):
    """BCE + Lovász hinge для бинарной сегментации.

    Ожидает:
        logits:  (B, 1, H, W) или (B, H, W), сырые логиты (без sigmoid)
        targets: той же формы, значения 0/1
    """

    def __init__(
        self,
        lovasz_weight: float = 1.0,
        bce_weight: float = 1.0,
        pos_weight: float | None = None,
        per_image: bool = False,
        ignore_index: int | None = None,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.lovasz_weight = lovasz_weight
        # pos_weight как буфер, чтобы автоматически ехал на нужное устройство
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.pos_weight = None
        self.lovasz = smp.losses.LovaszLoss(
            mode="binary", per_image=per_image, ignore_index=ignore_index
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
        lovasz = self.lovasz(logits, targets)
        return self.lovasz_weight * lovasz + self.bce_weight * bce
