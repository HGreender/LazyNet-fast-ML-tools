from dataclasses import dataclass, field


@dataclass
class EpochsData:
    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_loss: float = 1.0
    metrics: list[dict[str, float]] = field(default_factory=list)
