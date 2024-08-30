from dataclasses import dataclass


@dataclass
class BaseLossWrapper:
    total_loss: float
