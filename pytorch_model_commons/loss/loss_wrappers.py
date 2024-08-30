from dataclasses import dataclass
from torch import Tensor


@dataclass
class BaseLossWrapper:
    total: Tensor
