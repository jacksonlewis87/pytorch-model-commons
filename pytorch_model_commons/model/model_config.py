from dataclasses import dataclass

from data.data_config import DataConfig


@dataclass
class ModelConfig:
    learning_rate: float
    epochs: int
    checkpoint_path: str


@dataclass
class FullConfig:
    experiment_path: str
    data_config: DataConfig
    model_config: ModelConfig
