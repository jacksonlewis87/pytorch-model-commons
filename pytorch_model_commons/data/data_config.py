from dataclasses import dataclass


@dataclass
class DataConfig:
    input_path: str
    batch_size: int
    train_size: float
    data_split_path: str
