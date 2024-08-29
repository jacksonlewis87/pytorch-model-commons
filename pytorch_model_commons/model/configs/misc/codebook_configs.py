from dataclasses import dataclass


@dataclass
class CodebookConfig:
    codebook_size: int
    embedding_dim: int
    beta: float
