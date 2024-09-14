from dataclasses import dataclass
from enum import Enum


class AttentionType(Enum):
    CAUSAL = "causal"
    STANDARD = "standard"


@dataclass
class AttentionConfig:
    embedding_dim: int
    num_heads: int
    attn_drop_p: float
    resid_drop_p: float


@dataclass
class CausalSelfAttentionConfig(AttentionConfig):
    block_size: int
