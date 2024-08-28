from dataclasses import dataclass, field

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


@dataclass
class CausalSelfAttentionConfig:
    embedding_dim: int
    num_heads: int
    block_size: int
    attn_drop_p: float
    resid_drop_p: float


@dataclass
class TransformerBlockConfig:
    embedding_dim: int
    resid_drop_p: float
    attention_config: CausalSelfAttentionConfig = field(init=False)

    def __init__(self, embedding_dim: int, num_heads: int, block_size: int, attn_drop_p: float, resid_drop_p: float):
        self.embedding_dim = embedding_dim
        self.resid_drop_p = resid_drop_p

        self.attention_config = CausalSelfAttentionConfig(
            embedding_dim=self.embedding_dim,
            num_heads=num_heads,
            block_size=block_size,
            attn_drop_p=attn_drop_p,
            resid_drop_p=self.resid_drop_p,
        )


@dataclass
class GPTConfig:
    embedding_dim: int
    vocab_size: int
    block_size: int
    num_layers: int
    embed_drop_p: float
    block_config: TransformerBlockConfig = field(init=False)

    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        block_size: int,
        embed_drop_p: float,
        attn_drop_p: float,
        resid_drop_p: float,
    ):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_layers = num_layers
        self.embed_drop_p = embed_drop_p

        self.block_config = TransformerBlockConfig(
            embedding_dim=self.embedding_dim,
            num_heads=num_heads,
            block_size=self.block_size,
            attn_drop_p=attn_drop_p,
            resid_drop_p=resid_drop_p,
        )
