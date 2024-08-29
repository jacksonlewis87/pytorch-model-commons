from dataclasses import dataclass, field

from model.configs.transformer.block_configs import AttentionType, TransformerBlockConfig


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
        attention_type: AttentionType,
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
            attention_type=attention_type,
        )


@dataclass
class TransformerModuleConfig:
    embedding_dim: int
    vocab_size: int
    block_size: int
    num_layers: int
    embed_drop_p: float
    block_config: TransformerBlockConfig = field(init=False)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        block_size: int,
        embed_drop_p: float,
        attn_drop_p: float,
        resid_drop_p: float,
        attention_type: AttentionType,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.block_size = block_size
        self.num_layers = num_layers
        self.embed_drop_p = embed_drop_p

        self.block_config = TransformerBlockConfig(
            embedding_dim=self.hidden_dim,
            num_heads=num_heads,
            block_size=self.block_size,
            attn_drop_p=attn_drop_p,
            resid_drop_p=resid_drop_p,
            attention_type=attention_type,
        )
