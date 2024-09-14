from dataclasses import dataclass, field
from typing import Type

from pytorch_model_commons.model.configs.transformer.attention_configs import (
    AttentionConfig,
    AttentionType,
    CausalSelfAttentionConfig,
)
from pytorch_model_commons.utils import dict_to_dataclass


def get_attention_config_class(attention_type: AttentionType) -> Type[AttentionConfig]:
    if attention_type == AttentionType.CAUSAL:
        return CausalSelfAttentionConfig
    elif attention_type == AttentionType.STANDARD:
        return AttentionConfig
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")


@dataclass
class TransformerBlockConfig:
    embedding_dim: int
    resid_drop_p: float
    attention_type: str
    attention_config: AttentionConfig = field(init=False)

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attn_drop_p: float,
        resid_drop_p: float,
        attention_type: AttentionType,
        block_size: int = None,
    ):
        self.embedding_dim = embedding_dim
        self.resid_drop_p = resid_drop_p
        self.attention_type = attention_type.value

        attention_config_class = get_attention_config_class(attention_type=attention_type)

        self.attention_config = dict_to_dataclass(
            dataclass_type=attention_config_class,
            dict_obj={
                "embedding_dim": self.embedding_dim,
                "num_heads": num_heads,
                "block_size": block_size,
                "attn_drop_p": attn_drop_p,
                "resid_drop_p": self.resid_drop_p,
            },
        )
