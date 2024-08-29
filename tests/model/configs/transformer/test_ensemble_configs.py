from model.configs.transformer.block_configs import AttentionType, TransformerBlockConfig
from model.configs.transformer.ensemble_configs import GPTConfig


def test_gpt_config():
    embedding_dim = 128
    vocab_size = 512
    num_layers = 5
    num_heads = 2
    block_size = 64
    embed_drop_p = 0.1
    attn_drop_p = 0.1
    resid_drop_p = 0.2

    result = GPTConfig(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        block_size=block_size,
        embed_drop_p=embed_drop_p,
        attn_drop_p=attn_drop_p,
        resid_drop_p=resid_drop_p,
        attention_type=AttentionType.STANDARD,
    )

    assert result.embedding_dim == embedding_dim
    assert result.vocab_size == vocab_size
    assert result.block_size == block_size
    assert result.num_layers == num_layers
    assert result.embed_drop_p == embed_drop_p
    assert isinstance(result.block_config, TransformerBlockConfig)
