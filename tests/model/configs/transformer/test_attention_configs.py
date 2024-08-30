from pytorch_model_commons.model.configs.transformer.attention_configs import AttentionConfig, CausalSelfAttentionConfig


def test_attention_config():
    embedding_dim = 128
    num_heads = 2
    attn_drop_p = 0.1
    resid_drop_p = 0.2

    result = AttentionConfig(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        attn_drop_p=attn_drop_p,
        resid_drop_p=resid_drop_p,
    )

    assert result.embedding_dim == embedding_dim
    assert result.num_heads == num_heads
    assert result.attn_drop_p == attn_drop_p
    assert result.resid_drop_p == resid_drop_p


def test_causal_self_attention_config():
    embedding_dim = 128
    num_heads = 2
    block_size = 64
    attn_drop_p = 0.1
    resid_drop_p = 0.2

    result = CausalSelfAttentionConfig(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        block_size=block_size,
        attn_drop_p=attn_drop_p,
        resid_drop_p=resid_drop_p,
    )

    assert result.embedding_dim == embedding_dim
    assert result.num_heads == num_heads
    assert result.block_size == block_size
    assert result.attn_drop_p == attn_drop_p
    assert result.resid_drop_p == resid_drop_p
