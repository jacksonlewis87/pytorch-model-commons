from pytorch_model_commons.model.configs.transformer.block_configs import AttentionType, TransformerBlockConfig
from pytorch_model_commons.model.configs.transformer.ensemble_configs import GPTConfig, TransformerModuleConfig


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


def test_transformer_module_config():
    input_dim = 64
    hidden_dim = 128
    output_dim = 256
    num_layers = 5
    num_heads = 2
    block_size = 64
    embed_drop_p = 0.1
    attn_drop_p = 0.1
    resid_drop_p = 0.2

    result = TransformerModuleConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        block_size=block_size,
        embed_drop_p=embed_drop_p,
        attn_drop_p=attn_drop_p,
        resid_drop_p=resid_drop_p,
        attention_type=AttentionType.STANDARD,
    )

    assert result.input_dim == input_dim
    assert result.hidden_dim == hidden_dim
    assert result.output_dim == output_dim
    assert result.block_size == block_size
    assert result.num_layers == num_layers
    assert result.embed_drop_p == embed_drop_p
    assert isinstance(result.block_config, TransformerBlockConfig)
