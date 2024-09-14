from unittest.mock import Mock, patch

from pytorch_model_commons.model.configs.transformer.ensemble_configs import GPTConfig, TransformerModuleConfig


@patch("pytorch_model_commons.model.configs.transformer.ensemble_configs.TransformerBlockConfig")
def test_gpt_config(mock_transformer_block_config):
    embedding_dim = 128
    vocab_size = 512
    num_layers = 5
    num_heads = 2
    block_size = 64
    embed_drop_p = 0.1
    attn_drop_p = 0.1
    resid_drop_p = 0.2
    attention_type = Mock()

    result = GPTConfig(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        block_size=block_size,
        embed_drop_p=embed_drop_p,
        attn_drop_p=attn_drop_p,
        resid_drop_p=resid_drop_p,
        attention_type=attention_type,
    )

    mock_transformer_block_config.assert_called_once_with(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        block_size=block_size,
        attn_drop_p=attn_drop_p,
        resid_drop_p=resid_drop_p,
        attention_type=attention_type,
    )
    assert result.embedding_dim == embedding_dim
    assert result.vocab_size == vocab_size
    assert result.block_size == block_size
    assert result.num_layers == num_layers
    assert result.embed_drop_p == embed_drop_p


@patch("pytorch_model_commons.model.configs.transformer.ensemble_configs.TransformerBlockConfig")
def test_transformer_module_config(mock_transformer_block_config):
    input_dim = 64
    hidden_dim = 128
    output_dim = 256
    num_layers = 5
    num_heads = 2
    block_size = 64
    embed_drop_p = 0.1
    attn_drop_p = 0.1
    resid_drop_p = 0.2
    attention_type = Mock()

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
        attention_type=attention_type,
    )

    mock_transformer_block_config.assert_called_once_with(
        embedding_dim=hidden_dim,
        num_heads=num_heads,
        block_size=block_size,
        attn_drop_p=attn_drop_p,
        resid_drop_p=resid_drop_p,
        attention_type=attention_type,
    )
    assert result.input_dim == input_dim
    assert result.hidden_dim == hidden_dim
    assert result.output_dim == output_dim
    assert result.block_size == block_size
    assert result.num_layers == num_layers
    assert result.embed_drop_p == embed_drop_p
