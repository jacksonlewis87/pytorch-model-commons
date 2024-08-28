import pytest
from unittest.mock import Mock, create_autospec, patch

from model.model_configs import CausalSelfAttentionConfig, FullConfig, GPTConfig, ModelConfig, TransformerBlockConfig


@pytest.fixture
def mock_data_config():
    with patch("model.model_configs.DataConfig") as MockDataConfig:
        MockDataConfig.return_value = Mock()
        yield MockDataConfig.return_value


@pytest.fixture
def mock_model_config():
    with patch("model.model_configs.ModelConfig") as MockModelConfig:
        MockModelConfig.return_value = create_autospec(ModelConfig)
        yield MockModelConfig.return_value


@pytest.fixture
def full_config(mock_data_config, mock_model_config):
    return FullConfig(
        experiment_path="/path/to/experiment", data_config=mock_data_config, model_config=mock_model_config
    )


def test_full_config_initialization(full_config, mock_data_config, mock_model_config):
    assert full_config.experiment_path == "/path/to/experiment"
    assert full_config.data_config == mock_data_config
    assert full_config.model_config == mock_model_config


def test_model_config():
    checkpoint_path = "/path/to/checkpoint"
    epochs = 2
    learning_rate = 0.01

    result = ModelConfig(
        checkpoint_path=checkpoint_path,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    assert result.checkpoint_path == checkpoint_path
    assert result.epochs == epochs
    assert result.learning_rate == learning_rate


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


def test_transformer_block_config():
    embedding_dim = 128
    num_heads = 2
    block_size = 64
    attn_drop_p = 0.1
    resid_drop_p = 0.2

    result = TransformerBlockConfig(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        block_size=block_size,
        attn_drop_p=attn_drop_p,
        resid_drop_p=resid_drop_p,
    )

    assert result.embedding_dim == embedding_dim
    assert result.resid_drop_p == resid_drop_p
    assert isinstance(result.attention_config, CausalSelfAttentionConfig)
    assert result.attention_config.embedding_dim == embedding_dim
    assert result.attention_config.num_heads == num_heads
    assert result.attention_config.block_size == block_size
    assert result.attention_config.attn_drop_p == attn_drop_p
    assert result.attention_config.resid_drop_p == resid_drop_p


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
    )

    assert result.embedding_dim == embedding_dim
    assert result.vocab_size == vocab_size
    assert result.block_size == block_size
    assert result.num_layers == num_layers
    assert result.embed_drop_p == embed_drop_p
    assert isinstance(result.block_config, TransformerBlockConfig)
    assert result.block_config.embedding_dim == embedding_dim
    assert result.block_config.resid_drop_p == resid_drop_p
