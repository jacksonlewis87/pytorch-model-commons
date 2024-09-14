import pytest
import torch
from unittest.mock import MagicMock, patch

from pytorch_model_commons.model.transformer.blocks import TransformerBlock
from pytorch_model_commons.model.configs.transformer.block_configs import TransformerBlockConfig


class DummyAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x, layer_past=None):
        return x, layer_past


@pytest.fixture
def transformer_block_config():
    config = MagicMock(spec=TransformerBlockConfig)
    config.embedding_dim = 512
    config.resid_drop_p = 0.1
    config.attention_config = MagicMock()
    config.attention_type = "some-attention-type"
    return config


@pytest.fixture
@patch("pytorch_model_commons.model.transformer.blocks.get_attention_module", return_value=DummyAttention)
def transformer_block(mock_get_attention_module, transformer_block_config):
    return TransformerBlock(config=transformer_block_config)


def test_transformer_block_initialization(transformer_block, transformer_block_config):
    assert isinstance(transformer_block.ln1, torch.nn.LayerNorm)
    assert isinstance(transformer_block.ln2, torch.nn.LayerNorm)
    assert isinstance(transformer_block.attn, DummyAttention)
    assert isinstance(transformer_block.mlp, torch.nn.Sequential)
    assert isinstance(transformer_block.mlp[0], torch.nn.Linear)
    assert isinstance(transformer_block.mlp[1], torch.nn.GELU)
    assert isinstance(transformer_block.mlp[2], torch.nn.Linear)
    assert isinstance(transformer_block.mlp[3], torch.nn.Dropout)

    assert transformer_block.ln1.normalized_shape[0] == transformer_block_config.embedding_dim
    assert transformer_block.ln2.normalized_shape[0] == transformer_block_config.embedding_dim


def test_transformer_block_forward_pass(transformer_block, transformer_block_config):
    batch_size = 4
    seq_length = 10
    embedding_dim = transformer_block_config.embedding_dim
    dummy_input = torch.randn(batch_size, seq_length, embedding_dim)

    output = transformer_block(dummy_input)

    assert output.shape == (batch_size, seq_length, embedding_dim)


def test_transformer_block_forward_pass_with_layer_past(transformer_block, transformer_block_config):
    batch_size = 4
    seq_length = 10
    embedding_dim = transformer_block_config.embedding_dim
    dummy_input = torch.randn(batch_size, seq_length, embedding_dim)
    dummy_past_key = torch.randn(
        batch_size,
        transformer_block_config.attention_config.num_heads,
        seq_length,
        embedding_dim // transformer_block_config.attention_config.num_heads,
    )
    dummy_past_value = torch.randn(
        batch_size,
        transformer_block_config.attention_config.num_heads,
        seq_length,
        embedding_dim // transformer_block_config.attention_config.num_heads,
    )
    dummy_layer_past = (dummy_past_key, dummy_past_value)

    output, present = transformer_block(dummy_input, layer_past=dummy_layer_past)

    assert output.shape == (batch_size, seq_length, embedding_dim)
    assert isinstance(present, tuple)
    assert len(present) == 2
    assert present[0].shape == dummy_past_key.shape
    assert present[1].shape == dummy_past_value.shape


def test_forward_pass_with_return_present(transformer_block, transformer_block_config):
    transformer_block.eval()
    batch_size = 4
    seq_length = 10
    embedding_dim = transformer_block_config.embedding_dim
    dummy_input = torch.randn(batch_size, seq_length, embedding_dim)
    dummy_past_key = torch.randn(
        batch_size,
        transformer_block_config.attention_config.num_heads,
        seq_length,
        embedding_dim // transformer_block_config.attention_config.num_heads,
    )
    dummy_past_value = torch.randn(
        batch_size,
        transformer_block_config.attention_config.num_heads,
        seq_length,
        embedding_dim // transformer_block_config.attention_config.num_heads,
    )
    dummy_layer_past = (dummy_past_key, dummy_past_value)

    output, present = transformer_block(dummy_input, layer_past=dummy_layer_past)

    assert output.shape == (batch_size, seq_length, embedding_dim)
    assert isinstance(present, tuple)
    assert len(present) == 2
    assert present[0].shape == dummy_past_key.shape
    assert present[1].shape == dummy_past_value.shape
