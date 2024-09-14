import pytest
import torch
from unittest.mock import MagicMock, patch

from pytorch_model_commons.model.transformer.attention import (
    CausalSelfAttention,
    ScaledDotProductSelfAttention,
    get_attention_module,
)
from pytorch_model_commons.model.configs.transformer.attention_configs import (
    AttentionConfig,
    AttentionType,
    CausalSelfAttentionConfig,
)


@pytest.mark.parametrize(
    "attention_type, expected_module_str",
    [
        (AttentionType.CAUSAL.value, "CausalSelfAttention"),
        (AttentionType.STANDARD.value, "ScaledDotProductSelfAttention"),
    ],
)
def test_get_attention_module(attention_type: str, expected_module_str: str):
    with patch(f"pytorch_model_commons.model.transformer.attention.{expected_module_str}") as MockAttentionModule:
        module_class = get_attention_module(attention_type=attention_type)

        assert module_class == MockAttentionModule


def test_get_attention_module_invalid():
    with pytest.raises(ValueError, match="Unsupported attention type"):
        get_attention_module(attention_type="invalid_attention_type")


@pytest.fixture
def attention_config():
    config = MagicMock(spec=AttentionConfig)
    config.embedding_dim = 512
    config.num_heads = 2
    config.attn_drop_p = 0.1
    config.resid_drop_p = 0.1
    return config


@pytest.fixture
def scaled_dot_product_self_attention(attention_config):
    return ScaledDotProductSelfAttention(config=attention_config)


def test_scaled_dot_product_self_attention_initialization(scaled_dot_product_self_attention, attention_config):
    assert scaled_dot_product_self_attention.num_heads == attention_config.num_heads
    assert scaled_dot_product_self_attention.embedding_dim == attention_config.embedding_dim
    assert isinstance(scaled_dot_product_self_attention.key, torch.nn.Linear)
    assert isinstance(scaled_dot_product_self_attention.query, torch.nn.Linear)
    assert isinstance(scaled_dot_product_self_attention.value, torch.nn.Linear)
    assert isinstance(scaled_dot_product_self_attention.attn_drop, torch.nn.Dropout)
    assert isinstance(scaled_dot_product_self_attention.resid_drop, torch.nn.Dropout)
    assert isinstance(scaled_dot_product_self_attention.proj, torch.nn.Linear)


def test_scaled_dot_product_self_attention_forward_pass(scaled_dot_product_self_attention, attention_config):
    batch_size = 4
    seq_length = 10
    embedding_dim = attention_config.embedding_dim
    dummy_input = torch.randn(batch_size, seq_length, embedding_dim)

    output, _ = scaled_dot_product_self_attention(dummy_input)

    assert output.shape == (batch_size, seq_length, embedding_dim)


def test_scaled_dot_product_self_attention_forward_pass_with_layer_past(
    scaled_dot_product_self_attention, attention_config
):
    batch_size = 4
    seq_length = 10
    embedding_dim = attention_config.embedding_dim
    dummy_input = torch.randn(batch_size, seq_length, embedding_dim)
    past_key = torch.randn(
        batch_size,
        attention_config.num_heads,
        seq_length,
        embedding_dim // attention_config.num_heads,
    )
    past_value = torch.randn(
        batch_size,
        attention_config.num_heads,
        seq_length,
        embedding_dim // attention_config.num_heads,
    )
    dummy_layer_past = (past_key, past_value)

    output, present = scaled_dot_product_self_attention(dummy_input, layer_past=dummy_layer_past)

    assert output.shape == (batch_size, seq_length, embedding_dim)
    assert isinstance(present, tuple)
    assert len(present) == 2
    assert present[0].shape == (
        batch_size,
        attention_config.num_heads,
        seq_length + seq_length,
        embedding_dim // attention_config.num_heads,
    )
    assert present[1].shape == (
        batch_size,
        attention_config.num_heads,
        seq_length + seq_length,
        embedding_dim // attention_config.num_heads,
    )


@pytest.fixture
def causal_self_attention_config():
    config = MagicMock(spec=CausalSelfAttentionConfig)
    config.embedding_dim = 512
    config.num_heads = 2
    config.block_size = 128
    config.attn_drop_p = 0.1
    config.resid_drop_p = 0.1
    return config


@pytest.fixture
def causal_self_attention(causal_self_attention_config):
    return CausalSelfAttention(config=causal_self_attention_config)


def test_causal_self_attention_initialization(causal_self_attention, causal_self_attention_config):
    assert causal_self_attention.num_heads == causal_self_attention_config.num_heads
    assert causal_self_attention.embedding_dim == causal_self_attention_config.embedding_dim
    assert isinstance(causal_self_attention.key, torch.nn.Linear)
    assert isinstance(causal_self_attention.query, torch.nn.Linear)
    assert isinstance(causal_self_attention.value, torch.nn.Linear)
    assert isinstance(causal_self_attention.attn_drop, torch.nn.Dropout)
    assert isinstance(causal_self_attention.resid_drop, torch.nn.Dropout)
    assert isinstance(causal_self_attention.proj, torch.nn.Linear)
    assert causal_self_attention.mask.shape == (
        1,
        1,
        causal_self_attention_config.block_size,
        causal_self_attention_config.block_size,
    )


def test_causal_self_attention_forward_pass(causal_self_attention, causal_self_attention_config):
    batch_size = 4
    seq_length = 10
    embedding_dim = causal_self_attention_config.embedding_dim
    dummy_input = torch.randn(batch_size, seq_length, embedding_dim)

    output, _ = causal_self_attention(dummy_input)

    assert output.shape == (batch_size, seq_length, embedding_dim)


def test_causal_self_attention_forward_pass_with_layer_past(causal_self_attention, causal_self_attention_config):
    batch_size = 4
    seq_length = 10
    embedding_dim = causal_self_attention_config.embedding_dim
    dummy_input = torch.randn(batch_size, seq_length, embedding_dim)
    past_key = torch.randn(
        batch_size,
        causal_self_attention_config.num_heads,
        seq_length,
        embedding_dim // causal_self_attention_config.num_heads,
    )
    past_value = torch.randn(
        batch_size,
        causal_self_attention_config.num_heads,
        seq_length,
        embedding_dim // causal_self_attention_config.num_heads,
    )
    dummy_layer_past = (past_key, past_value)

    output, present = causal_self_attention(dummy_input, layer_past=dummy_layer_past)

    assert output.shape == (batch_size, seq_length, embedding_dim)
    assert isinstance(present, tuple)
    assert len(present) == 2
    assert present[0].shape == (
        batch_size,
        causal_self_attention_config.num_heads,
        seq_length + seq_length,
        embedding_dim // causal_self_attention_config.num_heads,
    )
    assert present[1].shape == (
        batch_size,
        causal_self_attention_config.num_heads,
        seq_length + seq_length,
        embedding_dim // causal_self_attention_config.num_heads,
    )


def test_causal_self_attention_mask(causal_self_attention, causal_self_attention_config):
    mask = causal_self_attention.mask.squeeze(0).squeeze(0)
    assert mask.shape[0] == causal_self_attention_config.block_size
    assert torch.all(mask.tril() == mask)
