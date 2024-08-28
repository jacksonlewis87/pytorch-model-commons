import pytest
import torch
from unittest.mock import MagicMock, patch

from model.blocks import CausalSelfAttention, GPT, TransformerBlock
from model.model_configs import CausalSelfAttentionConfig, GPTConfig, TransformerBlockConfig


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


class DummyCausalSelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, layer_past=None):
        return x, layer_past


@pytest.fixture
def transformer_block_config():
    config = MagicMock(spec=TransformerBlockConfig)
    config.embedding_dim = 512
    config.resid_drop_p = 0.1
    config.attention_config = MagicMock()
    return config


@pytest.fixture
@patch("model.blocks.CausalSelfAttention", return_value=DummyCausalSelfAttention())
def transformer_block(mock_causal_self_attention, transformer_block_config):
    return TransformerBlock(config=transformer_block_config)


def test_transformer_block_initialization(transformer_block, transformer_block_config):
    assert isinstance(transformer_block.ln1, torch.nn.LayerNorm)
    assert isinstance(transformer_block.ln2, torch.nn.LayerNorm)
    assert isinstance(transformer_block.attn, DummyCausalSelfAttention)
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


class DummyTransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


@pytest.fixture
def gpt_config():
    config = MagicMock(spec=GPTConfig)
    config.embedding_dim = 512
    config.vocab_size = 1000
    config.block_size = 128
    config.embed_drop_p = 0.1
    config.num_layers = 6
    config.block_config = MagicMock()
    return config


@pytest.fixture
@patch("model.blocks.TransformerBlock", return_value=DummyTransformerBlock())
def gpt(mock_transformer_block, gpt_config):
    return GPT(config=gpt_config)


def test_gpt_initialization(gpt, gpt_config):
    assert isinstance(gpt.tok_emb, torch.nn.Embedding)
    assert isinstance(gpt.pos_emb, torch.nn.Parameter)
    assert isinstance(gpt.drop, torch.nn.Dropout)
    assert isinstance(gpt.blocks, torch.nn.Sequential)
    assert isinstance(gpt.ln_f, torch.nn.LayerNorm)
    assert isinstance(gpt.head, torch.nn.Linear)
    assert gpt.block_size == gpt_config.block_size
    assert gpt.config == gpt_config


def test_gpt_weight_initialization(gpt):
    mean_tolerance = 1e-3
    std_tolerance = 1e-3

    for module in gpt.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            assert torch.allclose(module.weight.data.mean(), torch.tensor(0.0), atol=mean_tolerance)
            assert torch.allclose(module.weight.data.std(), torch.tensor(0.02), atol=std_tolerance)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                assert torch.all(module.bias.data == 0.0)
        elif isinstance(module, torch.nn.LayerNorm):
            assert torch.all(module.bias.data == 0.0)
            assert torch.all(module.weight.data == 1.0)


def test_gpt_forward_pass(gpt, gpt_config):
    batch_size = 4
    seq_length = 10
    dummy_idx = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_length))

    logits, _ = gpt.forward(dummy_idx)

    assert logits.shape == (batch_size, seq_length, gpt_config.vocab_size)


def test_gpt_forward_pass_with_embeddings(gpt, gpt_config):
    batch_size = 4
    seq_length = 10
    dummy_idx = torch.randint(0, gpt_config.vocab_size, (batch_size, seq_length))
    dummy_embeddings = torch.randn(batch_size, seq_length, gpt_config.embedding_dim)

    logits, _ = gpt.forward(dummy_idx, embeddings=dummy_embeddings)

    assert logits.shape == (batch_size, seq_length + dummy_embeddings.shape[1], gpt_config.vocab_size)
