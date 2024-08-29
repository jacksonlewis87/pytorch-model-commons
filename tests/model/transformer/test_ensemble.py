import pytest
import torch
from unittest.mock import MagicMock, patch

from model.transformer.ensemble import GPT, TransformerModule
from model.configs.transformer.ensemble_configs import GPTConfig, TransformerModuleConfig


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
@patch("model.transformer.ensemble.TransformerBlock", return_value=DummyTransformerBlock())
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


@pytest.fixture
def transformer_module_config():
    config = MagicMock(spec=TransformerModuleConfig)
    config.input_dim = 32
    config.hidden_dim = 64
    config.output_dim = 128
    config.block_size = 256
    config.embed_drop_p = 0.1
    config.num_layers = 6
    config.block_config = MagicMock()
    return config


@pytest.fixture
@patch("model.transformer.ensemble.TransformerBlock", return_value=DummyTransformerBlock())
def transformer_module(mock_transformer_block, transformer_module_config):
    return TransformerModule(config=transformer_module_config)


def test_transformer_module_initialization(transformer_module, transformer_module_config):
    assert isinstance(transformer_module.proj_in, torch.nn.Linear)
    assert isinstance(transformer_module.proj_out, torch.nn.Linear)
    assert isinstance(transformer_module.pos_emb, torch.nn.Parameter)
    assert isinstance(transformer_module.drop, torch.nn.Dropout)
    assert isinstance(transformer_module.blocks, torch.nn.Sequential)
    assert transformer_module.input_dim == transformer_module_config.input_dim
    assert transformer_module.hidden_dim == transformer_module_config.hidden_dim
    assert transformer_module.output_dim == transformer_module_config.output_dim
    assert transformer_module.block_size == transformer_module_config.block_size
    assert transformer_module.config == transformer_module_config


def test_transformer_module_weight_initialization(transformer_module):
    mean_tolerance = 1e-3
    std_tolerance = 1e-3

    for module in transformer_module.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            assert torch.allclose(module.weight.data.mean(), torch.tensor(0.0), atol=mean_tolerance)
            assert torch.allclose(module.weight.data.std(), torch.tensor(0.02), atol=std_tolerance)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                assert torch.all(module.bias.data == 0.0)
        elif isinstance(module, torch.nn.LayerNorm):
            assert torch.all(module.bias.data == 0.0)
            assert torch.all(module.weight.data == 1.0)

    for param in transformer_module.parameters():
        if param is not transformer_module.pos_emb:
            assert not torch.isnan(param).any()


def test_transformer_module_forward_pass(transformer_module):
    batch_size = 4
    seq_length = 10
    dummy_embeddings = torch.randn(batch_size, seq_length, transformer_module.input_dim)

    output = transformer_module.forward(dummy_embeddings)

    assert output.shape == (batch_size, seq_length, transformer_module.output_dim)
