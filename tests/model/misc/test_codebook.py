import pytest
import torch

from pytorch_model_commons.model.configs.misc.codebook_configs import CodebookConfig
from pytorch_model_commons.model.misc.codebook import Codebook


@pytest.fixture
def codebook():
    config = CodebookConfig(codebook_size=4, embedding_dim=64, beta=0.25)
    return Codebook(config)


def test_codebook_forward(codebook):
    batch_size, temporal_size, embedding_dim = 2, 8, 64
    z = torch.randn(batch_size, temporal_size, embedding_dim)

    z_q, min_encoding_indices, loss = codebook(z)

    assert z_q.shape == z.shape
    assert min_encoding_indices.shape == (batch_size * temporal_size,)
    assert loss.ndim == 0
    assert min_encoding_indices.max().item() < codebook.codebook_size
    assert min_encoding_indices.min().item() >= 0


def test_codebook_loss_range(codebook):
    batch_size, temporal_size, embedding_dim = 2, 8, 64
    z = torch.randn(batch_size, temporal_size, embedding_dim)

    _, _, loss = codebook(z)

    assert loss >= 0
