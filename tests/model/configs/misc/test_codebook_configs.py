from pytorch_model_commons.model.configs.misc.codebook_configs import CodebookConfig


def test_codebook_config():
    codebook_size = 128
    embedding_dim = 64
    beta = 0.1

    result = CodebookConfig(
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        beta=beta,
    )

    assert result.codebook_size == codebook_size
    assert result.embedding_dim == embedding_dim
    assert result.beta == beta
