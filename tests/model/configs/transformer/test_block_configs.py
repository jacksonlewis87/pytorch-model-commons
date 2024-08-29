import pytest
from unittest.mock import Mock, patch

from model.configs.transformer.block_configs import AttentionType, TransformerBlockConfig, get_attention_classes


@pytest.mark.parametrize(
    "attention_type, expected_config_str, expected_module_str",
    [
        (AttentionType.CAUSAL, "CausalSelfAttentionConfig", "CausalSelfAttention"),
        (AttentionType.STANDARD, "AttentionConfig", "ScaledDotProductSelfAttention"),
    ],
)
def test_get_attention_classes(attention_type: AttentionType, expected_config_str: str, expected_module_str: str):
    with patch(f"model.configs.transformer.block_configs.{expected_config_str}") as MockConfig, patch(
        f"model.configs.transformer.block_configs.{expected_module_str}"
    ) as MockAttentionModule:
        config_class, module_class = get_attention_classes(attention_type)

        assert config_class == MockConfig
        assert module_class == MockAttentionModule


def test_get_attention_classes_invalid():
    with pytest.raises(ValueError, match="Unsupported attention type"):
        get_attention_classes("invalid_attention_type")


@patch("model.configs.transformer.block_configs.get_attention_classes")
@patch("model.configs.transformer.block_configs.dict_to_dataclass")
def test_transformer_block_config(mock_dict_to_dataclass, mock_get_attention_classes):
    mock_0 = Mock()
    mock_1 = Mock()
    mock_get_attention_classes.return_value = [mock_0, mock_1]

    embedding_dim = 128
    num_heads = 2
    block_size = 64
    attn_drop_p = 0.1
    resid_drop_p = 0.2
    attention_type = AttentionType.STANDARD

    result = TransformerBlockConfig(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        block_size=block_size,
        attn_drop_p=attn_drop_p,
        resid_drop_p=resid_drop_p,
        attention_type=attention_type,
    )

    mock_get_attention_classes.assert_called_once_with(attention_type=attention_type)
    mock_dict_to_dataclass.assert_called_once_with(
        dataclass_type=mock_0,
        dict_obj={
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "block_size": block_size,
            "attn_drop_p": attn_drop_p,
            "resid_drop_p": resid_drop_p,
        },
    )

    assert result.embedding_dim == embedding_dim
    assert result.resid_drop_p == resid_drop_p
    assert result.attention_type == attention_type
    assert result.attention_module == mock_1
    assert result.attention_config == mock_dict_to_dataclass.return_value
