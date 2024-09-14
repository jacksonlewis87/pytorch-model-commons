import pytest
from unittest.mock import Mock, create_autospec, patch

from model.model_configs import FullConfig, ModelConfig


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
        experiment_path="/path/to/experiment",
        data_split_path="/path/to/data-split",
        data_config=mock_data_config,
        model_config=mock_model_config,
    )


def test_full_config_initialization(full_config, mock_data_config, mock_model_config):
    assert full_config.experiment_path == "/path/to/experiment"
    assert full_config.data_split_path == "/path/to/data-split"
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
