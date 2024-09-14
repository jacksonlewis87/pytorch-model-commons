import pytest
from unittest.mock import MagicMock, patch
import torch
from dataclasses import dataclass

from pytorch_model_commons.model.base_model import BasePLModule


@dataclass
class MockLossWrapper:
    loss1: float
    loss2: float


@dataclass
class MockModelConfig:
    learning_rate: float


@pytest.fixture
def mock_config():
    return MockModelConfig(learning_rate=0.001)


@pytest.fixture
def mock_loss_wrapper():
    return MockLossWrapper(loss1=0.5, loss2=1.5)


def test_log_losses(mock_config, mock_loss_wrapper):
    module = BasePLModule(config=mock_config)
    module.log = MagicMock()

    module._log_losses(mock_loss_wrapper, stage="train")

    module.log.assert_any_call("train_loss1", 0.5, on_step=True, on_epoch=True, prog_bar=True)
    module.log.assert_any_call("train_loss2", 1.5, on_step=True, on_epoch=True, prog_bar=True)


@patch("pytorch_model_commons.model.base_model.torch.optim.Adam")
def test_configure_optimizers(mock_adam, mock_config):
    module = BasePLModule(config=mock_config)
    module.parameters = MagicMock()

    optimizer = module.configure_optimizers()

    mock_adam.assert_called_once_with(module.parameters(), lr=mock_config.learning_rate)
    assert optimizer == mock_adam.return_value


def test_training_step_not_implemented(mock_config):
    module = BasePLModule(config=mock_config)

    with pytest.raises(NotImplementedError):
        module.training_step(torch.tensor([1.0]), 0)


def test_validation_step_not_implemented(mock_config):
    module = BasePLModule(config=mock_config)

    with pytest.raises(NotImplementedError):
        module.validation_step(torch.tensor([1.0]), 0)
