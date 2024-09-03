from unittest.mock import Mock, patch

from pytorch_model_commons.model.training import run_training


class MockModel:
    def __init__(self, config):
        super().__init__()
        self.config = config


@patch("pytorch_model_commons.model.training.ModelDriver")
@patch("pytorch_model_commons.model.training.os.makedirs")
def test_run_training(mock_makedirs, mock_model_driver):
    mock_config = Mock()
    mock_data_module = Mock()
    mock_model_driver_instance = Mock()
    mock_model_driver.return_value = mock_model_driver_instance

    run_training(
        config=mock_config,
        model_class=MockModel,
        data_module=mock_data_module,
    )

    mock_makedirs.assert_called_once_with(mock_config.experiment_path, exist_ok=True)
    mock_model_driver_call = mock_model_driver.call_args[1]
    assert mock_model_driver_call["full_config"] == mock_config
    assert isinstance(mock_model_driver_call["model"], MockModel)
    assert mock_model_driver_call["model"].config == mock_config
    assert mock_model_driver_call["data_module"] == mock_data_module
    mock_model_driver_instance.run_training.assert_called_once()
