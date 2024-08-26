import pytest
from pytorch_lightning.callbacks import LearningRateMonitor
from unittest.mock import patch, MagicMock

from model.model_driver import ModelDriver


@pytest.fixture
def mock_full_config():
    with patch("model.model_driver.FullConfig") as MockFullConfig:
        MockFullConfig.return_value = MagicMock()
        MockFullConfig.return_value.experiment_path = "/mock/path"
        MockFullConfig.return_value.model_config.checkpoint_path = "mock_checkpoint.ckpt"
        MockFullConfig.return_value.model_config.epochs = 10
        yield MockFullConfig.return_value


@pytest.fixture
def mock_model():
    with patch("model.model_driver.pl.LightningModule") as MockModel:
        yield MockModel.return_value


@pytest.fixture
def mock_data_module():
    with patch("model.model_driver.pl.LightningDataModule") as MockDataModule:
        MockDataModule.return_value.train_dataloader.return_value = MagicMock()
        MockDataModule.return_value.val_dataloader.return_value = MagicMock()
        yield MockDataModule.return_value


@pytest.fixture
def mock_trainer():
    with patch("model.model_driver.pl.Trainer") as MockTrainer:
        yield MockTrainer.return_value


@pytest.fixture
def mock_tensorboard_logger():
    with patch("model.model_driver.TensorBoardLogger") as MockTensorBoardLogger:
        yield MockTensorBoardLogger.return_value


@pytest.fixture
def model_driver(mock_full_config, mock_model, mock_data_module, mock_trainer, mock_tensorboard_logger):
    with patch("model.model_driver.os.makedirs") as MockMakedirs:
        with patch("model.model_driver.json.dumps") as MockDumps:
            return ModelDriver(full_config=mock_full_config, model=mock_model, data_module=mock_data_module)


@patch("model.model_driver.dataclasses")
@patch("model.model_driver.datetime")
@patch("model.model_driver.json")
@patch("model.model_driver.os")
def test_save_configs(mock_os, mock_json, mock_datetime, mock_dataclasses, model_driver, mock_full_config):
    mock_str_date = "2024-08-26_12-00-00"
    mock_utcnow = MagicMock()
    mock_utcnow.strftime.return_value = mock_str_date
    mock_datetime.utcnow.return_value = mock_utcnow

    with patch("model.model_driver.open", mock_open=True) as mock_open:
        model_driver.save_configs()
        mock_os.makedirs.assert_called_once_with(mock_full_config.experiment_path, exist_ok=True)
        mock_os.path.join.assert_called_once_with(mock_full_config.experiment_path, f"full_config_{mock_str_date}.json")
        mock_open.assert_called_once_with(mock_os.path.join.return_value, "w")
        mock_dataclasses.asdict.assert_called_once_with(mock_full_config)
        mock_json.dumps.assert_called_once_with(mock_dataclasses.asdict.return_value)
        mock_open().__enter__().write.assert_called_once_with(mock_json.dumps.return_value)


@patch.object(ModelDriver, "save_configs")
def test_run_training(mock_save_configs, model_driver, mock_data_module, mock_model, mock_trainer):
    with patch("model.model_driver.ModelDriver.setup_trainer", return_value=mock_trainer):
        model_driver.run_training()
        mock_save_configs.assert_called_once()
        mock_trainer.fit.assert_called_once_with(
            model=mock_model,
            train_dataloaders=mock_data_module.train_dataloader(),
            val_dataloaders=mock_data_module.val_dataloader(),
            ckpt_path=model_driver.full_config.model_config.checkpoint_path,
        )


def test_get_callbacks():
    callbacks = ModelDriver.get_callbacks()
    assert callbacks is not None
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], LearningRateMonitor)


@patch("model.model_driver.pl.Trainer")
@patch.object(ModelDriver, "get_callbacks")
def test_setup_trainer(mock_get_callbacks, mock_pl_trainer, model_driver, mock_tensorboard_logger):
    trainer = model_driver.setup_trainer()
    mock_pl_trainer.assert_called_once_with(
        default_root_dir=model_driver.full_config.experiment_path,
        max_epochs=model_driver.full_config.model_config.epochs,
        accelerator="cpu",
        logger=mock_tensorboard_logger,
        callbacks=mock_get_callbacks.return_value,
    )
    assert trainer == mock_pl_trainer.return_value
