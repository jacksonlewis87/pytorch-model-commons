import os
from pytorch_lightning import LightningDataModule, LightningModule
from typing import Type

from pytorch_model_commons.model.model_configs import FullConfig
from pytorch_model_commons.model.model_driver import ModelDriver


def run_training(config: FullConfig, model_class: Type[LightningModule], data_module: LightningDataModule):
    os.makedirs(config.experiment_path, exist_ok=True)

    model = model_class(config=config.model_config)

    ModelDriver(
        full_config=config,
        model=model,
        data_module=data_module,
    ).run_training()
