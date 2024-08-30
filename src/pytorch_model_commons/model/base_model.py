import torch
import pytorch_lightning as pl
from dataclasses import fields

from pytorch_model_commons.loss.loss_wrappers import BaseLossWrapper
from pytorch_model_commons.model.model_configs import ModelConfig


class BasePLModule(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def _log_losses(self, loss_wrapper: BaseLossWrapper, stage: str):
        """
        Logs the losses defined in the LossConfig dataclass.

        Args:
            loss_wrapper (BaseLossWrapper): An instance of BaseLossWrapper dataclass.
            stage (str): The stage of training, e.g., "train" or "val".
        """
        for loss in fields(loss_wrapper):
            self.log(
                f"{stage}_{loss.name}",
                getattr(loss_wrapper, loss.name),
                on_step=stage == "train",
                on_epoch=True,
                prog_bar=True,
            )

    def configure_optimizers(self):
        """
        Configures the optimizer.

        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Define the training step logic here. This method should be overridden in subclasses.

        Args:
            batch (torch.Tensor): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss tensor.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """
        Define the validation step logic here. This method should be overridden in subclasses.

        Args:
            batch (torch.Tensor): The input batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            dict: Dictionary containing validation metrics.
        """
        raise NotImplementedError("Subclasses should implement this method.")
