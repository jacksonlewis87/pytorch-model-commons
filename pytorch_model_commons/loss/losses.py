import torch
import torch.nn as nn


class GPTLoss(nn.Module):
    def __init__(self):
        super(GPTLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss given the logits and the input sequence.

        Parameters:
            logits (torch.Tensor): Logits from the model of shape (B, T, vocab_size).
            input_ids (torch.Tensor): Input token indices of shape (B, T).

        Returns:
            torch.Tensor: The computed loss.
        """
        # Shift the input_ids to create target sequence
        # The target sequence is the input shifted by one position to the right
        targets = input_ids[:, 1:].contiguous()  # Shape: (B, T-1)
        shifted_logits = logits[:, :-1, :].contiguous()  # Shape: (B, T-1, vocab_size)

        # Reshape logits and targets for CrossEntropyLoss
        batch_size, seq_length, vocab_size = shifted_logits.size()
        shifted_logits = shifted_logits.view(-1, vocab_size)  # Flatten to (B * (T-1), vocab_size)
        targets = targets.view(-1)  # Flatten to (B * (T-1))

        # Compute the loss
        loss = self.loss_fn(shifted_logits, targets)
        return loss
