import torch

from pytorch_model_commons.loss.losses import GPTLoss


def test_gpt_loss():
    input_sequence = torch.tensor(
        [
            [0, 1, 2],
            [2, 1, 0],
        ]
    )
    input_logits = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [1.0, 0.0, 1000.0]],
            [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1000.0, 0.0, -1.0]],
        ]
    )
    input_mask = torch.tensor([[0, 1], [1, 1]])
    loss_fn = GPTLoss()

    loss = loss_fn(input_logits, input_sequence, input_mask)

    assert torch.round(loss, decimals=4) == torch.tensor(1.4076)
