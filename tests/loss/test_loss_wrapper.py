from torch import tensor

from loss.loss_wrappers import BaseLossWrapper


def test_base_loss_wrapper():
    total = tensor(0.8)

    result = BaseLossWrapper(
        total=total,
    )

    assert result.total == total
