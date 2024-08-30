from loss.loss_wrappers import BaseLossWrapper


def test_base_loss_wrapper():
    total = 0.8

    result = BaseLossWrapper(
        total=total,
    )

    assert result.total == total
