from loss.loss_wrappers import BaseLossWrapper


def test_base_loss_wrapper():
    total_loss = 0.8

    result = BaseLossWrapper(
        total_loss=total_loss,
    )

    assert result.total_loss == total_loss
