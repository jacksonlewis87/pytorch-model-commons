from loss.loss_config import LossConfig


def test_loss_config():
    total_loss = 0.8

    result = LossConfig(
        total_loss=total_loss,
    )

    assert result.total_loss == total_loss
