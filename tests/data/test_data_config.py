from data.data_config import DataConfig


def test_data_config():
    input_path = "/path/to/checkpoint"
    batch_size = 2
    train_size = 0.8

    result = DataConfig(
        input_path=input_path,
        batch_size=batch_size,
        train_size=train_size,
    )

    assert result.input_path == input_path
    assert result.batch_size == batch_size
    assert result.train_size == train_size
