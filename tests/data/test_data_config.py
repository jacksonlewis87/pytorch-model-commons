from pytorch_model_commons.data.data_config import DataConfig


def test_data_config():
    input_path = "/path/to/checkpoint"
    batch_size = 2
    train_size = 0.8
    data_split_path = "/path/to/data-split"

    result = DataConfig(
        input_path=input_path,
        batch_size=batch_size,
        train_size=train_size,
        data_split_path=data_split_path,
    )

    assert result.input_path == input_path
    assert result.batch_size == batch_size
    assert result.train_size == train_size
    assert result.data_split_path == data_split_path
