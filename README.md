# Pytorch Model Commons
Common code for model development, training, and evaluation in pytorch

## Utilization
* Create a GitHub PAT
* Configure the PAT in poetry: `poetry config http-basic.github __token__ <PAT>`
* Add the dependency to `pyproject.toml`: `pytorch_model_commons = { git = "https://github.com/jacksonlewis87/pytorch-model-commons.git", branch = "main" }`
