# Pytorch Model Commons
Common code for model development, training, and evaluation in pytorch

## Utilization
* Create a GitHub PAT
* Configure the PAT in poetry: `poetry config http-basic.github __token__ <PAT>`
* Add the dependency to `pyproject.toml`: `pytorch_model_commons = { git = "https://github.com/jacksonlewis87/pytorch-model-commons.git", branch = "main" }`

## Creating new version
* Commit changes locally (don't modify poetry version, this will happen automatically)
* run `bash create_git_release` (documentation in [dev-utilities](https://github.com/jacksonlewis87/dev-utilities))
