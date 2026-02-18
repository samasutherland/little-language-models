from .components import *

import tomllib

def load_configs():
    with open("configs/model.toml", "rb") as f:
        model_cfg = tomllib.load(f)
    with open("configs/data.toml", "rb") as f:
        data_cfg = tomllib.load(f)
    with open("configs/training.toml", "rb") as f:
        train_cfg = tomllib.load(f)

    return model_cfg, data_cfg, train_cfg

