from pathlib import Path
from typing import Dict, Any

import yaml
import torch.nn as nn

from lib.components import LanguageModelFactory
from lib.components.base import BuildContext


def build_model_from_config(config_path: str | Path = "configs/model.yaml") -> nn.Module:
    """
    Load a language model from a YAML config file and construct the nn.Module.
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    model_params: Dict[str, Any] = dict(raw["model_parameters"])
    
    # Extract embedding_dim and create BuildContext
    embedding_dim = model_params.pop("embedding_dim")
    ctx = BuildContext(embedding_dim=embedding_dim)

    factory = LanguageModelFactory.model_validate(model_params)
    return factory.build(ctx)


__all__ = ["build_model_from_config"]

