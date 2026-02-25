from pathlib import Path
from typing import Dict, Any

import yaml
import torch.nn as nn

from lib.data_components import DataContext, TorchDataLoaderFactory



def build_dataloader_from_config(
    config_path: str | Path,
    ctx: DataContext,
) -> nn.Module:
    """
    Load a language model from a YAML config file and construct the nn.Module.
    ctx: build context (supplied by caller; see e.g. RunConfig.get_build_context).
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    model_params: Dict[str, Any] = dict(raw)
    factory = TorchDataLoaderFactory.model_validate(model_params)
    return factory.build(ctx)


__all__ = ["build_dataloader_from_config"]
