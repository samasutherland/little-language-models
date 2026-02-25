from pathlib import Path
from typing import Dict, Any

import yaml
import torch.nn as nn



def build_component_from_config(
    factory,#: Factory
    config_path: str | Path,
    ctx,#: Context
) -> nn.Module:
    """
    Build a components from the component factory and a YAML config file.
    ctx: context required by factory class.
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    model_params: Dict[str, Any] = dict(raw)
    builder = factory.model_validate(model_params)
    return builder.build(ctx)


__all__ = ["build_dataloader_from_config"]
