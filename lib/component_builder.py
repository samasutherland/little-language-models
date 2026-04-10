from pathlib import Path
from typing import Dict, Any, TypeVar

import yaml
import torch.nn as nn

from lib.base_classes import Context, Factory, T

F = TypeVar("F", bound=Factory)

def build_component_from_config(
    factory: F,
    config_path: str | Path,
    ctx: Context,
) -> tuple[T, Dict[str, Any]]:
    """
    Build a components from the component factory and a YAML config file.
    ctx: context required by factory class.
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
    config: Dict[str, Any] = dict(raw)
    builder = factory.model_validate(config)
    return builder.build(ctx), config

def build_component_from_dict(
    factory: F,
    config_dict: Dict[str, Any],
    ctx: Context,
) -> tuple[T, Dict[str, Any]]:
    """
    Build a components from the component factory and a config dict.
    ctx: context required by factory class.
    """

    builder = factory.model_validate(config_dict)
    return builder.build(ctx), config_dict


__all__ = ["build_component_from_config", "build_component_from_dict"]
