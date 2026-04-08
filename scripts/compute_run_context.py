from lib.utils import init_train_device
from lib.component_builder import build_component_from_config

from lib import Context


from lib.model_components import LanguageModelFactory
from lib.data_components import DataLoaderFactory
from lib.training_components import LoopFactory

import torch
from contextlib import nullcontext
from pathlib import Path
import yaml


device, autocast_context = init_train_device()

context_path = Path("../configs/context.yaml")
with context_path.open("r") as f:
    raw = yaml.safe_load(f)

run_context = Context(**context_dict)

