from typing import Optional, Literal, Annotated, Union
import warnings

from pydantic import BaseModel, ConfigDict, Field

from datasets import load_dataset

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
import torch

from lib.data_components.tokenizers import TokenizerFactory
from lib.data_components.context import DataContext



collate = partial(pad_collate_fn, pad_id=dataset.pad_id)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    collate_fn=collate,
    num_workers=nw,
    persistent_workers=opts.persistent_workers if nw > 1 else False,
    pin_memory=opts.pin_memory,
    prefetch_factor=opts.prefetch_factor if nw > 1 else None,
)