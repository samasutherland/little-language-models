from typing import Literal, Annotated, Union
from functools import partial

from pydantic import BaseModel, ConfigDict, Field

from torch.utils.data import DataLoader
import torch

from lib.data_components.context import DataContext
from lib.data_components.datasets import DatasetFactory

def pad_collate_fn(batch, pad_id):
    x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
    m = (x != pad_id).long()
    return {"input_ids": x, "attention_mask": m}

class TorchDataLoaderFactory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["torchdataloader"] = "torchdataloader"

    dataset_factory: DatasetFactory

    batch_size: int
    shuffle: bool
    prefetch_factor: int

    def build(self, ctx: DataContext) -> DataLoader:
        dataset = self.dataset_factory.build(ctx)

        collate = partial(pad_collate_fn, pad_id=dataset.pad_id)

        num_workers = ctx.require("dataloader_workers")
        persistent_workers = ctx.require("persistent_workers")
        pin_memory = ctx.require("pin_memory")

        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          prefetch_factor=self.prefetch_factor,
                          collate_fn=collate,
                          num_workers=num_workers,
                          persistent_workers=persistent_workers,
                          pin_memory=pin_memory,
                          )

DataLoaderFactory = Annotated[
    Union[TorchDataLoaderFactory], Field(discriminator="type")]