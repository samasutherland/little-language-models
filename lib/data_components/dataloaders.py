from typing import Literal, Annotated, Union
from functools import partial

from pydantic import ConfigDict, Field

from torch.utils.data import DataLoader, IterableDataset
import torch

from lib import Context, Factory
from lib.data_components.datasets import DatasetFactory

def pad_collate_fn(batch, pad_id):
    x = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_id)
    return x

class TorchDataLoaderFactory(Factory[DataLoader]):
    
    type: Literal["torchdataloader"] = "torchdataloader"

    dataset_factory: DatasetFactory

    prefetch_factor: int

    def build(self, ctx: Context) -> DataLoader:
        dataset = self.dataset_factory.build(ctx)

        collate = partial(pad_collate_fn, pad_id=dataset.pad_id)

        num_workers = ctx.require("dataloader_workers")
        persistent_workers = ctx.require("persistent_workers")
        pin_memory = ctx.require("pin_memory")
        batch_size = ctx.require("batch_size")
        
        shuffle = ctx.require("shuffle")
        if isinstance(dataset, IterableDataset):
            dataset.shuffle = shuffle
            shuffle = False
        return DataLoader(dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          prefetch_factor=self.prefetch_factor,
                          collate_fn=collate,
                          num_workers=num_workers,
                          persistent_workers=persistent_workers,
                          pin_memory=pin_memory,
                          )

DataLoaderFactory = Annotated[
    Union[TorchDataLoaderFactory], Field(discriminator="type")]