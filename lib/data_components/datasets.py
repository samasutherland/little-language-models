from typing import Optional, Literal, Annotated, Union
import random

from pydantic import ConfigDict, Field

from datasets import load_dataset
from datasets import Dataset as HFDataset

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torch

from lib.data_components.tokenizers import TokenizerFactory
from lib import Context, Factory

_HF_SPLIT_CACHE: dict[tuple[str, str], HFDataset] = {}


def _load_hf_split_cached(dataset_name: str, split_name: str) -> HFDataset:
    cache_key = (dataset_name, split_name)
    cached_split = _HF_SPLIT_CACHE.get(cache_key)
    if cached_split is None:
        cached_split = load_dataset(dataset_name, split=split_name)
        _HF_SPLIT_CACHE[cache_key] = cached_split
    return cached_split


def _interleave_holdout_split(hf_split: HFDataset, split: str, every_n: int) -> HFDataset:
    if every_n <= 1:
        raise ValueError("every_n must be greater than 1.")
    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'validation'.")

    keep_validation = split == "validation"
    return hf_split.filter(
        lambda _, idx: (idx % every_n == 0) if keep_validation else (idx % every_n != 0),
        with_indices=True,
    )


class _TokWrap:
    # Wrapper for a SentencePieceProcessor. This is necessary for multi-gpu as the base class cannot be pickled.
    def __init__(self, model: SentencePieceProcessor):
        self._enc = model

        self.pad_id = self._enc.pad_id()
        self.eos_id = self._enc.eos_id()
        self.unk_id = self._enc.unk_id()
        self.bos_id = self._enc.bos_id()
        self.vocab_size = self._enc.vocab_size()

    def Encode(self, s):
        return self._enc.Encode(s, out_type=int)

    def Decode(self, ids):
        return self._enc.Decode(ids)

class HFTextDataset(Dataset):
    def __init__(
        self,
        hf_split: HFDataset,
        tokenizer: SentencePieceProcessor,
        text_column: str,
        max_length=None,
        pack_to_max_length: bool = False,
    ):

        self.tok = _TokWrap(tokenizer)
        self.ds = hf_split
        self.text_column = text_column
        self.max_length = max_length
        self.pack_to_max_length = pack_to_max_length
        if self.text_column not in self.ds.column_names:
            raise ValueError(
                f"Dataset split is missing configured text column '{self.text_column}'. "
                f"Available columns: {self.ds.column_names}"
            )
        if self.pack_to_max_length:
            if self.max_length is None:
                raise ValueError("pack_to_max_length=True requires max_length to be set.")

    @property
    def pad_id(self):
        return self.tok.pad_id

    @property
    def eos_id(self):
        return self.tok.eos_id

    @property
    def bos_id(self):
        return self.tok.bos_id

    @property
    def unk_id(self):
        return self.tok.unk_id

    @property
    def vocab_size(self):
        return self.tok.vocab_size

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state["tok"] = None
    #     return state
    # 
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ids = []
        row_idx = i
        if not self.pack_to_max_length or self.max_length is None:
            ids = self.tok.Encode(self.ds[row_idx][self.text_column]) + [self.eos_id]
        else:
            while row_idx < len(self.ds) and len(ids) < self.max_length:
                row_ids = self.tok.Encode(self.ds[row_idx][self.text_column])
                ids.extend(row_ids + [self.eos_id])
                row_idx += 1
        if self.max_length is not None:
            if len(ids) >= self.max_length:
                # warnings.warn("Sequence length exceeds max_length. Truncating to max_length (first n tokens retained)")
                ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)


class HFTextIterableDataset(IterableDataset):
    def __init__(
        self,
        hf_split: HFDataset,
        tokenizer: SentencePieceProcessor,
        text_column: str,
        max_length: int,
        shuffle: bool,
        shuffle_buffer_size: int,
        shuffle_seed: Optional[int],
        drop_last: bool,
    ):
        self.tok = _TokWrap(tokenizer)
        self.ds = hf_split
        self.text_column = text_column
        self.max_length = max_length
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        self.drop_last = drop_last
        if self.text_column not in self.ds.column_names:
            raise ValueError(
                f"Dataset split is missing configured text column '{self.text_column}'. "
                f"Available columns: {self.ds.column_names}"
            )
        if self.max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        if self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be non-negative.")
        if self.shuffle and self.shuffle_buffer_size > 0 and self.shuffle_seed is None:
            raise ValueError("shuffle_seed must be set when shuffle is enabled with a non-zero shuffle_buffer_size.")

    @property
    def pad_id(self):
        return self.tok.pad_id

    @property
    def eos_id(self):
        return self.tok.eos_id

    @property
    def bos_id(self):
        return self.tok.bos_id

    @property
    def unk_id(self):
        return self.tok.unk_id

    @property
    def vocab_size(self):
        return self.tok.vocab_size

    def _iter_fixed_blocks(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        token_buffer = []
        for row_idx in range(worker_id, len(self.ds), num_workers):
            row_ids = self.tok.Encode(self.ds[row_idx][self.text_column])
            token_buffer.extend(row_ids + [self.eos_id])
            while len(token_buffer) >= self.max_length:
                yield token_buffer[:self.max_length]
                token_buffer = token_buffer[self.max_length:]

        if not self.drop_last and token_buffer:
            yield token_buffer

    def __iter__(self):
        if not self.shuffle or self.shuffle_buffer_size == 0:
            for block in self._iter_fixed_blocks():
                yield torch.tensor(block, dtype=torch.long)
            return

        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        seed = self.shuffle_seed + worker_id
        rng = random.Random(seed)
        block_buffer = []
        for block in self._iter_fixed_blocks():
            if len(block_buffer) < self.shuffle_buffer_size:
                block_buffer.append(block)
                continue
            replace_idx = rng.randrange(len(block_buffer))
            yield torch.tensor(block_buffer[replace_idx], dtype=torch.long)
            block_buffer[replace_idx] = block

        rng.shuffle(block_buffer)
        for block in block_buffer:
            yield torch.tensor(block, dtype=torch.long)


class HFTextFactory(Factory[HFTextDataset]):
    
    type: Literal["hftext"] = "hftext"

    tokenizer_factory: TokenizerFactory

    dataset: str
    text_column: str
    train_split: str
    validation_split: str
    interleave_holdout_every_n: int = 100
    max_length: Optional[int]
    pack_to_max_length: bool

    def build(self, ctx: Context) -> HFTextDataset:
        split_names = {"train": self.train_split, "validation": self.validation_split}
        split = ctx.require("split")

        if self.train_split == self.validation_split:
            hf_base_split = _load_hf_split_cached(self.dataset, self.train_split)
            hf_split = _interleave_holdout_split(hf_base_split, split, self.interleave_holdout_every_n)
        else:
            hf_split = _load_hf_split_cached(self.dataset, split_names[split])

        tokenizer = self.tokenizer_factory.build(ctx)

        return HFTextDataset(
            hf_split=hf_split,
            tokenizer=tokenizer,
            text_column=self.text_column,
            max_length=self.max_length,
            pack_to_max_length=self.pack_to_max_length,
        )


class HFTextIterableFactory(Factory[HFTextIterableDataset]):
    type: Literal["hftext_iterable"] = "hftext_iterable"

    tokenizer_factory: TokenizerFactory

    dataset: str
    text_column: str
    train_split: str
    validation_split: str
    interleave_holdout_every_n: int = 100
    max_length: int
    shuffle: bool = False
    shuffle_buffer_size: int
    shuffle_seed: Optional[int] = None
    drop_last: bool

    def build(self, ctx: Context) -> HFTextIterableDataset:
        split_names = {"train": self.train_split, "validation": self.validation_split}
        split = ctx.require("split")

        if self.train_split == self.validation_split:
            hf_base_split = _load_hf_split_cached(self.dataset, self.train_split)
            hf_split = _interleave_holdout_split(hf_base_split, split, self.interleave_holdout_every_n)
        else:
            hf_split = _load_hf_split_cached(self.dataset, split_names[split])
        tokenizer = self.tokenizer_factory.build(ctx)

        return HFTextIterableDataset(
            hf_split=hf_split,
            tokenizer=tokenizer,
            text_column=self.text_column,
            max_length=self.max_length,
            shuffle=self.shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle_seed=self.shuffle_seed,
            drop_last=self.drop_last,
        )


DatasetFactory = Annotated[
    Union[HFTextFactory, HFTextIterableFactory], Field(discriminator="type")]
