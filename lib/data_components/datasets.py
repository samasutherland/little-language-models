from typing import Optional, Literal, Annotated, Union
import warnings

from pydantic import ConfigDict, Field

from datasets import load_dataset
from datasets import Dataset as HFDataset

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
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
    ):

        self.tok = _TokWrap(tokenizer)
        self.ds = hf_split
        self.text_column = text_column
        self.max_length = max_length
        if self.text_column not in self.ds.column_names:
            raise ValueError(
                f"Dataset split is missing configured text column '{self.text_column}'. "
                f"Available columns: {self.ds.column_names}"
            )

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
        ids = self.tok.Encode(self.ds[i][self.text_column])
        ids = ids + [self.eos_id]
        if self.max_length is not None:
            if len(ids) >= self.max_length:
                # warnings.warn("Sequence length exceeds max_length. Truncating to max_length (first n tokens retained)")
                ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)


class HFTextFactory(Factory[HFTextDataset]):
    
    type: Literal["hftext"] = "hftext"

    tokenizer_factory: TokenizerFactory

    dataset: str
    text_column: str
    train_split: str
    validation_split: str
    max_length: Optional[int]

    def build(self, ctx: Context) -> HFTextDataset:
        split_names = {"train": self.train_split, "validation": self.validation_split}
        split = ctx.require("split")

        hf_split = _load_hf_split_cached(self.dataset, split_names[split])

        tokenizer = self.tokenizer_factory.build(ctx)

        return HFTextDataset(
            hf_split=hf_split,
            tokenizer=tokenizer,
            text_column=self.text_column,
            max_length=self.max_length,
        )


DatasetFactory = Annotated[
    Union[HFTextFactory], Field(discriminator="type")]
