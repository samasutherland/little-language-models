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

class SimpleStoriesBPEDataset(Dataset):
    def __init__(self, hf_split: HFDataset, tokenizer: SentencePieceProcessor, max_length=None):

        self.tok = _TokWrap(tokenizer)
        self.ds = hf_split
        self.max_length = max_length

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
        ids = self.tok.Encode(self.ds[i]["story"])
        ids = ids + [self.eos_id]
        if self.max_length is not None:
            if len(ids) >= self.max_length:
                # warnings.warn("Sequence length exceeds max_length. Truncating to max_length (first n tokens retained)")
                ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)


class SimpleStoriesBPEFactory(Factory[SimpleStoriesBPEDataset]):
    
    type: Literal["simplestoriesbpe"] = "simplestoriesbpe"

    tokenizer_factory: TokenizerFactory

    dataset: str
    max_length: Optional[int]

    def build(self, ctx: Context) -> SimpleStoriesBPEDataset:
        split = ctx.require("split")

        data = load_dataset(self.dataset)
        hf_split = data[split]

        tokenizer = self.tokenizer_factory.build(ctx)

        return SimpleStoriesBPEDataset(
            hf_split=hf_split,
            tokenizer=tokenizer,
            max_length=self.max_length,
        )


DatasetFactory = Annotated[
    Union[SimpleStoriesBPEFactory], Field(discriminator="type")]
