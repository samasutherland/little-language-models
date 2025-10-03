from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

import pickle
import torch

# Top-level wrapper so it's picklable
class _TokWrap:
    def __init__(self, model_path: str):
        self._enc = spm.SentencePieceProcessor(model_file=model_path)

        self.pad_id = self._enc.pad_id()
        self.eos_id = self._enc.eos_id()
        self.unk_id = self._enc.unk_id()
        self.bos_id = self._enc.bos_id()
        self.vocab_size = self._enc.vocab_size()

    def encode(self, s):
        return self._enc.encode(s, out_type=int)

    def decode(self, ids):
        return self._enc.decode(ids)

class SimpleStoriesBPEDataset(Dataset):
    def __init__(self, hf_split, model_path="data/tokenizers/unigram_4K.model", max_length=None):
        self.model_path = model_path

        # create tokenizer in the main process
        self.tok = _TokWrap(model_path)
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

    # Make dataset picklable for multiprocessing workers (spawn)
    def __getstate__(self):
        state = self.__dict__.copy()
        # don't try to pickle tokenizer objects; rebuild in the worker
        state["tok"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.tok is None:
            self.tok = _TokWrap(self.model_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        # extra safety in case tok wasn't rebuilt yet
        if self.tok is None:
            self.tok = _TokWrap(self.model_path)
        ids = self.tok.encode(self.ds[i]["story"])
        ids = ids + [self.eos_id]
        if self.max_length is not None:
            ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)
