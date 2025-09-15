from torch.utils.data import Dataset, DataLoader

from lib.tokenizers.raw_tokenizers import RegexBPE
from lib.model_layers.transformer import Transformer
import torch

class SimpleStoriesBPEDataset(Dataset):
    def __init__(self, hf_split, max_length=None, pad_id=None, end_id=None):
        self.tok = RegexBPE()
        self.tok.load_merge_dict("transformer/merge_dict.pickle")
        self.tok.load_vocab("transformer/vocab.pickle")
        self.ds = hf_split
        self.max_length = max_length
        if pad_id is None:
            try:
                self.pad_id = max(self.tok.vocab.keys()) + 1
            except:
                self.pad_id = 0
        else:
            self.pad_id = pad_id
        if end_id is None:
            try:
                self.end_id = max(self.tok.vocab.keys()) + 2
            except:
                self.end_id = 0
        else:
            self.end_id = end_id
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        ids = self.tok.encode(self.ds[i]["story"])
        if self.end_id is not None:
            ids = ids + [self.end_id]
        if self.max_length is not None:
            ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)