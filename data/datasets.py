from torch.utils.data import Dataset, DataLoader
import tiktoken
import pickle
from lib.tokenizers.raw_tokenizers import RegexBPE
import torch

# Top-level wrapper so it's picklable
class _TokWrap:
    def __init__(self, enc_, id2bytes_):
        self._enc = enc_
        self.vocab = id2bytes_  # keep id->bytes so max(self.tok.vocab.keys()) works

    def encode(self, s):
        return self._enc.encode(s)

    def decode(self, ids):
        return self._enc.decode(ids)

def _build_tok(vocab_path: str, merge_dict_path: str):
    # build a tiktoken Encoding from your saved pickles
    with open(vocab_path, "rb") as f:
        id2bytes = pickle.load(f)  # dict[int, bytes]
    # merge_dict not strictly needed for tiktoken runtime, but we keep the load to mirror original behavior
    with open(merge_dict_path, "rb") as f:
        _ = pickle.load(f)

    mergeable_ranks = {b: i for i, b in id2bytes.items()}
    pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    enc = tiktoken.Encoding(
        name="rbpe_custom",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
    return _TokWrap(enc, id2bytes)

class SimpleStoriesBPEDataset(Dataset):
    def __init__(self, hf_split, max_length=None, pad_id=None, end_id=None):
        # remember paths so workers can rebuild tokenizer after unpickling
        self._vocab_path = "transformer/vocab.pickle"
        self._merge_dict_path = "transformer/merge_dict.pickle"

        # create tokenizer in the main process
        self.tok = _build_tok(self._vocab_path, self._merge_dict_path)
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

        self.max_id = max(max(self.tok.vocab.keys(), default=-1), self.end_id, self.pad_id)

    # Make dataset picklable for multiprocessing workers (spawn)
    def __getstate__(self):
        state = self.__dict__.copy()
        # don't try to pickle tokenizer objects; rebuild in the worker
        state["tok"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.tok is None:
            self.tok = _build_tok(self._vocab_path, self._merge_dict_path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        # extra safety in case tok wasn't rebuilt yet
        if self.tok is None:
            self.tok = _build_tok(self._vocab_path, self._merge_dict_path)
        ids = self.tok.encode(self.ds[i]["story"])
        if self.end_id is not None:
            ids = ids + [self.end_id]
        if self.max_length is not None:
            ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)
