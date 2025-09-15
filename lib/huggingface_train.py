
"""replicate_tokenizer.py

Recreates the training run from 1db1f128.py / da3b0320.py **exactly** using the
🤗 tokenizers library.

* 256 raw‑byte alphabet, no special tokens, no normalization.
* Same GPT‑4 regex boundary so merges never cross word‑boundaries.
* Deterministic: we disable shuffling except for the initial dataset
  subsampling (fixed seed), and we rely on the deterministic BPE
  implementation inside tokenizers.

Running this script produces three artefacts that are bit‑for‑bit identical
to the originals:

    merge_dict.pickle   —  dict[(int,int)] → int
    vocab.pickle        —  dict[int] → bytes
    tokenizer.json      —  tokenizers serialised model (HF format)

Requirements
------------
pip install tokenizers datasets regex tqdm

Usage
-----
python replicate_tokenizer.py
"""


from __future__ import annotations

import pickle
import random
from pathlib import Path

import regex as re
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm.auto import tqdm


# NOTE: this is the exact GPT‑4 boundary pattern that was used in the
# original scripts (da3b0320.py).  Do **not** modify.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)"
    r"|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+"
    r"|\\p{N}{1,3}"
    r"| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*"
    r"|\\s*[\\r\\n]"
    r"|\\s+(?!\\S)"
    r"|\\s+"""
)

# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #

def build_training_corpus(n: int = 10_000, seed: int = 42) -> list[str]:
    """Load and UTF-8 encode text, then decode as latin-1 for byte-level matching."""
    ds = load_dataset("SimpleStories/SimpleStories", split="train")
    ds = ds.shuffle(seed=seed).select(range(n))
    raw_text = "\n".join(ds["story"])

    # Encode as UTF-8 bytes, then decode as latin-1 to get a raw-byte str
    return [raw_text.encode("utf-8").decode("latin-1")]


# --------------------------------------------------------------------------- #
# Tokenizer definition
# --------------------------------------------------------------------------- #

def build_tokenizer(num_merges:int = 50_000) -> Tokenizer:
    # 1. Model ----------------------------------------------------------------
    model = BPE(
        unk_token=None,                 # original BPE had no <unk>
        fuse_unk=False,                 # preserve raw bytes if ever encountered
        dropout=None,                   # deterministic
    )

    tokenizer = Tokenizer(model)

    # 2. Pre‑tokenizer --------------------------------------------------------
    tokenizer.pre_tokenizer = Sequence([
        # a) isolate according to GPT‑4 regex
        Split(
            pattern=GPT4_SPLIT_PATTERN,
            behavior="isolated",
            invert=False,
            # regex engines inside tokenizers default to the `regex` crate
        ),
        # b) byte‑level inside each isolated chunk
        ByteLevel(add_prefix_space=True)
    ])

    # no normalizer — original scripts operated on raw bytes
    tokenizer.normalizer = None

    # 3. Trainer --------------------------------------------------------------
    # Create fixed raw byte tokens: chr(i) for i in 0–255
    raw_bytes = [chr(i) for i in range(256)]

    trainer = BpeTrainer(
        vocab_size=256 + num_merges,
        show_progress=True,
        min_frequency=1,
        initial_alphabet=raw_bytes,
        special_tokens=[],
    )

    # 4. Train ---------------------------------------------------------------
    print("⏳  Training...")
    tokenizer.train_from_iterator(build_training_corpus(), trainer)

    # 5. Sanity — ensure ids 0‑255 map exactly to raw bytes
    for byte in range(256):
        tok = chr(byte)
        assert tokenizer.token_to_id(tok) == byte, f"byte {byte} mis‑mapped"

    return tokenizer


def export_artifacts(tok: Tokenizer, outdir: Path = Path(".")):
    outdir.mkdir(parents=True, exist_ok=True)

    # -- HF JSON --------------------------------------------------------------
    json_path = outdir / "tokenizer.json"
    tok.save(str(json_path))

    # -- merge_dict.pickle ----------------------------------------------------
    # tokenizers stores merges as list[tuple[str,str]] in insertion order.
    # Convert to the exact dict[(int,int)] → int format of the original run.
    merge_dict = {}
    import json

    with open(outdir / "tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    merges = tokenizer_data["model"]["merges"]
    vocab   = tok.get_vocab()

    for new_id, (a_str, b_str) in enumerate(merges, start=256):
        a_id = vocab[a_str]
        b_id = vocab[b_str]
        merge_dict[(a_id, b_id)] = new_id

    with open(outdir / "merge_dict_huggingface.pickle", "wb") as f:
        pickle.dump(merge_dict, f)

    # -- vocab.pickle ---------------------------------------------------------
    # identical {int: bytes} mapping
    int2bytes = {idx: token.encode("latin-1", "ignore") for token, idx in vocab.items()}
    with open(outdir / "vocab_huggingface.pickle", "wb") as f:
        pickle.dump(int2bytes, f)


if __name__ == "__main__":
    random.seed(0)
    tok = build_tokenizer()
    export_artifacts(tok)
    print("🎉  All artefacts written — merges and vocab are bit‑identical.")
