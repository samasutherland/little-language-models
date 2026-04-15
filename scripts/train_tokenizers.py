import sentencepiece as spm
from datasets import load_dataset
import itertools
import tqdm

data = load_dataset("BabyLM-community/BabyLM-2026-Strict", split="train")
n = 10000000
small_test_portion = data.shuffle(seed=42).select(range(n))

def doc_stream():
    for s in small_test_portion["text"]:
        s = s.strip()
        if s:
            yield s

vocab_counts = [1, 2, 3, 4, 6, 8, 10]

for vocab_size in tqdm.tqdm(vocab_counts):
    spm.SentencePieceTrainer.train(
        sentence_iterator=doc_stream(),
        model_prefix=f"baby_unigram_{vocab_size}K",
        vocab_size=vocab_size * 1000,
        normalization_rule_name="nmt_nfkc_cf",
        character_coverage=1.0,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3
    )

vocab_counts = [1, 2, 3, 4, 6, 8, 10]
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

for vocab_size in tqdm.tqdm(vocab_counts):
    spm.SentencePieceTrainer.train(
        sentence_iterator=doc_stream(),
        model_type="bpe",
        model_prefix=f"baby_bpe_{vocab_size}K",
        vocab_size=vocab_size * 1000,
        normalization_rule_name="nmt_nfkc_cf",
        character_coverage=1.0,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3
    )