import sentencepiece as spm
from datasets import load_dataset
from pathlib import Path
import regex  # Unicode-aware regex module
import pickle

# ---------------------------------------------------------------------------- #
# Configuration
# ---------------------------------------------------------------------------- #
MODEL_PREFIX   = "rbpe_sp"
MERGES         = 50_000
RAW_BYTES      = 256
VOCAB_SIZE     = RAW_BYTES + MERGES
TMP_TXT        = Path("rbpe_training.txt")
SEED           = 42
N_THREADS      = 64

# ---------------------------------------------------------------------------- #
# Regex-based tokenizer used in original RegexDaskBPE
# ---------------------------------------------------------------------------- #
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
split_pattern = regex.compile(GPT4_SPLIT_PATTERN)

def gpt4_chunk(text):
    return " ".join(split_pattern.findall(text))

# ---------------------------------------------------------------------------- #
# Load dataset and preprocess using GPT4-style regex chunking
# ---------------------------------------------------------------------------- #
print("Loading and preprocessing dataset...")
ds = load_dataset("SimpleStories/SimpleStories", split="train")
train_slice = ds.shuffle(seed=SEED).select(range(10_000))

with TMP_TXT.open("w", encoding="utf-8") as f:
    for story in train_slice["story"]:
        f.write(gpt4_chunk(story) + " ")

# ---------------------------------------------------------------------------- #
# Train SentencePiece with byte-level BPE and regex-based pretokenization
# ---------------------------------------------------------------------------- #
print("Training SentencePiece tokenizer...")
spm.SentencePieceTrainer.Train(
    input=str(TMP_TXT),
    model_prefix=MODEL_PREFIX,
    model_type="bpe",
    vocab_size=VOCAB_SIZE,
    byte_fallback=True,
    character_coverage=1.0,
    split_by_whitespace=True,
    split_by_unicode_script=False,
    split_digits=False,
    add_dummy_prefix=False,
    remove_extra_whitespaces=False,
    shuffle_input_sentence=False,
    input_sentence_size=0,
    normalization_rule_name="identity",
    hard_vocab_limit=False,
    num_threads=N_THREADS,
    bos_id=-1, eos_id=-1, pad_id=-1,
)

# ---------------------------------------------------------------------------- #
# Extract merge dictionary using SentencePiece public API
# ---------------------------------------------------------------------------- #
print("Extracting merge dictionary...")
sp = spm.SentencePieceProcessor()
sp.load(f"{MODEL_PREFIX}.model")

merge_dict = {}
for i in range(sp.get_piece_size()):
    piece = sp.id_to_piece(i)
    if i >= RAW_BYTES + 1:  # skip <unk> and byte pieces
        b = piece.encode("utf-8")
        lhs, rhs = b[:-1], b[-1:]
        merge_dict[(lhs, rhs)] = i

with open("merge_dict_sentencepiece.pickle", "wb") as f:
    pickle.dump(merge_dict, f)

print("Done! Model and merge_dict.pickle saved.")
