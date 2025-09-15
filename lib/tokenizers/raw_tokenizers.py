from tqdm import tqdm
from collections import Counter
import regex as re
import operator
from functools import partial
import pickle

class Tokenizer:
    def __init__(self, merge_dict: dict=None, vocab: dict=None):
        self.merge_dict = {} if merge_dict is not None else merge_dict
        self.vocab = {idx: bytes([idx]) for idx in range(256)} if vocab is None else vocab

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_merge_dict(self, filename):
        with open(filename, 'rb') as f:
            self.merge_dict = pickle.load(f)

    def load_vocab(self, filename):
        with open(filename, 'rb') as f:
            self.vocab = pickle.load(f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class BPE(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, train_string, target_dict_size=1000, new_token=None, pbar=True):
        int_list = list(train_string.encode('utf-8'))
        initial_dictionary_size = len(set(int_list))
        initial_tokens = len(int_list)
        if new_token is None:
            new_token = 256
        replacement_dict = {}
        if pbar:
            iter = tqdm(range(target_dict_size - initial_dictionary_size))
        else:
            iter = range(target_dict_size - initial_dictionary_size)
        for i in iter:
            bigram = self.most_common_bigram(int_list)
            int_list = self.replace_bigrams(int_list, bigram, new_token)
            replacement_dict[bigram] = new_token
            new_token += 1

        self.merge_dict = replacement_dict
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (token_1, token_2), new_token in replacement_dict.items():
            self.vocab[new_token] = self.vocab[token_1] + self.vocab[token_2]

        print(f'Dictionary size:\nInitial: {initial_dictionary_size}\nNew: {target_dict_size}')
        print(f'\nInput tokens:\nInitial: {initial_tokens}\nNew: {len(int_list)}')

    def encode(self, string, pbar=False):
        int_list = list(string.encode('utf-8'))
        if pbar:
            iter = tqdm(self.merge_dict.items())
        else:
            iter = self.merge_dict.items()
        for bigram, token in iter:
            int_list = self.replace_bigrams(int_list, bigram, token)
        return int_list

    def decode(self, tokens, pbar=False):
        if pbar:
            iter = tqdm(tokens)
        else:
            iter = tokens
        return (b"".join(self.vocab[token] for token in iter)).decode('utf-8', errors='replace')

    @staticmethod
    def most_common_bigram(int_list):
        bigrams = Counter(zip(int_list, int_list[1:]))
        return bigrams.most_common(1)[0][0]

    @staticmethod
    def replace_bigrams(int_list, bigram, replace):
        new_list =  []
        i = 0
        while True:
            if tuple(int_list[i:i+2]) == bigram:
                new_list.append(replace)
                i += 2
            else:
                new_list.append(int_list[i])
                i += 1
            if i == len(int_list):
                return new_list

    @staticmethod
    def insert_bigrams(int_list, token, bigram):
        new_list = []
        i = 0
        while True:
            if int_list[i] == token:
                new_list += list(bigram)
            else:
                new_list.append(int_list[i])
            i += 1
            if i == len(int_list):
                return new_list

# Pulled from minbpe karpathy
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexBPE(Tokenizer):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)

    def chunk_input(self, input_str):
        return re.findall(self.compiled_pattern, input_str)

    def train(self, train_string, num_merges=1000, new_token=None, pbar=True):
        chunks = self.chunk_input(train_string)
        encoded_chunks = [self.encode_chunk(chunk) for chunk in chunks]

        # Count all bigrams
        counters = [Counter(zip(chunk, chunk[1:])) for chunk in encoded_chunks]
        total_counter = self.counter_sum_keep_negative(*counters)

        if new_token is None:
            new_token = 257

        merge_iter = tqdm(range(num_merges)) if pbar else range(num_merges)
        for _ in merge_iter:
            if not total_counter:
                break  # No more bigrams to merge
            bigram_to_replace = total_counter.most_common(1)[0][0]

            new_encoded_chunks = []
            counter_deltas = []

            for chunk in encoded_chunks:
                new_chunk, delta = self.replace_bigram(chunk, bigram_to_replace, new_token)
                new_encoded_chunks.append(new_chunk)
                counter_deltas.append(delta)

            encoded_chunks = new_encoded_chunks
            delta_counter = self.counter_sum_keep_negative(*counter_deltas)
            total_counter = self.counter_sum_keep_negative(total_counter, delta_counter)

            self.merge_dict[bigram_to_replace] = new_token
            new_token += 1

        # Update vocab
        for (token1, token2), tok in self.merge_dict.items():
            self.vocab[tok] = self.vocab[token1] + self.vocab[token2]

    def encode(self, string, pbar=False):
        int_list = list(string.encode('utf-8'))
        merge_iter = tqdm(self.merge_dict.items()) if pbar else self.merge_dict.items()
        for bigram, token in merge_iter:
            int_list = self.replace_bigrams(int_list, bigram, token)
        return int_list

    def decode(self, tokens, pbar=False):
        iter_tokens = tqdm(tokens) if pbar else tokens
        return b"".join(self.vocab.get(tok, b"") for tok in iter_tokens).decode('utf-8', errors='replace')

    @staticmethod
    def encode_chunk(chunk):
        return list(chunk.encode('utf-8'))

    @staticmethod
    def replace_bigram(chunk, bigram, new_token):
        new_chunk = []
        i = 0
        delta = Counter()

        while i < len(chunk):
            if i < len(chunk) - 1 and (chunk[i], chunk[i+1]) == bigram:
                if i > 0:
                    delta[(chunk[i-1], chunk[i])] -= 1
                    delta[(chunk[i-1], new_token)] += 1
                if i + 2 < len(chunk):
                    delta[(chunk[i+1], chunk[i+2])] -= 1
                    delta[(new_token, chunk[i+2])] += 1
                delta[bigram] -= 1
                new_chunk.append(new_token)
                i += 2
            else:
                new_chunk.append(chunk[i])
                i += 1
        return new_chunk, delta

    @staticmethod
    def replace_bigrams(seq, bigram, replacement):
        result = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == bigram:
                result.append(replacement)
                i += 2
            else:
                result.append(seq[i])
                i += 1
        return result

    @staticmethod
    def counter_sum_keep_negative(*counters):
        result = Counter()
        for c in counters:
            for k, v in c.items():
                result[k] += v
        return result