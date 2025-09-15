from tqdm import tqdm
from collections import Counter
import regex as re
import operator
from dask import delayed, compute, bag
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

# Pulled from minbpe karpathy
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexDaskBPE(Tokenizer):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)

    def chunk_input(self, input):
        return re.findall(self.compiled_pattern, input)

    def train(self, train_string, num_merges=1000, new_token=None, pbar=True, npartitions=32):

        chunks = self.chunk_input(train_string)

        chunk_bag = bag.from_sequence(chunks, npartitions=npartitions)

        encoded_chunks = chunk_bag.map(self.encode_chunk).persist()


        counters = encoded_chunks.map(lambda x: Counter(zip(x, x[1:])))
        total_counter = counters.fold(binop=operator.add).compute()

        if new_token is None:
            new_token = 257

        for i in tqdm(range(num_merges)):
            bigram_to_replace = total_counter.most_common(1)[0][0]

            replacement_func = partial(self.replace_bigram_with_counter, bigram=bigram_to_replace, new_token=new_token)
            new_chunks_and_counters = encoded_chunks.map(replacement_func).persist()

            old_encoded_chunks = encoded_chunks
            encoded_chunks = new_chunks_and_counters.map(lambda x: x[0])#.filter(lambda x: len(x)>1)
            counter_update = new_chunks_and_counters.map(lambda x: x[1]).fold(binop=self.counter_sum_keep_negative).compute()
            del old_encoded_chunks

            self.merge_dict[bigram_to_replace] = new_token

            total_counter += counter_update
            self.vocab[new_token] = self.vocab[bigram_to_replace[0]] + self.vocab[bigram_to_replace[1]]
            new_token += 1

    def encode(self, string, pbar=False):
        chunks = self.chunk_input(string)
        chunk_bag = bag.from_sequence(chunks, npartitions=32)
        encoded_chunks = chunk_bag.map(self.encode_chunk).persist()

        for bigram, token in tqdm(self.merge_dict.items(), disable=not pbar):
            replacement_func = partial(self.replace_bigram, bigram=bigram, new_token=token)
            encoded_chunks = encoded_chunks.map(replacement_func).persist()

        return encoded_chunks.fold(binop=operator.add).compute()

    def decode(self, tokens, pbar=True):
        if pbar:
            iter = tqdm(tokens)
        else:
            iter = tokens
        return (b"".join(self.vocab[token] for token in iter)).decode('utf-8', errors='replace')

    # Just named functions so can see what's happening in dask.
    @staticmethod
    def extract_chunks(output):
        return output[0]

    @staticmethod
    def extract_counters(output):
        return output[1]

    @staticmethod
    def filter_chunks(output):
        return len(output) > 1

    @staticmethod
    def encode_chunk(chunk):
        return list(chunk.encode('utf-8'))

    @staticmethod
    def replace_bigram_with_counter(chunk, bigram, new_token):
        new_chunk = []
        i = 0
        counter_update = Counter()

        while i < len(chunk):
            if i < len(chunk) - 1 and (chunk[i], chunk[i+1]) == bigram:
                if i > 0:
                    counter_update[(chunk[i-1], chunk[i])] -= 1
                    counter_update[(chunk[i-1], new_token)] += 1
                if i + 2 < len(chunk):
                    counter_update[(chunk[i+1], chunk[i+2])] -= 1
                    counter_update[(new_token, chunk[i+2])] += 1

                counter_update[bigram] -= 1
                new_chunk.append(new_token)
                i+=2
            else:
                new_chunk.append(chunk[i])
                i+=1
        return new_chunk, counter_update

    @staticmethod
    def replace_bigram(chunk, bigram, new_token):
        new_chunk = []
        i = 0

        while i < len(chunk):
            if i < len(chunk) - 1 and (chunk[i], chunk[i+1]) == bigram:
                new_chunk.append(new_token)
                i+=2
            else:
                new_chunk.append(chunk[i])
                i+=1
        return new_chunk

    @staticmethod
    def counter_sum_keep_negative(a, b):
        a = a.copy()
        for k, v in b.items():
            a[k] += v
        return a
