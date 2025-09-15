
from lib.tokenizers.raw_tokenizers import RegexDaskBPE
import pickle


from datasets import load_dataset
from dask.distributed import Client

if __name__ == "__main__":
    client = Client(n_workers=32, threads_per_worker=1)
    print(client)

    tiny_stories = load_dataset("SimpleStories/SimpleStories")

    n = 10000
    small_test_portion = tiny_stories['train'].shuffle(seed=42).select(range(n))

    tokenizer_string = ' '.join(small_test_portion['story'])
    tokenizer_int_list = list(tokenizer_string.encode('utf-8'))
    print(f"Training on {len(tokenizer_int_list)} tokens")

    rbpe = RegexDaskBPE()
    try:
        rbpe.train(tokenizer_string, npartitions=64, num_merges=50000)
    except:
        pass


    with open('merge_dict.pickle', 'wb') as file:
        pickle.dump(rbpe.merge_dict, file)

    with open('vocab.pickle', 'wb') as file:
        pickle.dump(rbpe.vocab, file)