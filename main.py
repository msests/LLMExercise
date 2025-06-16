import os
import sys
from Dataset.WikiText.Loader import WikiTextDataset
from Tokenize.BPE import TokenizeBPE
from Embedding.Word2Vec.CBOW import CBOW

train_set, val_set, test_set = WikiTextDataset()

bpe = TokenizeBPE([train_set],
                  option_wrap_sentence=True,
                  option_preserve_tokens=True,
                  preserve_tokens=[".", "?", "!", ",", ":", ";"])
# bpe.Process()

# bpe.SaveToFile("token_list_wiki_2.txt")

bpe.LoadFromFile("token_list_wiki.txt")

for data in train_set:
    print(bpe.Stringify(bpe.Tokenize(data["text"])))

# cbow = CBOW(train_set, bpe, 100, 3)
# cbow.Train()
