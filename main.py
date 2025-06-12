import os
import sys
from Dataset.WikiText.Loader import WikiTextDataset
from Tokenize.BPE import TokenizeBPE

train_set, val_set, test_set = WikiTextDataset()

bpe = TokenizeBPE([train_set, val_set, test_set])
bpe.Process()

bpe.SaveToFile("token_list.txt")
