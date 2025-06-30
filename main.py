import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
from Dataset.WikiText.Loader import WikiTextDataset
from Tokenize.BPE import TokenizeBPE
from Embedding.Word2Vec.CBOW import CBOW
from LanguageModel.BasicTransformer import DecoderOnlyTransformer

# train_set, val_set, test_set = WikiTextDataset()

bpe = TokenizeBPE([],
                  option_wrap_sentence=True,
                  option_preserve_tokens=True,
                  preserve_tokens=[".", "?", "!", ",", ":", ";"])
# bpe.ProcessMultiProcess(num_processes=4)

# bpe.SaveToFile("token_list_wiki_2.txt")‘


# def ToOneHot(text: list[int]) -> torch.Tensor:
#     one_hot = torch.zeros(len(text), len(bpe.token_list))
#     for i, token in enumerate(text):
#         one_hot[i, token] = 1
#     return one_hot


bpe.LoadFromFile("bpe_wiki_text_tokens.txt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transformer = DecoderOnlyTransformer(bpe, device=device)
# transformer.to(device)
# loss_fn = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.5, patience=3)

# for epoch in range(10):

#     total_loss = 0
#     total_batch = 0

#     print(f"Epoch {epoch}")
#     print(f"Total data: {len(train_set)}")

#     data_index = 0
#     for i in range(1000):

#         sample_index = None
#         indices = None
#         while sample_index is None:
#             index = random.randint(0, len(train_set) - 1)
#             indices = bpe.Tokenize(train_set[index]["text"])
#             if len(indices) > 128:
#                 sample_index = index

#         data_index += 1
#         print(f"Data {data_index} / 1000")

#         if len(indices) < 256:
#             # use padding
#             indices = indices + [1] * (256 - len(indices))

#         seq_len = 128

#         input_list = []
#         target_list = []
#         for index in range(len(indices) - seq_len):
#             input_slice = torch.tensor(indices[index:index + seq_len])
#             input_list.append(input_slice)

#             target_slice = torch.tensor(indices[index + 1:index + seq_len + 1])
#             target_list.append(target_slice)

#         for index in range(len(input_list) - 127):
#             input_batch = torch.stack(input_list[index:index + 128]).to(device)
#             target_batch = torch.stack(
#                 target_list[index:index + 128]).to(device)

#             output = transformer(input_batch)

#             loss = loss_fn(output, target_batch)
#             total_loss += loss.item()
#             total_batch += 1

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     print(total_loss / total_batch)
#     scheduler.step(total_loss / total_batch)

#     print(f"Learning rate: {scheduler.get_last_lr()}")

#     torch.save(transformer.state_dict(), f"transformer.pt")


def loadWeights():
    transformer.load_state_dict(torch.load("transformer.pt"))
    transformer.to(device)


loadWeights()


def stringify(indices: list[int]) -> str:
    return bpe.Stringify(indices)


text = "Hello, how are you?"
text_indices = bpe.Tokenize(text)

for i in range(1000):
    input_batch = None
    if len(text_indices) >= 128:
        input_batch = torch.tensor(text_indices[-128:]).to(device)
    # pad to 128
    else:
        padding_size = 128 - len(text_indices)
        input_batch = F.pad(torch.tensor(text_indices),
                            (0, padding_size), value=1)
        input_batch = input_batch.to(device)
    input_batch = input_batch.unsqueeze(0)

    output: torch.Tensor = transformer(input_batch)
    output = output.transpose(1, 2)
    new_indices = (output.argmax(dim=-1))

    # 找到new_indices[0]中第一个值为1的索引
    pad_positions = torch.where(new_indices[0] == 1)[0]
    if len(pad_positions) > 0:
        pad_index = pad_positions[0].item()
        last_index = pad_index - 1
    else:
        last_index = -1

    text_indices = text_indices + [new_indices[0][last_index].item()]

    print(stringify(text_indices))
