import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import sys
import re

from matplotlib import pyplot as plt
from Tokenize.BPE import TokenizeBPE
from Dataset.WikiText.Loader import WikiTextDataset


class CBOWNet(nn.Module):
    def __init__(self,
                 tokenize_bpe: TokenizeBPE,
                 embedding_size: int = 100,
                 window_size: int = 7,
                 device: str = "cpu"):
        super().__init__()
        self.tokenize_bpe = tokenize_bpe
        self.window_size = window_size
        self.embedding_size = embedding_size

        self.context_word_vector = nn.Parameter(
            torch.normal(0, 0.01, (embedding_size, len(tokenize_bpe.token_list)), device=device))
        self.target_word_vector = nn.Parameter(
            torch.normal(0, 0.01, (embedding_size, len(tokenize_bpe.token_list)), device=device))

    def forward(self, x):
        length = x.shape[1]
        x = torch.sum(x, dim=1, keepdim=True)  # [batch_size, 1, vocab_size]
        x = x / length
        # [batch_size, 1, embedding_size]
        hidden = torch.matmul(x, self.context_word_vector.T)
        # [batch_size, 1, vocab_size]
        output = torch.matmul(hidden, self.target_word_vector)
        return output.squeeze(1)  # [batch_size, vocab_size]


class CBOW():
    def __init__(self, data_set: Dataset,
                 tokenize_bpe: TokenizeBPE,
                 embedding_size: int = 100,
                 window_size: int = 4):
        self.tokenize_bpe = tokenize_bpe
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.data_set = data_set

    def ToOneHot(self, text: list[int]) -> torch.Tensor:
        one_hot = torch.zeros(len(text), len(self.tokenize_bpe.token_list))
        for i, token in enumerate(text):
            one_hot[i, token] = 1
        return one_hot

    def Train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = CBOWNet(self.tokenize_bpe, self.embedding_size,
                      self.window_size, device)
        net.to(device)
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
        loss_fn = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)

        for epoch in range(10):
            print(f"Epoch {epoch}")
            print(f"Total data: {len(self.data_set)}")

            cur = 0
            for data in self.data_set:
                text = data['text']
                text = self.tokenize_bpe.Tokenize(text)
                if len(text) < 20 or len(text) > 1000:
                    cur += 1
                    continue

                text = self.ToOneHot(text)

                loss = 0
                batch = None
                target = None
                for i in range(self.window_size, len(text) - self.window_size):
                    context = text[i-self.window_size:i]
                    context = torch.cat(
                        [context, text[i+1:i+self.window_size+1]])
                    context = context.unsqueeze(0).to(device)

                    if batch is None:
                        batch = context
                    else:
                        batch = torch.cat([batch, context], dim=0)

                    target_idx = text[i].argmax()
                    # 获取目标词的索引而不是one-hot向量
                    if target is None:
                        target = torch.tensor(
                            [target_idx], dtype=torch.long).to(device)
                    else:
                        target = torch.cat([target, torch.tensor(
                            [target_idx], dtype=torch.long).to(device)], dim=0)

                output = net(batch)
                loss += loss_fn(output, target) / len(text)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # scheduler.step()

                cur += 1

                if cur % 100 == 0:
                    print(
                        f"loss: {loss.item()}, in {cur} lr: {scheduler.get_last_lr()[0]}")
