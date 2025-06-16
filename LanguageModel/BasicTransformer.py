import torch
import torch.nn as nn

from Tokenize.BPE import TokenizeBPE


class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int = None):
        super().__init__()

        if output_size is None:
            output_size = input_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self,
                 tokenize_bpe: TokenizeBPE,
                 embedding_size: int = 100,
                 window_size: int = 7,
                 device: str = "cpu"):
        super().__init__()
        self.tokenize_bpe = tokenize_bpe
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.device = device

        self.mha = nn.MultiheadAttention(
            embed_dim=100,
            num_heads=4,
            dropout=0.5
        )

        self.norm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x1 = self.mha(x, x, x)
        x1 = x + x1
        x1 = self.norm(x1)
        return x1
