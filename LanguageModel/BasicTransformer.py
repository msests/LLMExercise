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
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self,
                 embedding_size: int = 512):
        super().__init__()
        self.embedding_size = embedding_size

        self.mha = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=4,
            dropout=0.5
        )

    def forward(self, x, mask):
        # 调整输入维度顺序：(batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)

        # 将布尔掩码转换为float掩码，True位置设为0，False位置设为-inf
        mask = mask.float().masked_fill(mask == 0, float('-inf'))

        # 计算注意力
        attn_output, _ = self.mha(x, x, x, attn_mask=mask)

        # 将输出维度调整回原来的顺序
        return attn_output.transpose(0, 1)


class DecoderBlock(nn.Module):
    def __init__(self,
                 embedding_size: int,
                 device: str = "cpu"):
        super().__init__()

        self.device = device
        self.attn_block = AttentionBlock(embedding_size)

        self.norm1 = nn.LayerNorm(embedding_size)

        self.mlp = MLP(
            input_size=embedding_size,
            hidden_size=embedding_size * 4
        )

        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        # Calculate attention mask
        batch_size, seq_len, embedding_size = x.shape
        # 确保掩码维度与序列长度匹配
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.bool()

        # Calculate attention weights
        x1 = self.attn_block(x, mask=mask)
        x1 = x1 + x
        x1 = self.norm1(x1)

        # Calculate MLP output
        x2 = self.mlp(x1)
        x2 = x2 + x1
        x2 = self.norm2(x2)

        return x2


class DecoderOnlyTransformer(nn.Module):
    def __init__(self,
                 tokenize_bpe: TokenizeBPE,
                 embedding_size: int = 512,
                 seq_len: int = 128,
                 device: str = "cpu"):
        super().__init__()
        self.tokenize_bpe = tokenize_bpe
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.device = device

        self.position_encoding = None
        self.PreCalcPositionEncoding(batch_size=128)

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenize_bpe.token_list),
            embedding_dim=embedding_size
        )

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embedding_size, device)
            for _ in range(6)
        ])

        # 添加输出层，将embedding_size映射到词表大小
        self.output_layer = nn.Linear(
            embedding_size, len(tokenize_bpe.token_list))

    def PreCalcPositionEncoding(self,
                                batch_size: int):
        position = torch.arange(
            self.embedding_size // 2).expand(self.seq_len, -1).to(self.device)
        position = position / \
            torch.pow(
                10000, 2 * torch.arange(self.embedding_size // 2) / self.embedding_size).to(self.device)
        position = position.unsqueeze(0).expand(batch_size, -1, -1)

        # 交错合并正弦和余弦值
        sin_pos = torch.sin(position)
        cos_pos = torch.cos(position)
        self.position_encoding = torch.cat([sin_pos, cos_pos], dim=-1)

    def AddPositionEmbedding(self, x):
        return x + self.position_encoding

    def forward(self, x):
        # 确保输入张量是Long类型
        x = x.long()
        x1 = self.embedding(x)
        x1 = self.AddPositionEmbedding(x1)

        for block in self.decoder_blocks:
            x1 = block(x1)

        # 将输出映射到词表大小
        output = self.output_layer(x1)

        # 调整维度顺序以匹配CrossEntropyLoss的要求
        output = output.transpose(1, 2)  # [batch_size, num_classes, seq_len]

        return output
