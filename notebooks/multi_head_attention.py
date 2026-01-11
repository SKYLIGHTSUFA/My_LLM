from torch import nn
import torch
from notebooks.head_attention import HeadAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_size: int,
        head_size: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.heads = nn.ModuleList(
            [HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * head_size, emb_size)

    def forward(self, x: float) -> torch.Tensor:
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        out = self.linear(concat)
        return out
