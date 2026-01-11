import torch
from torch import nn
from multi_head_attention import MultiHeadAttention
from feed_forward_network import FeedForward


class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.feed_forward = FeedForward(emb_size, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block
        attn_output = self.attention(x)
        x = x + attn_output
        x = self.norm1(x)

        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.norm2(x)
        return x