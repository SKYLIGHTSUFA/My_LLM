from torch import nn
import torch

class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        
        self.key = nn.Linear(emb_size, head_size)
        self.query = nn.Linear(emb_size, head_size)
        self.value = nn.Linear(emb_size, head_size)
        # triangle mask shaped (1, max_seq_len, max_seq_len) so it can broadcast over batch
        self.triangle = torch.tril(torch.ones((max_seq_len, max_seq_len))).unsqueeze(0)



    def forward(self, x: float) -> torch.Tensor:
        Q = self.query(x)  # (batch_size, seq_len, head_size)
        K = self.key(x)    # (batch_size, seq_len, head_size)
        V = self.value(x)  # (batch_size, seq_len, head_size)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_size ** 0.5)
        triangle = self.triangle[:, :x.size(1), :x.size(1)]
        attention = attention.masked_fill(triangle == 0, float('-inf'))
        attention = torch.softmax(attention, dim=-1)
        out = torch.matmul(attention, V)
        return out