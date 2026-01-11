from torch import nn
import torch

class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self.emb_size = emb_size
        self.max_len = max_seq_len
        
        self.matrice = nn.Embedding(max_seq_len, emb_size)

    def forward(self, seq_len: int) -> torch.Tensor:
        vector = self.matrice.weight[:seq_len, :]
        return vector