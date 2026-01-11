from torch import nn
import torch

class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self.matrice = nn.Embedding(max_seq_len, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len)
        bsz, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)  # (seq_len,)
        pos_emb = self.matrice(positions)  # (seq_len, emb_size)
        return pos_emb.unsqueeze(0).expand(bsz, -1, -1)  # (batch_size, seq_len, emb_size)
