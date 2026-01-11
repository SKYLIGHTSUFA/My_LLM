from torch import nn
import torch
class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.matrice = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.matrice(x)