import torch
from torch import nn
from head_attention import HeadAttention
from multi_head_attention import MultiHeadAttention
from feed_forward_network import FeedForward
from decoder import Decoder
from positional_embeddings import PositionalEmbeddings
from token_embedding import TokenEmbeddings

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        emb_size: int,
        num_heads: int,
        head_size: int,
        num_layers: int,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embeddings = TokenEmbeddings(vocab_size, emb_size)
        self.positional_embeddings = PositionalEmbeddings(max_seq_len, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.decoders = nn.ModuleList(
            [
                Decoder(num_heads, emb_size, head_size, max_seq_len, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_layer = nn.Linear(emb_size, vocab_size)

    def forward(self, x: int) -> torch.Tensor:
        token_emb = self.token_embeddings(x)
        pos_emb = self.positional_embeddings(x)
        x = token_emb + pos_emb
        x = self.dropout(x)
        for decoder in self.decoders:
            x = decoder(x)
        out = self.final_layer(x)
        return out
    
    def generate(self, x: torch.Tensor, max_new_tokens: int, do_sample: bool, temperature: float = 1.0) -> torch.Tensor:     
        generated = x
        for _ in range(max_new_tokens):
            ctx = generated[:, -self.max_seq_len:]  # (B, T') [web:16]
            logits = self.forward(ctx)              # (B, T', vocab)
            next_logits = logits[:, -1, :]/temperature          # (B, vocab) [web:16]
            probs = torch.softmax(next_logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
        }, path)
    
    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model