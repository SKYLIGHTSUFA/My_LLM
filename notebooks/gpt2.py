import torch
from torch import nn
from torch import nn
import torch

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
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleList(
            [HeadAttention(emb_size, head_size, max_seq_len) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * head_size, emb_size)

    def forward(self, x: float) -> torch.Tensor:
        head_outputs = [head(x) for head in self.heads]
        concat = torch.cat(head_outputs, dim=-1)
        out = self.linear(concat)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, 4 * emb_size)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(4 * emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.feed_forward = FeedForward(emb_size, dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        # Финальный слой нормализации, применяется после блока декодера
        self.final_norm = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block with pre-norm
        # Пропустите тензор x через первый слой нормализации
        normalized_x = self.norm1(x)
        # Затем пропустите его через экземпляр MultiHeadAttention
        attn_output = self.attention(normalized_x)
        # Выходной тензор из блока внимания сложите с исходным x
        x = x + attn_output
        
        # Feed-forward block with pre-norm
        # Получившийся тензор пропустите через второй слой нормализации
        normalized_x = self.norm2(x)
        # Затем подайте его на вход экземпляру FFN
        ff_output = self.feed_forward(normalized_x)
        # Выходной тензор из FFN сложите с тем, что поступил на вход второго слоя нормализации
        x = x + ff_output
        
        # Примените финальную нормализацию после последнего блока декодера
        # x = self.final_norm(x)
        # Верните итоговый тензор размером batch_size × seq_len × emb_size
        return x

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

    
class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.matrice = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.matrice(x)

import torch
from torch import nn


class GPT2(nn.Module):
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
        self.normalization = nn.LayerNorm(emb_size)
        self.final_layer = nn.Linear(emb_size, vocab_size)

    def forward(self, x: int) -> torch.Tensor:
        token_emb = self.token_embeddings(x)
        pos_emb = self.positional_embeddings(x)
        x = token_emb + pos_emb
        x = self.dropout(x)
        for decoder in self.decoders:
            x = decoder(x)
        x = self.normalization(x)
        out = self.final_layer(x)
        return out

    def fit(self, train_loader, valid_loader, num_epoch: int, learning_rate: float):
        device = getattr(self, "device", "cpu")  # если в __init__ сохранили self.device
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(num_epoch):
            # ---- train ----
            self.train()
            train_losses = []

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long()

                logits = self(inputs)                  # (B, T, V)
                logits = logits.reshape(-1, logits.size(-1))   # (B*T, V)  [web:1][web:13]
                targets = targets.reshape(-1)           # (B*T,)          [web:1]

                loss = loss_fn(logits, targets)         # CE expects (N,C) and (N,) [web:1]
                self.loss = loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # ---- valid ----
            self.eval()
            valid_losses = []

            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device).long()

                    logits = self(inputs)
                    logits = logits.reshape(-1, logits.size(-1))
                    targets = targets.reshape(-1)

                    vloss = loss_fn(logits, targets)
                    self.val_loss = vloss
                    valid_losses.append(vloss.item())

    
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        generated = x
        for _ in range(max_new_tokens):
            ctx = generated[:, -self.max_seq_len :]  # (B, T') [web:16]
            logits = self.forward(ctx)  # (B, T', vocab)
            next_logits = logits[:, -1, :] / temperature  # (B, vocab) [web:16]
            
            if do_sample:
                if top_k is not None:
                    values, _ = torch.topk(next_logits, k=top_k, dim=-1)
                    kth = values[..., -1, None] 
                    next_logits = next_logits.masked_fill(next_logits < kth, -float("inf"))

                if top_p is not None:
                    pass

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            probs = torch.softmax(next_logits, dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "vocab_size": self.vocab_size,
                "max_seq_len": self.max_seq_len,
                "emb_size": self.emb_size,
                "num_heads": self.num_heads,
                "head_size": self.head_size,
                "num_layers": self.num_layers,
            },
            path,
        )

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            max_seq_len=checkpoint["max_seq_len"],
            emb_size=checkpoint["emb_size"],
            num_heads=checkpoint["num_heads"],
            head_size=checkpoint["head_size"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model


class GetData(torch.utils.data.Dataset):
    def __init__(self, data: str, seq_len: int, device: int):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int):
        x = torch.tensor(
            [self.data[i] for i in range(idx, idx + self.seq_len)])
        y = torch.tensor(
            [self.data[i] for i in range(idx + 1, idx + self.seq_len + 1)])
        return x.to(self.device), y.to(self.device)