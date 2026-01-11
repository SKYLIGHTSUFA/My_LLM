import torch
from torch import nn

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