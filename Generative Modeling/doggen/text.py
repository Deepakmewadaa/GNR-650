"""Fixed UTF-8 byte vocabulary: no pretrained tokenizer or embeddings."""
import torch
from torch import nn

VOCAB_SIZE = 258  # padding=0, UTF-8 bytes=1..256, end=257
MAX_LENGTH = 192


def tokenize(prompts, device=None):
    tokens = torch.zeros(len(prompts), MAX_LENGTH, dtype=torch.long)
    for i, prompt in enumerate(prompts):
        data = list(prompt.lower().strip().encode('utf-8')[:MAX_LENGTH - 1])
        tokens[i, :len(data)+1] = torch.tensor([b+1 for b in data] + [257])
    return tokens.to(device)


class TextEncoder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 64, padding_idx=0)
        self.gru = nn.GRU(64, dim, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens):
        seq, _ = self.gru(self.embedding(tokens))
        # Gather the last actual token instead of the hidden state after padding.
        index = tokens.ne(0).sum(1).sub(1).clamp_min(0)
        mask = tokens.ne(0).unsqueeze(-1)
        mean = (seq * mask).sum(1) / mask.sum(1).clamp_min(1)
        last = seq[torch.arange(tokens.shape[0], device=tokens.device), index]
        return self.norm((mean + last) * .5)
