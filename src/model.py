import torch, torch.nn as nn


class MusicLSTM(nn.Module):
    """Embedding -> LSTM(256) -> Linear(vocab). Predicts the next note-event token."""
    def __init__(self, vocab_size, emb=128, hidden=256, layers=1, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        h, state = self.lstm(self.emb(x), state)
        return self.out(h), state
