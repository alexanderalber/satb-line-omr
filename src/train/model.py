"""The line model and the CTC plumbing around it.

Lifted out of `scripts/14_smoke_train.py` so that the overfit gate and the real
run share one definition instead of drifting apart. The architecture is
unchanged from the smoke test -- four conv blocks collapsing the height, a
2x256 BiLSTM, CTC on top, width reduced by 4 as the frame budget in
`work/13_vocab_frames.json` requires.

Training rig, not delivered code. What ships is the exported graph, not this.
"""
import numpy as np
import torch
import torch.nn as nn

WIDTH_REDUCTION = 4
BLANK = 0


class CRNN(nn.Module):
    def __init__(self, n_classes: int, height: int = 128):
        super().__init__()
        self.height = height
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                       # H/2,  W/2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),                       # H/4,  W/4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                       # H/8,  W/4
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                       # H/16, W/4
        )
        feat = 128 * (height // 16)
        self.proj = nn.Linear(feat, 256)
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.1)
        self.out = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.cnn(x)                        # (B, C, H', W')
        b, c, h, w = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        x = self.proj(x)
        x, _ = self.rnn(x)
        return self.out(x)                     # (B, W', n_classes)


def greedy(logits, i2w) -> list[str]:
    """CTC best-path decode of one sample's logits."""
    ids = logits.argmax(-1).tolist()
    out, prev = [], -1
    for k in ids:
        if k != prev and k != BLANK:
            out.append(i2w[k])
        prev = k
    return out


def ser(ref: list[str], hyp: list[str]) -> float:
    """Symbol error rate: Levenshtein distance over token sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m] / n


def edits(ref: list[str], hyp: list[str]) -> list[tuple[str, str, str]]:
    """The individual edits of the alignment: (op, ref_token, hyp_token).

    Needed for contract 4 of the frontend handoff, which wants the error
    *distribution* and not just a rate: a substitution of `4` by `2` is a
    duration error, `g` by `a` a pitch error, and those cost the editor
    different amounts of work.
    """
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                out.append(("sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            out.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            out.append(("ins", "", hyp[j - 1]))
            j -= 1
    out.reverse()
    return out


def collate(batch, w2i, device, height=128):
    """Pad a batch to equal width; ink becomes high values, paper 0.

    Samples are kept as uint8 (see `train.data.to_store`) and become float here,
    one batch at a time. Float arrays are still accepted so that callers holding
    a freshly normalised staff -- the evaluation on the real PDF does -- do not
    have to know about the storage format.
    """
    widths = [s["x"].shape[1] for s in batch]
    maxw = max(widths)
    x = np.ones((len(batch), 1, height, maxw), dtype=np.float32)
    for i, s in enumerate(batch):
        a = s["x"]
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        x[i, 0, :, :a.shape[1]] = a
    xt = torch.from_numpy(1.0 - x).to(device)
    targets = [torch.tensor([w2i[t] for t in s["tokens"]], dtype=torch.long)
               for s in batch]
    in_lens = torch.tensor([w // WIDTH_REDUCTION for w in widths], dtype=torch.long)
    tg_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return xt, torch.cat(targets).to(device), in_lens, tg_lens
