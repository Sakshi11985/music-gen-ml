"""Train the LSTM.

Usage:  python src/train.py --epochs 30

Training uses overlapping windows of `seq_len` tokens from the training part of every piece.
Validation and test use `evaluate_excerpts`: the model reads each held-out excerpt from its start
and predicts every next token. This uses every piece (most val/test excerpts are shorter than one
window) and exactly the same targets as compare_models.py. <unk> targets are ignored everywhere.
"""
import argparse, pickle, math, random, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import MusicLSTM
from utils import path, ensure_parent

UNK = 0


def make_windows(songs, tok2id, seq_len):
    """Input/target windows shifted by one token, stride seq_len//2."""
    X, Y = [], []
    for toks in songs.values():
        ids = [tok2id.get(t, UNK) for t in toks]
        for i in range(0, len(ids) - seq_len, max(seq_len // 2, 1)):
            X.append(ids[i:i + seq_len]); Y.append(ids[i + 1:i + seq_len + 1])
    return torch.tensor(X), torch.tensor(Y)


def train_epoch(model, X, Y, opt, bs, dev):
    model.train()
    perm = torch.randperm(len(X))
    tot_loss = tot_hit = tot_n = 0
    for i in range(0, len(X), bs):
        idx = perm[i:i + bs]
        x, y = X[idx].to(dev), Y[idx].to(dev)
        logits, _ = model(x)
        logits, y = logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        mask = y != UNK
        n = mask.sum().item()
        if n == 0:
            continue
        loss = F.cross_entropy(logits, y, ignore_index=UNK)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot_loss += loss.item() * n; tot_n += n
        tot_hit += (logits.argmax(-1)[mask] == y[mask]).sum().item()
    return tot_loss / tot_n, tot_hit / tot_n


@torch.no_grad()
def evaluate_excerpts(model, songs, tok2id, dev="cpu"):
    """(mean NLL, accuracy, number of targets) over every held-out excerpt, read from its start."""
    model.eval()
    nll = hits = n = 0
    for toks in songs.values():
        ids = [tok2id.get(t, UNK) for t in toks]
        if len(ids) < 2:
            continue
        logits, _ = model(torch.tensor([ids[:-1]], device=dev))
        y = torch.tensor(ids[1:], device=dev)
        mask = y != UNK
        if mask.sum() == 0:
            continue
        lg = logits[0][mask]
        nll += F.cross_entropy(lg, y[mask], reduction="sum").item()
        hits += (lg.argmax(-1) == y[mask]).sum().item()
        n += mask.sum().item()
    return nll / n, hits / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=path("data", "processed.pkl"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=path("checkpoints", "lstm.pt"))
    ap.add_argument("--plot", default=path("outputs", "lstm_loss.png"))
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with open(a.data, "rb") as f:
        D = pickle.load(f)
    tok2id = D["tok2id"]
    Xtr, Ytr = make_windows(D["train"], tok2id, a.seq_len)
    print(f"train windows {len(Xtr)} | val/test excerpts {len(D['val'])}/{len(D['test'])} | "
          f"vocab {len(D['vocab'])} | device {dev}")

    model = MusicLSTM(len(D["vocab"]), hidden=a.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    tr_hist, va_hist, best = [], [], float("inf")
    ensure_parent(a.ckpt)
    for ep in range(1, a.epochs + 1):
        trl, tra = train_epoch(model, Xtr, Ytr, opt, a.batch, dev)
        val, vaa, _ = evaluate_excerpts(model, D["val"], tok2id, dev)
        tr_hist.append(trl); va_hist.append(val)
        flag = ""
        if val < best:
            best, flag = val, "  * saved"
            torch.save(dict(model=model.state_dict(), vocab=D["vocab"], hidden=a.hidden), a.ckpt)
        print(f"epoch {ep:3d} | train loss {trl:.4f} acc {tra:.3f} | val loss {val:.4f} acc {vaa:.3f}{flag}")

    ck = torch.load(a.ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(ck["model"])
    tel, tea, n = evaluate_excerpts(model, D["test"], tok2id, dev)
    print(f"\nTEST ({n} targets)  loss {tel:.4f}  perplexity {math.exp(tel):.2f}  next-note accuracy {tea:.3f}")

    plt.plot(range(1, len(tr_hist) + 1), tr_hist, label="train")
    plt.plot(range(1, len(va_hist) + 1), va_hist, label="val")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy (nats)"); plt.legend(); plt.title("LSTM loss")
    plt.savefig(ensure_parent(a.plot), dpi=150, bbox_inches="tight")
    print("Saved", a.ckpt, "and", a.plot)


if __name__ == "__main__":
    main()
