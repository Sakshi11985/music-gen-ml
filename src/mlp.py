"""Vanilla neural network (MLP) next-note model - the paper's second model.
It sees the previous `context` notes at once (no recurrent memory) and predicts the next one.

Usage:  python src/mlp.py --epochs 30
"""
import argparse, math, pickle, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import tokens_to_midi


class MusicMLP(nn.Module):
    def __init__(self, vocab_size, context=32, emb=32, hidden=1024, dropout=0.0):
        super().__init__()
        self.context = context
        self.emb = nn.Embedding(vocab_size, emb)
        self.net = nn.Sequential(nn.Linear(context * emb, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, vocab_size))

    def forward(self, x):                       # x: (batch, context) token ids
        return self.net(self.emb(x).flatten(1))


def make_samples(songs, tok2id, context):
    """Every position i>=1 of every piece becomes one sample:
    x = previous `context` token ids (left-padded with 0), y = the token at i.
    This uses exactly the same targets as the bigram baseline."""
    X, Y = [], []
    for toks in songs.values():
        ids = [tok2id.get(t, 0) for t in toks]
        for i in range(1, len(ids)):
            prev = ids[max(0, i - context):i]
            X.append([0] * (context - len(prev)) + prev)
            Y.append(ids[i])
    return torch.tensor(X), torch.tensor(Y)


def run_epoch(model, X, Y, loss_fn, opt, bs, dev):
    train = opt is not None
    model.train(train)
    perm = torch.randperm(len(X)) if train else torch.arange(len(X))
    tot_loss = tot_hit = tot_n = 0
    with torch.set_grad_enabled(train):
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            x, y = X[idx].to(dev), Y[idx].to(dev)
            n = (y != 0).sum().item()
            if n == 0:
                continue
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * n; tot_n += n
            tot_hit += ((logits.argmax(-1) == y) & (y != 0)).sum().item()
    return tot_loss / tot_n, tot_hit / tot_n


def load_mlp(ckpt="checkpoints/mlp.pt", dev="cpu"):
    ck = torch.load(ckpt, map_location=dev)
    m = MusicMLP(len(ck["vocab"]), ck["context"], ck["emb"], ck["hidden"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["vocab"]


@torch.no_grad()
def generate_mlp(model, vocab, seed_tokens, length=100, temperature=1.0, dev="cpu"):
    tok2id = {t: i for i, t in enumerate(vocab)}
    ids = [tok2id.get(t, 0) for t in seed_tokens]
    out = list(seed_tokens)
    for _ in range(length):
        prev = ids[-model.context:]
        x = torch.tensor([[0] * (model.context - len(prev)) + prev], device=dev)
        probs = torch.softmax(model(x)[0] / max(temperature, 1e-3), -1)
        probs[0] = 0                                  # never emit <unk>
        nxt = torch.multinomial(probs / probs.sum(), 1).item()
        ids.append(nxt); out.append(vocab[nxt])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed.pkl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--context", type=int, default=32, help="how many previous notes the network sees")
    ap.add_argument("--emb", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=1024, help="1024 as in the Stanford paper")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--ckpt", default="checkpoints/mlp.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = pickle.load(open(a.data, "rb"))
    Xtr, Ytr = make_samples(D["train"], D["tok2id"], a.context)
    Xva, Yva = make_samples(D["val"], D["tok2id"], a.context)
    Xte, Yte = make_samples(D["test"], D["tok2id"], a.context)
    print(f"samples  train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}  | vocab {len(D['vocab'])} | device {dev}")

    model = MusicMLP(len(D["vocab"]), a.context, a.emb, a.hidden, a.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lf = nn.CrossEntropyLoss(ignore_index=0)
    tr_hist, va_hist, best = [], [], 1e9
    for ep in range(1, a.epochs + 1):
        trl, tra = run_epoch(model, Xtr, Ytr, lf, opt, a.batch, dev)
        val, vaa = run_epoch(model, Xva, Yva, lf, None, a.batch, dev)
        tr_hist.append(trl); va_hist.append(val)
        print(f"epoch {ep:3d} | train loss {trl:.4f} acc {tra:.3f} | val loss {val:.4f} acc {vaa:.3f}")
        if val < best:
            best = val
            torch.save(dict(model=model.state_dict(), vocab=D["vocab"], context=a.context,
                            emb=a.emb, hidden=a.hidden), a.ckpt)

    model, vocab = load_mlp(a.ckpt, dev)
    tel, tea = run_epoch(model, Xte, Yte, lf, None, a.batch, dev)
    print(f"\nTEST  loss {tel:.4f}  perplexity {math.exp(tel):.2f}  next-note accuracy {tea:.3f}")

    plt.plot(tr_hist, label="train"); plt.plot(va_hist, label="val")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.legend(); plt.title("Vanilla NN (MLP) loss")
    plt.savefig("outputs/mlp_loss.png", dpi=150, bbox_inches="tight")
    seed = next(iter(D["test"].values()))[:16]
    tokens_to_midi(generate_mlp(model, vocab, seed, 200, 1.0), "outputs/mlp_sample.mid")
    print("Saved outputs/mlp_loss.png and outputs/mlp_sample.mid")


if __name__ == "__main__":
    main()
