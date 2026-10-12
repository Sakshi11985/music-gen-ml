"""Generate music with the trained LSTM.

Usage:  python src/generate.py --length 200 --temperature 1.0 --out outputs/lstm_sample.mid
"""
import argparse, pickle, random, torch
from model import MusicLSTM
from utils import tokens_to_midi, path


def load_model(ckpt=path("checkpoints", "lstm.pt"), dev="cpu"):
    ck = torch.load(ckpt, map_location=dev, weights_only=True)
    m = MusicLSTM(len(ck["vocab"]), hidden=ck["hidden"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    return m, ck["vocab"]


@torch.no_grad()
def generate(model, vocab, seed_tokens, length=200, temperature=1.0, dev="cpu"):
    tok2id = {t: i for i, t in enumerate(vocab)}
    ids = [tok2id.get(t, 0) for t in seed_tokens]
    x = torch.tensor([ids], device=dev)
    logits, state = model(x)
    out = list(seed_tokens)
    for _ in range(length):
        probs = torch.softmax(logits[0, -1] / max(temperature, 1e-3), -1)
        probs[0] = 0                       # never emit <unk>
        nxt = torch.multinomial(probs / probs.sum(), 1).item()
        out.append(vocab[nxt])
        logits, state = model(torch.tensor([[nxt]], device=dev), state)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=path("checkpoints", "lstm.pt"))
    ap.add_argument("--data", default=path("data", "processed.pkl"))
    ap.add_argument("--length", type=int, default=200)
    ap.add_argument("--seed_len", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None, help="set for a reproducible sample")
    ap.add_argument("--out", default=path("outputs", "lstm_sample.mid"))
    a = ap.parse_args()
    if a.seed is not None:
        random.seed(a.seed); torch.manual_seed(a.seed)
    model, vocab = load_model(a.ckpt)
    with open(a.data, "rb") as f:
        D = pickle.load(f)
    seed = next(iter(D["test"].values()))[: a.seed_len]
    tokens_to_midi(generate(model, vocab, seed, a.length, a.temperature), a.out)
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
