"""Evaluate EVERY model on exactly the same test targets (a fair comparison).

For each held-out piece excerpt and each position i >= 1, the model predicts token i from the
tokens before it. Targets that are <unk> are ignored for all models.

Usage:  python src/compare_models.py        (writes outputs/results.json)
"""
import json, math, pickle
from collections import Counter
import torch, torch.nn.functional as F
from baseline_nb import Bigram
from utils import path, ensure_parent

with open(path("data", "processed.pkl"), "rb") as f:
    D = pickle.load(f)
tok2id, vocab = D["tok2id"], D["vocab"]
V = len(vocab)
test_ids = [[tok2id.get(t, 0) for t in toks] for toks in D["test"].values()]
n_targets = sum(1 for ids in test_ids for t in ids[1:] if t != 0)
rows = []


def add(name, nll, hits, note=""):
    rows.append({"model": name, "perplexity": round(math.exp(nll / n_targets), 2),
                 "accuracy": round(100 * hits / n_targets, 1), "note": note})


# 1) Unigram: always uses overall note frequencies (no context at all)
cnt = Counter(t for toks in D["train"].values() for t in (tok2id.get(x, 0) for x in toks) if t != 0)
total, k = sum(cnt.values()), 0.1
best = cnt.most_common(1)[0][0]
nll = hits = 0
for ids in test_ids:
    for t in ids[1:]:
        if t != 0:
            nll -= math.log((cnt[t] + k) / (total + k * V)); hits += t == best
add("Unigram (no context)", nll, hits)

# 2) Bigram Markov
bg = Bigram(vocab, tok2id).fit(D["train"])
loss, _, acc = bg.evaluate(D["test"])
add("Bigram Markov", loss * n_targets, acc * n_targets)

# 3) Vanilla NN (MLP), if trained
try:
    from mlp import load_mlp, make_samples
    mlp, _ = load_mlp()
    X, Y = make_samples(D["test"], tok2id, mlp.context)
    nll = hits = 0
    with torch.no_grad():
        for i in range(0, len(X), 512):
            x, y = X[i:i + 512], Y[i:i + 512]
            m = y != 0
            if m.sum() == 0:
                continue
            lg = mlp(x)
            nll += F.cross_entropy(lg[m], y[m], reduction="sum").item()
            hits += (lg.argmax(-1)[m] == y[m]).sum().item()
    add("Vanilla NN (MLP)", nll, hits, f"sees previous {mlp.context} notes")
except Exception as e:
    print("(MLP skipped:", e, ")")

# 4) LSTM (reads each whole excerpt from its start)
try:
    from generate import load_model
    from train import evaluate_excerpts
    lstm, _ = load_model()
    loss, acc, n = evaluate_excerpts(lstm, D["test"], tok2id)
    assert n == n_targets, (n, n_targets)
    add(f"LSTM ({lstm.lstm.hidden_size} units)", loss * n, acc * n)
except Exception as e:
    print("(LSTM skipped:", e, ")")

print(f"\nAll models scored on the same {n_targets} test targets ({len(test_ids)} piece excerpts)\n")
print(f"{'Model':24s} {'perplexity':>11s} {'accuracy %':>11s}")
for r in rows:
    print(f"{r['model']:24s} {r['perplexity']:11.2f} {r['accuracy']:11.1f}   {r['note']}")
out = ensure_parent(path("outputs", "results.json"))
with open(out, "w") as f:
    json.dump({"n_targets": n_targets, "rows": rows}, f, indent=2)
print("\nSaved", out)
