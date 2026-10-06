"""Compare generated melodies with real test melodies. Writes outputs/generation_metrics.json.

  * pc-dist     : distance between 12-bin pitch-class histograms and the real test music (lower = closer)
  * |interval|  : average jump in semitones between consecutive notes (real music is about 2-3)
  * distinct-4  : fraction of unique 4-note patterns (very low = repetitive, 1.0 = no repeats at all)

Usage:  python src/evaluate.py
"""
import json, os, pickle, numpy as np
from baseline_nb import NaiveBayesLike, Bigram
from generate import load_model, generate
from utils import parse_token

D = pickle.load(open("data/processed.pkl", "rb"))
vocab = D["vocab"]


def stats(tokens):
    p = [parse_token(t)[0][-1] for t in tokens]
    h = np.bincount([x % 12 for x in p], minlength=12) / max(len(p), 1)
    grams = [tuple(p[i:i + 4]) for i in range(len(p) - 3)]
    return h, float(np.abs(np.diff(p)).mean()), len(set(grams)) / max(len(grams), 1)


real_test = [t for toks in D["test"].values() for t in toks]
real_train = [t for toks in D["train"].values() for t in toks]
h_real, _, _ = stats(real_test)

songs = list(D["test"].keys())[:6]
nb = NaiveBayesLike().fit(D["train_full"])
bg = Bigram(vocab, D["tok2id"]).fit(D["train"])
lstm, _ = load_model()
try:
    from mlp import load_mlp, generate_mlp
    mlp, _ = load_mlp()
except Exception as e:
    mlp = None
    print("(MLP skipped:", e, ")")

gen = {"Random notes": [], "Naive Bayes-like": [], "Bigram Markov": [], "LSTM": []}
if mlp is not None:
    gen["Vanilla NN (MLP)"] = []
for s in songs:
    gen["Random notes"] += [vocab[i] for i in np.random.randint(1, len(vocab), 100)]
    gen["Naive Bayes-like"] += nb.generate(D["test_full"][s][:100])
    gen["Bigram Markov"] += bg.generate(D["test"][s][:1], 100)
    gen["LSTM"] += generate(lstm, vocab, D["test"][s][:16], 100, 1.0)
    if mlp is not None:
        gen["Vanilla NN (MLP)"] += generate_mlp(mlp, vocab, D["test"][s][:16], 100, 1.0)

result = {}
print(f"{'':20s} {'pc-dist':>8s} {'|interval|':>11s} {'distinct-4':>11s}")
for name, toks in [("Real music (train)", real_train), ("Real music (test)", real_test)] + list(gen.items()):
    h, j, d4 = stats(toks)
    pc = 0.5 * float(np.abs(h - h_real).sum())
    result[name] = {"pc": round(pc, 3), "jump": round(j, 2), "d4": round(d4, 3)}
    print(f"{name:20s} {pc:8.3f} {j:11.2f} {d4:11.3f}")
os.makedirs("outputs", exist_ok=True)
json.dump(result, open("outputs/generation_metrics.json", "w"), indent=2)
print("\nSaved outputs/generation_metrics.json")
