"""Compare generated melodies with real test melodies.

Metrics (all on the melody line, pieces are key-normalised):
  * pitch-class distance : total-variation distance between 12-bin pitch-class histograms (lower = closer)
  * mean |interval|      : average jump in semitones between consecutive notes (real music is ~2-3)
  * distinct-4           : fraction of unique 4-note patterns (very low = repetitive, very high = random)

Usage:  python src/evaluate.py
"""
import pickle, random, numpy as np
from baseline_nb import NaiveBayesLike, Bigram
from generate import load_model, generate
from utils import parse_token

D = pickle.load(open("data/processed.pkl", "rb"))
vocab = D["vocab"]


def pitches(tokens):
    return [parse_token(t)[0][-1] for t in tokens]


def stats(tokens):
    p = pitches(tokens)
    h = np.bincount([x % 12 for x in p], minlength=12) / max(len(p), 1)
    jumps = np.abs(np.diff(p))
    grams = [tuple(p[i:i + 4]) for i in range(len(p) - 3)]
    return h, jumps.mean(), len(set(grams)) / max(len(grams), 1)


real_test = [t for toks in D["test"].values() for t in toks]
real_train = [t for toks in D["train"].values() for t in toks]
h_real, _, _ = stats(real_test)

songs = list(D["test"].keys())[:6]
nb = NaiveBayesLike().fit(D["train_full"])
bg = Bigram(vocab, D["tok2id"]).fit(D["train"])
model, _ = load_model()

gen = {"random tokens": [], "Naive Bayes-like": [], "Bigram Markov": [], "LSTM (T=1.0)": []}
for s in songs:
    gen["random tokens"] += [vocab[i] for i in np.random.randint(1, len(vocab), 100)]
    gen["Naive Bayes-like"] += nb.generate(D["test_full"][s][:100])
    gen["Bigram Markov"] += bg.generate(D["test"][s][:1], 100)
    gen["LSTM (T=1.0)"] += generate(model, vocab, D["test"][s][:16], 100, 1.0)

print(f"{'':20s} {'pc-dist':>8s} {'|interval|':>11s} {'distinct-4':>11s}")
for name, toks in [("real train", real_train), ("real test", real_test)] + list(gen.items()):
    h, j, d4 = stats(toks)
    print(f"{name:20s} {0.5 * np.abs(h - h_real).sum():8.3f} {j:11.2f} {d4:11.3f}")
