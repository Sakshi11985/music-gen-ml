"""Compare generated music with real test music (pitch-histogram distance; lower = closer).

Usage:  python src/evaluate.py
"""
import pickle, numpy as np
from baseline_nb import NaiveBayesLike
from generate import load_model, generate
from utils import pitch_histogram

D = pickle.load(open("data/processed.pkl", "rb"))
real = [t for toks in D["test"].values() for t in toks]
guide = next(iter(D["test"].values()))
tv = lambda a, b: 0.5 * np.abs(a - b).sum()            # total-variation distance

nb = NaiveBayesLike().fit(D["train"])
nb_out = nb.generate(guide[:300])
model, vocab = load_model()
lstm_out = generate(model, vocab, guide[:16], 300, 1.0)
rnd = [vocab[i] for i in np.random.randint(1, len(vocab), 300)]   # random-note baseline

h = pitch_histogram(real)
for name, toks in [("random tokens", rnd), ("Naive Bayes-like", nb_out), ("LSTM", lstm_out)]:
    print(f"{name:18s} pitch-histogram distance to real music: {tv(h, pitch_histogram(toks)):.3f}")
print(f"NB top-1 accuracy (test): {nb.accuracy(D['test']):.3f}")
