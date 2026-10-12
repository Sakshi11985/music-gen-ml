"""Compare generated melodies with real test melodies. Writes outputs/generation_metrics.json.

  * pc-dist     : total-variation distance between 12-bin pitch-class histograms of the
                  generated music and the real test music (0 = identical, 1 = disjoint)
  * |interval|  : average jump in semitones between consecutive melody notes (real music is about 2-3)
  * distinct-4  : fraction of unique 4-note patterns (very low = repetitive, 1.0 = no repeats at all)

Usage:  python src/evaluate.py [--seed 0] [--n_songs 0] [--length 100]
"""
import argparse, json, pickle, random
import numpy as np
from baseline_nb import NaiveBayesLike, Bigram
from utils import parse_token, path, ensure_parent


def stats(tokens):
    p = [parse_token(t)[0][-1] for t in tokens if parse_token(t)[0]]
    h = np.bincount([x % 12 for x in p], minlength=12) / max(len(p), 1)
    grams = [tuple(p[i:i + 4]) for i in range(len(p) - 3)]
    jump = float(np.abs(np.diff(p)).mean()) if len(p) > 1 else 0.0
    return h, jump, len(set(grams)) / max(len(grams), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_songs", type=int, default=0, help="test pieces used as seeds / guides (0 = all)")
    ap.add_argument("--length", type=int, default=100, help="generated note events per piece")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    try:
        import torch; torch.manual_seed(a.seed)
    except ImportError:
        pass

    with open(path("data", "processed.pkl"), "rb") as f:
        D = pickle.load(f)
    vocab = D["vocab"]
    real_test = [t for toks in D["test"].values() for t in toks]
    real_train = [t for toks in D["train"].values() for t in toks]
    h_real, _, _ = stats(real_test)

    songs = list(D["test"].keys())[: a.n_songs or None]
    nb = NaiveBayesLike().fit(D["train_full"])
    bg = Bigram(vocab, D["tok2id"]).fit(D["train"])

    neural = {}
    try:
        from generate import load_model, generate
        neural["LSTM"] = (load_model()[0], generate)
    except Exception as e:
        print("(LSTM skipped:", e, ")")
    try:
        from mlp import load_mlp, generate_mlp
        neural["Vanilla NN (MLP)"] = (load_mlp()[0], generate_mlp)
    except Exception as e:
        print("(MLP skipped:", e, ")")

    gen = {"Random notes": [], "Naive Bayes-like": [], "Bigram Markov": []}
    gen.update({name: [] for name in neural})
    for s in songs:
        gen["Random notes"] += [vocab[i] for i in np.random.randint(1, len(vocab), a.length)]
        gen["Naive Bayes-like"] += nb.generate(D["test_full"][s][: a.length])
        gen["Bigram Markov"] += bg.generate(D["test"][s][:1], a.length)[1:]          # drop the seed
        for name, (model, fn) in neural.items():
            seed = D["test"][s][:16]
            gen[name] += fn(model, vocab, seed, a.length, 1.0)[len(seed):]          # drop the seed

    result = {}
    print(f"{'':20s} {'pc-dist':>8s} {'|interval|':>11s} {'distinct-4':>11s}")
    for name, toks in [("Real music (train)", real_train), ("Real music (test)", real_test)] + list(gen.items()):
        h, j, d4 = stats(toks)
        pc = 0.5 * float(np.abs(h - h_real).sum())
        result[name] = {"pc": round(pc, 3), "jump": round(j, 2), "d4": round(d4, 3)}
        print(f"{name:20s} {pc:8.3f} {j:11.2f} {d4:11.3f}")
    out = ensure_parent(path("outputs", "generation_metrics.json"))
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved", out)


if __name__ == "__main__":
    main()
