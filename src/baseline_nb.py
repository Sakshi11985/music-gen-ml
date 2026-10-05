"""Naive-Bayes-like baseline: P(right-hand notes | left-hand chord).

Usage:
  python src/baseline_nb.py --data data/processed.pkl --out outputs/nb_sample.mid
"""
import argparse, pickle, random
from collections import Counter, defaultdict
from utils import parse_token, make_token, tokens_to_midi, SPLIT_PITCH


def split_hands(tok):
    pitches, dur = parse_token(tok)
    left = tuple(p for p in pitches if p < SPLIT_PITCH)
    right = tuple(p for p in pitches if p >= SPLIT_PITCH)
    return left, right, dur


class NaiveBayesLike:
    def fit(self, songs):
        self.cond = defaultdict(Counter)   # left chord -> Counter(right notes)
        self.marg = Counter()              # right notes overall (fallback)
        for toks in songs.values():
            for t in toks:
                l, r, _ = split_hands(t)
                self.cond[l][r] += 1
                self.marg[r] += 1
        return self

    def sample(self, left):
        c = self.cond.get(left) or self.marg
        keys, w = zip(*c.items())
        return random.choices(keys, weights=w)[0]

    def predict(self, left):
        return (self.cond.get(left) or self.marg).most_common(1)[0][0]

    def accuracy(self, songs):
        """Top-1 accuracy of predicting the right-hand notes from the left-hand chord."""
        hit = tot = 0
        for toks in songs.values():
            for t in toks:
                l, r, _ = split_hands(t)
                hit += self.predict(l) == r
                tot += 1
        return hit / max(tot, 1)

    def generate(self, left_sequence_tokens):
        """For every left-hand chord/duration in the guide sequence, sample right-hand notes."""
        out = []
        for t in left_sequence_tokens:
            l, _, d = split_hands(t)
            out.append(make_token(l + self.sample(l), d))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed.pkl")
    ap.add_argument("--out", default="outputs/nb_sample.mid")
    ap.add_argument("--length", type=int, default=200)
    a = ap.parse_args()
    D = pickle.load(open(a.data, "rb"))
    nb = NaiveBayesLike().fit(D["train"])
    print(f"NB top-1 accuracy  train: {nb.accuracy(D['train']):.3f}   test: {nb.accuracy(D['test']):.3f}")
    guide = next(iter(D["test"].values()))[: a.length]
    tokens_to_midi(nb.generate(guide), a.out)
    print("Wrote", a.out)


if __name__ == "__main__":
    main()
