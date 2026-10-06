"""Two non-neural baselines.

1) Naive Bayes-like:  P(melody note | bass note)     (from the Stanford paper: P(note | chord))
2) Bigram Markov:     P(next melody token | current melody token)  <- fair baseline for the LSTM

Usage:  python src/baseline_nb.py --data data/processed.pkl
"""
import argparse, math, pickle, random
from collections import Counter, defaultdict
from utils import parse_token, make_token, tokens_to_midi


class NaiveBayesLike:
    """Context = lowest pitch (bass) of the event; predicts the highest pitch (melody)."""
    def fit(self, songs_full):
        self.cond = defaultdict(Counter)
        self.marg = Counter()
        for toks in songs_full.values():
            for t in toks:
                p, _ = parse_token(t)
                self.cond[p[0]][p[-1]] += 1
                self.marg[p[-1]] += 1
        return self

    def _dist(self, bass):
        return self.cond.get(bass) or self.marg

    def sample(self, bass):
        keys, w = zip(*self._dist(bass).items())
        return random.choices(keys, weights=w)[0]

    def accuracy(self, songs_full):
        hit = tot = 0
        for toks in songs_full.values():
            for t in toks:
                p, _ = parse_token(t)
                hit += self._dist(p[0]).most_common(1)[0][0] == p[-1]
                tot += 1
        return hit / max(tot, 1)

    def generate(self, guide_full_tokens):
        """Keep the guide's bass line + durations, sample a melody note for every event."""
        out = []
        for t in guide_full_tokens:
            p, d = parse_token(t)
            out.append(make_token({p[0], self.sample(p[0])}, d))
        return out


class Bigram:
    """P(next | current) over melody tokens with add-k smoothing."""
    def __init__(self, vocab, tok2id, k=0.1):
        self.V, self.tok2id, self.k, self.vocab = len(vocab), tok2id, k, vocab
        self.c = defaultdict(Counter)

    def _ids(self, toks):
        return [self.tok2id.get(t, 0) for t in toks]

    def fit(self, songs):
        for toks in songs.values():
            ids = self._ids(toks)
            for a, b in zip(ids[:-1], ids[1:]):
                self.c[a][b] += 1
        return self

    def prob(self, a, b):
        ca = self.c[a]
        return (ca[b] + self.k) / (sum(ca.values()) + self.k * self.V)

    def evaluate(self, songs):
        """(loss, perplexity, top-1 accuracy), ignoring <unk> targets - same rule as train.py."""
        nll = hit = n = 0
        for toks in songs.values():
            ids = self._ids(toks)
            for a, b in zip(ids[:-1], ids[1:]):
                if b == 0:
                    continue
                nll -= math.log(self.prob(a, b)); n += 1
                best = self.c[a].most_common(1)[0][0] if self.c[a] else -1
                hit += best == b
        return nll / n, math.exp(nll / n), hit / n

    def generate(self, seed_tokens, length=200):
        out = list(seed_tokens)
        cur = self.tok2id.get(out[-1], 0)
        for _ in range(length):
            ca = self.c[cur]
            if not ca:
                cur = random.randrange(1, self.V)
            else:
                keys, w = zip(*ca.items())
                cur = random.choices(keys, weights=w)[0]
                if cur == 0:
                    cur = random.randrange(1, self.V)
            out.append(self.vocab[cur])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed.pkl")
    ap.add_argument("--out", default="outputs/nb_sample.mid")
    ap.add_argument("--length", type=int, default=200)
    a = ap.parse_args()
    D = pickle.load(open(a.data, "rb"))

    nb = NaiveBayesLike().fit(D["train_full"])
    print(f"Naive Bayes-like  P(melody | bass)   top-1 accuracy  train {nb.accuracy(D['train_full']):.3f}  test {nb.accuracy(D['test_full']):.3f}")
    guide = next(iter(D["test_full"].values()))[: a.length]
    tokens_to_midi(nb.generate(guide), a.out)

    bg = Bigram(D["vocab"], D["tok2id"]).fit(D["train"])
    loss, ppl, acc = bg.evaluate(D["test"])
    print(f"Bigram Markov     P(next | current)  test loss {loss:.4f}  perplexity {ppl:.2f}  next-note accuracy {acc:.3f}")
    seed = next(iter(D["test"].values()))[:1]
    tokens_to_midi(bg.generate(seed, a.length), a.out.replace("nb_", "bigram_"))
    print("Wrote", a.out, "and", a.out.replace("nb_", "bigram_"))


if __name__ == "__main__":
    main()
