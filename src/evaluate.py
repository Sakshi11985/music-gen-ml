"""Evaluate generated melody against real test music.

Metrics:
- Pitch-class histogram distance
- Average melodic interval
- Repetition rate
- Distinct-4
- Next-token accuracy for NB and LSTM

Usage:
    python src/evaluate.py
"""

import pickle
import numpy as np
import torch

from baseline_nb import NaiveBayesLike
from generate import load_model, generate


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

with open("data/processed.pkl", "rb") as f:
    D = pickle.load(f)

real = [
    token
    for tokens in D["test"].values()
    for token in tokens
]

guide = next(iter(D["test"].values()))


# ------------------------------------------------------------
# Token helpers
# ------------------------------------------------------------

def token_pitch(token):
    """Extract pitch from token such as '74_0.25'."""
    try:
        return int(token.split("_")[0])
    except Exception:
        return None


def pitch_class_histogram(tokens):
    """Normalized histogram over 12 pitch classes."""
    hist = np.zeros(12, dtype=float)

    for token in tokens:
        p = token_pitch(token)

        if p is not None:
            hist[p % 12] += 1

    total = hist.sum()

    if total > 0:
        hist /= total

    return hist


def histogram_distance(a, b):
    """Total variation distance."""
    return 0.5 * np.abs(a - b).sum()


def average_interval(tokens):
    """Average absolute pitch jump."""
    pitches = [
        token_pitch(token)
        for token in tokens
    ]

    pitches = [
        p for p in pitches
        if p is not None
    ]

    if len(pitches) < 2:
        return 0.0

    intervals = [
        abs(pitches[i] - pitches[i - 1])
        for i in range(1, len(pitches))
    ]

    return float(np.mean(intervals))


def repetition_rate(tokens):
    """Fraction of adjacent tokens that are identical."""
    if len(tokens) < 2:
        return 0.0

    repeats = sum(
        tokens[i] == tokens[i - 1]
        for i in range(1, len(tokens))
    )

    return repeats / (len(tokens) - 1)


def distinct_n(tokens, n=4):
    """Fraction of unique n-grams."""
    if len(tokens) < n:
        return 0.0

    grams = [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]

    return len(set(grams)) / len(grams)


# ------------------------------------------------------------
# Generate baseline outputs
# ------------------------------------------------------------

nb = NaiveBayesLike().fit(D["train"])

nb_out = nb.generate(
    guide[:300]
)


model, vocab = load_model()

lstm_out = generate(
    model,
    vocab,
    guide[:16],
    300,
    1.0
)


# Random baseline
rnd = [
    vocab[i]
    for i in np.random.randint(
        1,
        len(vocab),
        300
    )
]


# ------------------------------------------------------------
# Real test statistics
# ------------------------------------------------------------

real_hist = pitch_class_histogram(real)

print()
print("REAL TEST MUSIC")
print(
    f"average interval = "
    f"{average_interval(real):.2f}"
)
print(
    f"repetition rate  = "
    f"{repetition_rate(real):.3f}"
)
print(
    f"distinct-4       = "
    f"{distinct_n(real, 4):.3f}"
)


# ------------------------------------------------------------
# Generated music metrics
# ------------------------------------------------------------

print()
print("GENERATED MUSIC")
print("-" * 95)

for name, tokens in [
    ("Random tokens", rnd),
    ("Naive Bayes-like", nb_out),
    ("LSTM", lstm_out),
]:

    pc_hist = pitch_class_histogram(tokens)

    distance = histogram_distance(
        real_hist,
        pc_hist
    )

    interval = average_interval(tokens)
    repetition = repetition_rate(tokens)
    d4 = distinct_n(tokens, 4)

    print(
        f"{name:18s} "
        f"pitch-distance={distance:.3f}  "
        f"avg-interval={interval:.2f}  "
        f"repetition={repetition:.3f}  "
        f"distinct-4={d4:.3f}"
    )


# ------------------------------------------------------------
# Next-token accuracy
# ------------------------------------------------------------

print()
print("NEXT-TOKEN ACCURACY")

# Naive Bayes accuracy
nb_acc = nb.accuracy(D["test"])

print(
    f"Naive Bayes-like test: "
    f"{nb_acc:.3f}"
)


# ------------------------------------------------------------
# LSTM test accuracy
# ------------------------------------------------------------

tok2id = D["tok2id"]

model.eval()

correct = 0
total = 0

try:

    with torch.no_grad():

        for tokens in D["test"].values():

            ids = [
                tok2id.get(token, 0)
                for token in tokens
            ]

            if len(ids) < 2:
                continue

            x = torch.tensor(
                ids[:-1],
                dtype=torch.long
            ).unsqueeze(0)

            y = torch.tensor(
                ids[1:],
                dtype=torch.long
            )

            result = model(x)

            # The model returns (logits, hidden_state)
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result

            predictions = logits.argmax(
                dim=-1
            ).squeeze(0)

            correct += (
                predictions == y
            ).sum().item()

            total += len(y)

    if total > 0:

        lstm_acc = correct / total

        print(
            f"LSTM test:          "
            f"{lstm_acc:.3f}"
        )

    else:
        print(
            "LSTM test: no valid "
            "test sequences"
        )

except Exception as e:

    print(
        "Could not calculate "
        f"LSTM test accuracy: {e}"
    )


print()
print("Evaluation complete.")