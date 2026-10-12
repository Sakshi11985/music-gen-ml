"""Parse MIDI files into tokens and split into train/val/test.

Usage:  python src/preprocess.py --midi_dir data/midi --out data/processed.pkl

Every piece is:
(1) transposed to C major / A minor,
(2) 'chordified' so each new note event becomes one FULL token
    'pitches_duration', e.g. '48.64.67_1.0'.

From the full tokens we derive:
  * MELODY tokens ('top-pitch_duration', e.g. '67_1.0')
      -> used by the LSTM, MLP and bigram baseline
  * FULL tokens (kept as *_full)
      -> used by the Naive Bayes-like model
"""

import argparse
import glob
import os
import pickle
import random
from collections import Counter

from music21 import converter, chord
from utils import make_token, parse_token


def midi_to_tokens(path, normalize=True):
    score = converter.parse(path)

    if normalize:
        # Transpose so every piece is in C major / A minor.
        try:
            k = score.analyze("key")
            target = 0 if k.mode == "major" else 9

            semis = (target - k.tonic.pitchClass) % 12

            if semis > 6:
                semis -= 12

            score = score.transpose(semis)

        except Exception:
            pass

    chords = score.chordify().recurse().getElementsByClass(chord.Chord)

    toks = []

    for c in chords:
        dur = round(float(c.quarterLength) * 2) / 2

        # Quantise duration to eighth notes and limit range.
        dur = min(max(dur, 0.5), 4.0)

        toks.append(
            make_token(
                [p.midi for p in c.pitches],
                dur
            )
        )

    return toks


def to_melody(tok):
    pitches, dur = parse_token(tok)

    # Highest pitch = melody line.
    return f"{pitches[-1]}_{dur}"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--midi_dir",
        default="data/midi"
    )

    ap.add_argument(
        "--out",
        default="data/processed.pkl"
    )

    ap.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="rarer melody tokens become <unk>"
    )

    ap.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="0 = all files"
    )

    a = ap.parse_args()

    files = sorted(
        glob.glob(
            os.path.join(
                a.midi_dir,
                "**",
                "*.mid*"
            ),
            recursive=True
        )
    )

    if a.max_files:
        files = files[:a.max_files]

    assert files, f"No MIDI files found in {a.midi_dir}"

    print(f"Parsing {len(files)} files ...")

    # ---------------------------------------------------------
    # Parse every MIDI file
    # ---------------------------------------------------------

    full = {}

    for i, f in enumerate(files, 1):

        try:
            t = midi_to_tokens(f)

            # Drop empty events.
            t = [
                x for x in t
                if parse_token(x)[0]
            ]

            # Keep only sufficiently long pieces.
            if len(t) > 20:
                full[os.path.basename(f)] = t

        except Exception as e:
            print(
                "  skipped",
                f,
                "->",
                e
            )

        if i % 10 == 0:
            print(
                f"  {i}/{len(files)}"
            )

    # ---------------------------------------------------------
    # SONG-LEVEL TRAIN / VALIDATION / TEST SPLIT
    #
    # A complete composition belongs to only ONE split.
    # This prevents the same composition from appearing in
    # both training and testing data.
    # ---------------------------------------------------------

    names = list(full.keys())

    # Fixed seed makes the split reproducible.
    random.Random(42).shuffle(names)

    n_files = len(names)

    n_train = int(0.8 * n_files)
    n_val = int(0.1 * n_files)

    train_names = names[:n_train]

    val_names = names[
        n_train:n_train + n_val
    ]

    test_names = names[
        n_train + n_val:
    ]

    sp = {
        "train": {
            name: full[name]
            for name in train_names
        },

        "val": {
            name: full[name]
            for name in val_names
        },

        "test": {
            name: full[name]
            for name in test_names
        }
    }

    print(
        f"Song split: "
        f"train {len(train_names)}, "
        f"val {len(val_names)}, "
        f"test {len(test_names)}"
    )

    # ---------------------------------------------------------
    # Create full-token and melody-token datasets
    # ---------------------------------------------------------

    out = {}

    for s, songs in sp.items():

        # Full chord/event tokens.
        out[s + "_full"] = songs

        # Melody tokens.
        out[s] = {
            k: [
                to_melody(x)
                for x in v
            ]
            for k, v in songs.items()
        }

    # ---------------------------------------------------------
    # Build vocabulary ONLY from training melodies
    # ---------------------------------------------------------

    cnt = Counter(
        tok
        for t in out["train"].values()
        for tok in t
    )

    vocab = [
        "<unk>"
    ] + sorted(
        tok
        for tok, c in cnt.items()
        if c >= a.min_freq
    )

    out["vocab"] = vocab

    out["tok2id"] = {
        t: i
        for i, t in enumerate(vocab)
    }

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    n_notes = sum(
        len(t)
        for t in full.values()
    )

    test_toks = [
        x
        for t in out["test"].values()
        for x in t
    ]

    unk = (
        sum(
            x not in out["tok2id"]
            for x in test_toks
        )
        / max(len(test_toks), 1)
    )

    print(
        f"{len(full)} songs, "
        f"{n_notes} note events, "
        f"melody vocab size {len(vocab)}, "
        f"test <unk> rate {unk:.3f}"
    )

    # ---------------------------------------------------------
    # Save processed dataset
    # ---------------------------------------------------------

    with open(a.out, "wb") as f:
        pickle.dump(out, f)

    print(
        "Saved",
        a.out
    )


if __name__ == "__main__":
    main()
