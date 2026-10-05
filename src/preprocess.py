"""Parse MIDI files into normalized melody note-event tokens.

Pipeline:
1. Detect the key.
2. Transpose major pieces to C major and minor pieces to A minor.
3. Chordify the score to obtain time-aligned events.
4. Keep only the highest pitch from each event as the melody note.
5. Quantize duration to 16th-note steps.
6. Build a vocabulary from training data.

Usage:
    python src/preprocess.py --midi_dir data/midi --out data/processed.pkl
"""

import argparse
import glob
import os
import pickle
from collections import Counter

from music21 import converter, chord, interval, pitch
from utils import make_token


def normalize_key(score):
    """Transpose major keys to C major and minor keys to A minor."""
    try:
        detected_key = score.analyze("key")

        if detected_key.mode == "minor":
            target = pitch.Pitch("A")
        else:
            target = pitch.Pitch("C")

        transposition = interval.Interval(detected_key.tonic, target)
        return score.transpose(transposition)

    except Exception:
        # If key detection fails, leave the score unchanged.
        return score


def midi_to_tokens(path):
    """Convert one MIDI file into melody pitch-duration tokens."""
    score = converter.parse(path)

    # Normalize the key before extracting melody.
    score = normalize_key(score)

    # Align simultaneous notes/chords.
    chords = (
        score.chordify()
        .recurse()
        .getElementsByClass(chord.Chord)
    )

    tokens = []

    for c in chords:
        if not c.pitches:
            continue

        # Use the highest pitch as the melody note.
        melody_pitch = max(c.pitches, key=lambda p: p.midi)

        # Quantize duration to 16th-note units.
        dur = round(float(c.quarterLength) * 4) / 4

        # Keep durations in a reasonable range.
        dur = min(max(dur, 0.25), 4.0)

        tokens.append(
            make_token([melody_pitch.midi], dur)
        )

    return tokens


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
        help="Tokens occurring fewer than this become <unk>"
    )

    ap.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="0 = use all files"
    )

    args = ap.parse_args()

    files = sorted(
        glob.glob(
            os.path.join(
                args.midi_dir,
                "**",
                "*.mid*"
            ),
            recursive=True
        )
    )

    if args.max_files:
        files = files[:args.max_files]

    assert files, f"No MIDI files found in {args.midi_dir}"

    print(f"Parsing {len(files)} files ...")

    songs = {}

    for i, filepath in enumerate(files, 1):
        try:
            tokens = midi_to_tokens(filepath)

            if len(tokens) > 20:
                songs[os.path.basename(filepath)] = tokens

        except Exception as e:
            print("  skipped", filepath, "->", e)

        if i % 10 == 0:
            print(f"  {i}/{len(files)}")

    # Split each song chronologically:
    # 80% train, 10% validation, 10% test.
    train = {}
    val = {}
    test = {}

    for name, tokens in songs.items():
        n = len(tokens)

        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        train[name] = tokens[:train_end]
        val[name] = tokens[train_end:val_end]
        test[name] = tokens[val_end:]

    # Build vocabulary ONLY from training data.
    counts = Counter(
        token
        for tokens in train.values()
        for token in tokens
    )

    vocab = ["<unk>"] + sorted(
        token
        for token, count in counts.items()
        if count >= args.min_freq
    )

    tok2id = {
        token: idx
        for idx, token in enumerate(vocab)
    }

    n_events = sum(
        len(tokens)
        for tokens in songs.values()
    )

    print(
        f"{len(songs)} songs, "
        f"{n_events} melody events, "
        f"vocab size {len(vocab)}"
    )

    print("Most common tokens:")

    for token, count in counts.most_common(10):
        print(f"  {token}: {count}")

    os.makedirs(
        os.path.dirname(args.out) or ".",
        exist_ok=True
    )

    with open(args.out, "wb") as f:
        pickle.dump(
            dict(
                train=train,
                val=val,
                test=test,
                vocab=vocab,
                tok2id=tok2id
            ),
            f
        )

    print("Saved", args.out)


if __name__ == "__main__":
    main()