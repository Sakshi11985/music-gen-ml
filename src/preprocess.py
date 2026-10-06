"""Parse MIDI files into tokens and split into train/val/test.

Usage:  python src/preprocess.py --midi_dir data/midi --out data/processed.pkl

Every piece is (1) transposed to C major / A minor, (2) 'chordified' so each new note event
becomes one FULL token 'pitches_duration', e.g. '48.64.67_1.0'.
From the full tokens we derive:
  * MELODY tokens ('top-pitch_duration', e.g. '67_1.0')  -> used by the LSTM and bigram baseline
  * FULL tokens (kept as *_full)                          -> used by the Naive Bayes-like model
"""
import argparse, glob, os, pickle
from collections import Counter
from music21 import converter, chord
from utils import make_token, parse_token


def midi_to_tokens(path, normalize=True):
    score = converter.parse(path)
    if normalize:                       # transpose so every piece is in C major / A minor
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
        dur = round(float(c.quarterLength) * 2) / 2      # quantise to eighth notes
        dur = min(max(dur, 0.5), 4.0)
        toks.append(make_token([p.midi for p in c.pitches], dur))
    return toks


def to_melody(tok):
    pitches, dur = parse_token(tok)
    return f"{pitches[-1]}_{dur}"        # highest pitch = melody line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", default="data/midi")
    ap.add_argument("--out", default="data/processed.pkl")
    ap.add_argument("--min_freq", type=int, default=2, help="rarer melody tokens become <unk>")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.midi_dir, "**", "*.mid*"), recursive=True))
    if a.max_files:
        files = files[: a.max_files]
    assert files, f"No MIDI files found in {a.midi_dir}"
    print(f"Parsing {len(files)} files ...")

    full = {}
    for i, f in enumerate(files, 1):
        try:
            t = midi_to_tokens(f)
            t = [x for x in t if parse_token(x)[0]]          # drop empty events
            if len(t) > 20:
                full[os.path.basename(f)] = t
        except Exception as e:
            print("  skipped", f, "->", e)
        if i % 10 == 0:
            print(f"  {i}/{len(files)}")

    # split every song by position 80/10/10 (works even with very few songs)
    sp = {"train": {}, "val": {}, "test": {}}
    for name, t in full.items():
        n = len(t)
        sp["train"][name], sp["val"][name], sp["test"][name] = t[: int(.8 * n)], t[int(.8 * n): int(.9 * n)], t[int(.9 * n):]

    out = {}
    for s, songs in sp.items():
        out[s + "_full"] = songs
        out[s] = {k: [to_melody(x) for x in v] for k, v in songs.items()}

    cnt = Counter(tok for t in out["train"].values() for tok in t)
    vocab = ["<unk>"] + sorted(tok for tok, c in cnt.items() if c >= a.min_freq)
    out["vocab"] = vocab
    out["tok2id"] = {t: i for i, t in enumerate(vocab)}

    n_notes = sum(len(t) for t in full.values())
    test_toks = [x for t in out["test"].values() for x in t]
    unk = sum(x not in out["tok2id"] for x in test_toks) / max(len(test_toks), 1)
    print(f"{len(full)} songs, {n_notes} note events, melody vocab size {len(vocab)}, test <unk> rate {unk:.3f}")

    with open(a.out, "wb") as f:
        pickle.dump(out, f)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
