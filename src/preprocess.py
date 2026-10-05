"""Parse MIDI files into note-event tokens and split into train/val/test.

Usage:  python src/preprocess.py --midi_dir data/midi --out data/processed.pkl

Each 'chordified' time step (a new note event) becomes one token 'pitches_duration',
e.g. '48.64.67_1.0'  (matches the 'one vector per note event' idea of the paper).
"""
import argparse, glob, os, pickle
from collections import Counter
from music21 import converter, chord
from utils import make_token


def midi_to_tokens(path):
    score = converter.parse(path)
    chords = score.chordify().recurse().getElementsByClass(chord.Chord)
    toks = []
    for c in chords:
        dur = round(float(c.quarterLength) * 4) / 4      # quantise to 16th notes
        dur = min(max(dur, 0.25), 4.0)
        toks.append(make_token([p.midi for p in c.pitches], dur))
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", default="data/midi")
    ap.add_argument("--out", default="data/processed.pkl")
    ap.add_argument("--min_freq", type=int, default=2, help="rarer tokens become <unk>")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all files")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.midi_dir, "**", "*.mid*"), recursive=True))
    if a.max_files:
        files = files[: a.max_files]
    assert files, f"No MIDI files found in {a.midi_dir}"
    print(f"Parsing {len(files)} files ...")

    songs = {}
    for i, f in enumerate(files, 1):
        try:
            t = midi_to_tokens(f)
            if len(t) > 20:
                songs[os.path.basename(f)] = t
        except Exception as e:
            print("  skipped", f, "->", e)
        if i % 10 == 0:
            print(f"  {i}/{len(files)}")

    # Split every song by position 80/10/10 (works even with very few songs)
    train, val, test = {}, {}, {}
    for name, t in songs.items():
        n = len(t)
        train[name], val[name], test[name] = t[: int(.8 * n)], t[int(.8 * n): int(.9 * n)], t[int(.9 * n):]

    cnt = Counter(tok for t in train.values() for tok in t)
    vocab = ["<unk>"] + sorted(tok for tok, c in cnt.items() if c >= a.min_freq)
    tok2id = {t: i for i, t in enumerate(vocab)}
    n_notes = sum(len(t) for t in songs.values())
    print(f"{len(songs)} songs, {n_notes} note events, vocab size {len(vocab)}")

    with open(a.out, "wb") as f:
        pickle.dump(dict(train=train, val=val, test=test, vocab=vocab, tok2id=tok2id), f)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
