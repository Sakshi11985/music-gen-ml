"""Shared helpers: project paths, token <-> pitches/duration, MIDI writing."""
import os
from music21 import stream, chord, note, tempo

# Repo root, so scripts work no matter which folder you run them from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(*parts):
    """Absolute path inside the repo, e.g. path('data', 'processed.pkl')."""
    return os.path.join(ROOT, *parts)


def ensure_parent(file_path):
    """Create the folder a file will be written into (data/, outputs/, checkpoints/)."""
    d = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(d, exist_ok=True)
    return file_path


def make_token(pitches, dur):
    return ".".join(str(p) for p in sorted(pitches)) + "_" + str(dur)


def parse_token(tok):
    """'48.64_1.0' -> ((48, 64), 1.0).  Special tokens ('<unk>') -> ((), 0.5)."""
    if "_" not in tok:
        return (), 0.5
    p, d = tok.rsplit("_", 1)
    pitches = tuple(int(x) for x in p.split(".") if x != "")
    return pitches, float(d)


def tokens_to_midi(tokens, out_path, bpm=100):
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=bpm))
    for tok in tokens:
        pitches, dur = parse_token(tok)
        if not pitches:
            s.append(note.Rest(quarterLength=dur))
        elif len(pitches) == 1:
            s.append(note.Note(pitches[0], quarterLength=dur))
        else:
            s.append(chord.Chord(list(pitches), quarterLength=dur))
    s.write("midi", fp=ensure_parent(out_path))
    return out_path
