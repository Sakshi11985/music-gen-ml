"""Shared helpers: token <-> pitches/duration, MIDI writing."""
from music21 import stream, chord, note, tempo

SPLIT_PITCH = 60  # MIDI 60 = middle C. Below = "left hand", at/above = "right hand"


def make_token(pitches, dur):
    return ".".join(str(p) for p in sorted(pitches)) + "_" + str(dur)


def parse_token(tok):
    """'48.64_1.0' -> ((48, 64), 1.0).  Special tokens ('<unk>') -> ((), 0.5)."""
    if "_" not in tok:
        return (), 0.5
    p, d = tok.rsplit("_", 1)
    pitches = tuple(int(x) for x in p.split(".") if x != "")
    return pitches, float(d)


def tokens_to_midi(tokens, path, bpm=100):
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
    s.write("midi", fp=path)
    return path


def pitch_histogram(tokens):
    """Normalised histogram over 128 MIDI pitches."""
    import numpy as np
    h = np.zeros(128)
    for t in tokens:
        for p in parse_token(t)[0]:
            h[p] += 1
    return h / max(h.sum(), 1)
