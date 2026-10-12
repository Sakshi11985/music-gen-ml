"""Build the dataset: export pieces from music21's built-in corpus as MIDI files into data/midi/.

Usage:  python export_corpus.py          (no download needed; takes a few minutes)

Bach = 4-part chorales, Beethoven / Mozart / Haydn = mostly string-quartet movements.
Files that music21 cannot expand (badly formed repeats) are skipped, so you get 64 files.
"""
import os
from music21 import corpus, converter

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "data", "midi")
PLAN = {"bach": 40, "beethoven": 10, "mozart": 10, "haydn": 10}   # composer -> max files tried


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for composer, limit in PLAN.items():
        paths = list(corpus.getComposer(composer))[:limit]
        ok = 0
        for i, p in enumerate(paths):
            out = os.path.join(OUT_DIR, f"{composer}_{i:03d}.mid")
            try:
                converter.parse(p).write("midi", fp=out)
                ok += 1
            except Exception as e:
                print("  skipped", os.path.basename(str(p)), "->", e)
        print(composer, ok, "files written")
        total += ok
    print("Total MIDI files:", total, "in", OUT_DIR)


if __name__ == "__main__":
    main()
