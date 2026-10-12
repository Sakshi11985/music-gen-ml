import os
import csv
from music21 import corpus, converter

os.makedirs("data/midi", exist_ok=True)

plan = {
    "bach": 40,
    "beethoven": 10,
    "mozart": 10,
    "haydn": 10,
}
metadata = []
total = 0
for composer, limit in plan.items():
    paths = list(corpus.getComposer(composer))[:limit]
    ok = 0

    for i, p in enumerate(paths):
        out = os.path.join(
            "data",
            "midi",
            f"{composer}_{i:03d}.mid"
        )

        try:
            converter.parse(p).write("midi", fp=out)

            metadata.append({
                "composer": composer,
                "source": str(p),
                "midi_file": os.path.basename(out)
            })

            ok += 1

        except Exception as e:
            print(
                "  skipped",
                os.path.basename(str(p)),
                "->",
                e
            )
    print(composer, ok, "files written")
    total += ok

metadata_path = os.path.join("data", "metadata.csv")

with open(metadata_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["composer", "source", "midi_file"]
    )
    writer.writeheader()
    writer.writerows(metadata)

print("Total MIDI files:", total)
print("Metadata saved to:", metadata_path)
