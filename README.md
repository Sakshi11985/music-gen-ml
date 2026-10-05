# Generating Music with Machine Learning

UE24CS352A Machine Learning mini-project. Based on the Stanford CS229 (2018) project
*Music Composition with Machine Learning* (Kang, Kim, Ringdahl).

We generate piano music from MIDI files with two models:

| Model | Idea |
|---|---|
| **Naive Bayes-like** (baseline) | Learns P(right-hand notes \| left-hand chord) and samples notes for a guide chord sequence |
| **LSTM** (main) | Predicts the next note-event token (pitches + duration); 1 LSTM layer, 256 hidden units |

Each "chordified" note event is one token, e.g. `48.64.67_1.0` (MIDI pitches, duration in quarter notes).

## 1. Setup
```bash
git clone <your-repo-url> && cd music-gen-ml
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Data
Put classical piano `.mid` files in `data/midi/` (subfolders are fine), e.g.
- piano-midi.de (Bach, Beethoven, Mozart, Chopin ...)
- or any public MIDI set. Raw data is not committed (see `.gitignore`).

## 3. Run
```bash
python src/preprocess.py --midi_dir data/midi --out data/processed.pkl   # parse + 80/10/10 split
python src/baseline_nb.py                                                # NB accuracy + outputs/nb_sample.mid
python src/train.py --epochs 30                                          # trains LSTM, prints test loss/perplexity
python src/generate.py --length 200 --temperature 1.0                    # outputs/lstm_sample.mid
python src/evaluate.py                                                   # pitch-histogram comparison
streamlit run app.py                                                     # live demo
```
Outputs: `checkpoints/lstm.pt`, `outputs/lstm_loss.png`, generated `.mid` files in `outputs/`.
Open `.mid` files in MuseScore, VLC (with a soundfont), or an online MIDI player.

## Project structure
```
app.py            Streamlit demo
src/preprocess.py MIDI -> tokens, vocab, split
src/baseline_nb.py  Naive Bayes-like model
src/model.py / train.py / generate.py   LSTM
src/evaluate.py   metrics
```

## Evaluation
Test loss, perplexity, next-note accuracy (LSTM); top-1 accuracy (NB); pitch-histogram
distance between generated and real music for random / NB / LSTM.

## Team
- <Your name, SRN> - <contribution>
- <Teammate name, SRN> - <contribution>
