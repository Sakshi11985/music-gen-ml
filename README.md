# Music Generation with Machine Learning

UE24CS352A Machine Learning mini-project. Based on the Stanford CS229 (2018) project
*Generating Music with Machine Learning* (Kang, Kim, Ringdahl).

The models read a melody note by note and predict the next note. Repeating that prediction
composes a new melody. We compare one neural sequence model (LSTM) with a plain neural
network (MLP) and three simple statistical baselines, all trained on the same 64 classical pieces.

## Models

| Model | What it predicts | Context it sees | File |
|---|---|---|---|
| **LSTM** (main model) | next melody token | every earlier note (recurrent memory); 1 layer, 256 hidden units | `src/model.py`, `src/train.py` |
| **Vanilla NN (MLP)** | next melody token | the previous 32 notes at once; 1 hidden layer of 1024 units | `src/mlp.py` |
| **Bigram Markov** | next melody token | the current note only | `src/baseline_nb.py` |
| **Naive Bayes-like** | melody (top) note of an event | the bass (lowest) note of the same event | `src/baseline_nb.py` |
| **Unigram** (reference) | next melody token | nothing, just overall note frequencies | `src/compare_models.py` |

## Data and tokens

* **Source:** 64 pieces exported from music21's built-in corpus by `export_corpus.py`
  (40 Bach chorales, 9 Beethoven, 8 Mozart, 7 Haydn; the non-Bach pieces are mostly string-quartet
  movements, so this is not piano-only music). No download is needed.
* **Preprocessing** (`src/preprocess.py`): every piece is transposed to C major / A minor, then
  *chordified*, so each moment where the sounding notes change becomes one event.
  Durations are rounded to eighth notes and clipped to 0.5-4 beats.
* **Tokens:**
  * full event: `48.64.67_1.0` (MIDI pitches 48, 64, 67, lasting 1 beat), used by Naive Bayes;
  * melody token: `67_1.0` (only the highest pitch), used by the LSTM, MLP and bigram.
* **Vocabulary:** 209 melody tokens (tokens seen fewer than 2 times in training become `<unk>`;
  1.5% of test tokens are `<unk>`). In total 38,989 note events.
* **Split:** each piece is split *by position*: its first 80% is training, the next 10% validation,
  the last 10% test. So the test set contains unseen continuations of pieces the models were trained
  on, not unseen pieces (see Limitations).

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/Sakshi11985/music-gen-ml.git
cd music-gen-ml
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pip install torch` is slow or fails, install the CPU-only build from
[pytorch.org](https://pytorch.org/get-started/locally/) first, then run the line above again.

Tested with Python 3.13, torch 2.14.1 (CPU), music21 10.5.0, streamlit 1.65.0, numpy 2.5.3,
matplotlib 3.11.2.

## Run

Run the commands from the repo folder (scripts also work from inside `src/`).

```bash
python export_corpus.py            # 1. build data/midi/ (64 .mid files, about 3 min)
python src/preprocess.py           # 2. tokens + 80/10/10 split -> data/processed.pkl (about 3 min)
python src/baseline_nb.py          # 3. Naive Bayes + bigram scores, sample MIDI files
python src/train.py --epochs 30    # 4. train the LSTM (about 1-2 min on CPU)
python src/mlp.py --epochs 30      # 5. train the MLP
python src/generate.py --seed 0    # 6. LSTM sample -> outputs/lstm_sample.mid
python src/compare_models.py       # 7. all models on the same test notes -> outputs/results.json
python src/evaluate.py             # 8. statistics of generated music -> outputs/generation_metrics.json
streamlit run app.py               # 9. interactive demo in the browser
```

Trained checkpoints (`checkpoints/lstm.pt`, `checkpoints/mlp.pt`) are included, so after steps 1-2
you can skip training and go straight to steps 6-9.

The `data/` folder is not in the repo. If the app says `data/processed.pkl` not found, run steps 1 and 2.

Open `.mid` files in MuseScore, an online MIDI player, or VLC with a soundfont.
The app plays audio directly with a built-in synthesiser.

## Results

### Next-note prediction (`src/compare_models.py`)
All models predict the same 3,806 held-out melody notes (`<unk>` targets excluded).
Perplexity = exp(mean cross-entropy in nats); lower is better. Accuracy = top-1 next-note accuracy.

| Model | Test perplexity | Next-note accuracy |
|---|---|---|
| Unigram (no context) | 49.93 | 6.5% |
| Bigram Markov | 16.28 | 33.0% |
| Vanilla NN (MLP) | 15.98 | 31.8% |
| **LSTM (256 units)** | **12.97** | **33.6%** |

Naive Bayes-like (predicts the melody note of an event from its bass note): 13.1% top-1 test accuracy.
It answers a different question, so it is not in the table.

**Run-to-run variation:** retraining the LSTM with seeds 0, 1 and 2 gave test perplexity
12.97, 13.20 and 13.45 (accuracy 33.6-35.5%). The LSTM beats the bigram and MLP in every run,
but differences of about 0.5 perplexity between LSTM runs are noise.

The LSTM overfits quickly: validation loss is lowest around epoch 6-9 and then rises while
training loss keeps falling (`outputs/lstm_loss.png`). `train.py` keeps the checkpoint with the
lowest validation loss.

### Statistics of generated music (`src/evaluate.py`)
Each model generates 100 note events for each of the 64 test excerpts (seed notes are not counted).

| Source | Pitch-class distance to real music | Avg jump (semitones) | Distinct 4-note patterns |
|---|---|---|---|
| Real music (test) | 0.000 | 2.90 | 0.574 |
| Random notes | 0.174 | 16.37 | 0.999 |
| Naive Bayes-like | 0.046 | 9.36 | 0.994 |
| Bigram Markov | 0.079 | 2.34 | 0.614 |
| LSTM | 0.129 | 2.46 | 0.559 |
| Vanilla NN (MLP) | 0.133 | 3.00 | 0.559 |

* *Pitch-class distance:* total-variation distance between 12-bin pitch-class histograms (0 to 1).
* *Avg jump:* mean absolute interval between consecutive melody notes.
* *Distinct 4-note patterns:* share of 4-note sequences that are unique (low = repetitive).

How to read it: the sequence models (bigram, LSTM, MLP) move in small steps like real music, while
Naive Bayes jumps about 9 semitones per note because it has no memory of the previous note. Naive
Bayes still has the best pitch-class distance, because it keeps the real bass line; a single metric
does not capture musical quality. These numbers come from one sampling seed.

## Project structure

```
export_corpus.py        build data/midi/ from the music21 corpus
app.py                  Streamlit demo (generate, compare models, results, explanation)
src/
  utils.py              repo paths, token <-> pitches/duration, MIDI writing
  preprocess.py         MIDI -> tokens, vocabulary, 80/10/10 split
  baseline_nb.py        Naive Bayes-like and bigram Markov models
  model.py              LSTM network
  train.py              LSTM training and evaluation
  generate.py           LSTM sampling with temperature
  mlp.py                MLP model, training and sampling
  compare_models.py     fair next-note comparison of all models
  evaluate.py           statistics of generated music
checkpoints/            trained LSTM and MLP weights
outputs/                results JSON, loss plots, sample MIDI files
```

## Limitations

* Small dataset (64 pieces, about 39k events) and mixed instrumentation, not piano-only.
* Test excerpts come from the same pieces as the training data, so scores measure continuing a
  known piece, not generalising to new pieces. A by-piece split would be a fairer test.
* Only the top melody line is modelled by the sequence models; harmony is dropped.
* The LSTM overfits after a few epochs; dropout or a smaller network might help.
* No listening study; quality is judged only by the statistics above.
* The encoder-decoder model from the Stanford paper is not implemented.

## Reference

M. Kang, J. Kim, E. Ringdahl, *Generating Music with Machine Learning*, Stanford CS229 project, 2018.
[Report](https://cs229.stanford.edu/proj2018/report/18.pdf)

## Team

| Name | SRN |
|---|---|
| Sakshi P Shetty | PES1UG24AM395 |
| Vallabha Anand Bhilare | PES1UG24AM414 |

Section G, team G-25. Allotted problem statement 22: *Generating music with Machine Learning*
([report](https://cs229.stanford.edu/proj2018/report/18.pdf), [poster](https://cs229.stanford.edu/proj2018/poster/18.pdf)).
