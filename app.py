"""Live demo.  Run:  streamlit run app.py"""
import io, json, os, sys, pickle, random, tempfile, wave
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from baseline_nb import NaiveBayesLike, Bigram
from utils import parse_token, tokens_to_midi

st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="wide")

SEED_LEN, SR = 16, 22050
BG, SEED_C, GEN_C, REAL_C = "#0e1117", "#6b7280", "#a78bfa", "#34d399"
MODEL_INFO = {
    "LSTM": "Neural network with memory. Our main model.",
    "Vanilla NN (MLP)": "Neural network that sees the previous 32 notes at once, no memory.",
    "Bigram Markov": "Looks only at the previous note. Simple baseline.",
    "Naive Bayes-like": "Picks a melody note from the bass note only.",
}


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1rem;
       background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 50%, #db2777 100%);}
.hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.hero p {margin: .3rem 0 0 0; color: #f3e8ff; font-size: 1.05rem;}
.card {padding: 1rem 1.2rem; border-radius: 12px; background: #1a1f2b; margin-bottom: .6rem;}
</style>
<div class="hero"><h1>🎵 AI Music Generator</h1>
<p>Machine learning models trained on classical music compose a new melody, one note at a time.</p></div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data / models
@st.cache_resource(show_spinner="Loading data...")
def load_data():
    with open("data/processed.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading the trained LSTM...")
def load_lstm():
    from generate import load_model
    return load_model()


@st.cache_resource(show_spinner="Loading the vanilla network...")
def load_mlp_model():
    from mlp import load_mlp
    return load_mlp()


@st.cache_resource
def get_nb():
    return NaiveBayesLike().fit(load_data()["train_full"])


@st.cache_resource
def get_bigram():
    D = load_data()
    return Bigram(D["vocab"], D["tok2id"]).fit(D["train"])


try:
    D = load_data()
    assert "train_full" in D
except FileNotFoundError:
    st.error("`data/processed.pkl` not found. Run:  `python src\\preprocess.py --midi_dir data\\midi --out data\\processed.pkl`")
    st.stop()
except AssertionError:
    st.error("`data/processed.pkl` is from an older version. Re-run `python src\\preprocess.py --midi_dir data\\midi --out data\\processed.pkl`")
    st.stop()


def piece(song, suffix=""):
    """Whole piece in order (train + val + test parts). suffix '' = melody tokens, '_full' = full chords."""
    return D["train" + suffix][song] + D["val" + suffix][song] + D["test" + suffix][song]


def pretty(name):
    base = name.replace(".mid", "")
    try:
        composer, idx = base.rsplit("_", 1)
        return f"{composer.title()} - piece {int(idx)}"
    except ValueError:
        return base


# ----------------------------------------------------------------- generation
def make_tokens(model, song, length, temp):
    """Returns (tokens, number_of_seed_tokens)."""
    melody = piece(song, "")
    if model == "LSTM":
        from generate import generate
        m, vocab = load_lstm()
        seed = melody[:SEED_LEN]
        return generate(m, vocab, seed, length, temp), len(seed)
    if model == "Vanilla NN (MLP)":
        from mlp import generate_mlp
        m, vocab = load_mlp_model()
        seed = melody[:SEED_LEN]
        return generate_mlp(m, vocab, seed, length, temp), len(seed)
    if model == "Bigram Markov":
        seed = melody[:SEED_LEN]
        return get_bigram().generate(seed, length), len(seed)
    return get_nb().generate(piece(song, "_full")[:length]), 0


def set_seed(value):
    if value:
        random.seed(value); np.random.seed(value)
        try:
            import torch; torch.manual_seed(value)
        except Exception:
            pass


# ----------------------------------------------------------------- audio / plots
def synth_wav(tokens, bpm):
    """Simple built-in synth, so the demo needs no internet and no sound font."""
    spb, chunks = 60.0 / bpm, []
    for t in tokens:
        pitches, dur = parse_token(t)
        n = max(int(dur * spb * SR), 1)
        x = np.arange(n) / SR
        sig = np.zeros(n)
        for p in pitches:
            f = 440.0 * 2 ** ((p - 69) / 12)
            sig += np.sin(2 * np.pi * f * x) + 0.5 * np.sin(4 * np.pi * f * x) + 0.25 * np.sin(6 * np.pi * f * x)
        sig /= 1.75 * max(len(pitches), 1)
        env = np.minimum(1.0, x / 0.01) * np.exp(-2.5 * x) * np.minimum(1.0, (n - np.arange(n)) / (0.02 * SR))
        chunks.append(sig * env)
    audio = np.concatenate(chunks) if chunks else np.zeros(1)
    audio = audio / max(np.abs(audio).max(), 1e-6) * 0.85
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def midi_bytes(tokens, bpm):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "music.mid")
        tokens_to_midi(tokens, path, bpm)
        return open(path, "rb").read()


def piano_roll(tokens, seed_n=0, color=GEN_C):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    x, lo, hi = 0.0, 127, 0
    for i, t in enumerate(tokens):
        pitches, dur = parse_token(t)
        for p in pitches:
            ax.add_patch(Rectangle((x, p - 0.4), dur * 0.92, 0.8, color=SEED_C if i < seed_n else color))
            lo, hi = min(lo, p), max(hi, p)
        x += dur
    ax.set_xlim(0, max(x, 1)); ax.set_ylim(lo - 2, hi + 2)
    ax.set_xlabel("time (beats)", color="#9ca3af"); ax.set_ylabel("pitch", color="#9ca3af")
    ax.tick_params(colors="#9ca3af")
    for s in ax.spines.values():
        s.set_color("#374151")
    if seed_n:
        ax.text(0.01, 0.95, "grey = real seed notes   purple = AI generated", transform=ax.transAxes,
                color="#d1d5db", fontsize=8, va="top")
    fig.tight_layout()
    return fig


def stats(tokens, bpm):
    top = [parse_token(t)[0][-1] for t in tokens if parse_token(t)[0]]
    jumps = np.abs(np.diff(top)) if len(top) > 1 else np.array([0.0])
    return {"notes": len(tokens), "secs": sum(parse_token(t)[1] for t in tokens) * 60 / bpm,
            "jump": float(jumps.mean()), "range": max(top) - min(top)}


def show_result(tokens, seed_n, bpm, key, color=GEN_C, compact=False):
    st.audio(synth_wav(tokens, bpm), format="audio/wav")
    fig = piano_roll(tokens, seed_n, color)
    st.pyplot(fig); plt.close(fig)
    s = stats(tokens, bpm)
    c = st.columns(4)
    c[0].metric("Note events", s["notes"])
    c[1].metric("Length", f"{s['secs']:.0f} s")
    c[2].metric("Avg note jump", f"{s['jump']:.1f}", help="Average step between consecutive notes, in semitones. Real music is about 2.9; random notes about 16.")
    c[3].metric("Pitch range", f"{s['range']} semitones")
    if not compact:
        d = st.columns(2)
        d[0].download_button("⬇ Download MIDI", midi_bytes(tokens, bpm), "generated.mid", key=key + "_mid")
        d[1].download_button("⬇ Download WAV", synth_wav(tokens, bpm), "generated.wav", key=key + "_wav")
    return s


# ----------------------------------------------------------------- sidebar
songs = sorted(D["train"].keys())
labels = {pretty(s): s for s in songs}
with st.sidebar:
    st.header("Settings")
    model = st.radio("Model", list(MODEL_INFO), captions=list(MODEL_INFO.values()))
    song = labels[st.selectbox("Starting piece", list(labels), help="The LSTM and bigram start from its first 16 notes; Naive Bayes follows its bass line.")]
    length = st.slider("Length (note events)", 30, 300, 100)
    temp = st.slider("Creativity (temperature)", 0.3, 1.5, 1.0, 0.1, disabled=model not in ("LSTM", "Vanilla NN (MLP)"),
                     help="Low = safe and repetitive. High = adventurous and messier. Neural networks only.")
    bpm = st.slider("Tempo (BPM)", 60, 180, 100, 10)
    seed_val = st.number_input("Random seed (0 = different every time)", 0, 99999, 0)
    st.caption("Tip: use the same seed to get the same piece again.")

tab1, tab2, tab3, tab4 = st.tabs(["🎹 Generate", "⚖️ Compare models", "📊 Results", "ℹ️ How it works"])

# ----------------------------------------------------------------- tab 1
with tab1:
    st.markdown(f"**{model}** will continue **{pretty(next(k for k, v in labels.items() if v == song))}**. "
                "Press the button, then play the result.")
    if st.button("✨ Generate music", type="primary"):
        set_seed(seed_val)
        try:
            with st.spinner("Composing..."):
                toks, seed_n = make_tokens(model, song, length, temp)
            st.session_state["last"] = (toks, seed_n, model, bpm)
        except Exception as e:
            st.error(f"Could not generate with {model}: {e}")
            if model == "LSTM":
                st.info("Train the model first:  `python src\\train.py --epochs 20`")
            if model == "Vanilla NN (MLP)":
                st.info("Train the model first:  `python src\\mlp.py --epochs 30`")
    if "last" in st.session_state:
        toks, seed_n, m_used, bpm_used = st.session_state["last"]
        st.subheader(f"Result - {m_used}")
        show_result(toks, seed_n, bpm_used, "single")
    else:
        st.markdown('<div class="card">👈 Choose a model and a starting piece in the sidebar, then press <b>Generate music</b>.</div>',
                    unsafe_allow_html=True)

# ----------------------------------------------------------------- tab 2
with tab2:
    st.write("Generate from the same starting piece with every model and listen to the difference.")
    if st.button("⚖️ Compare all models", type="primary"):
        set_seed(seed_val)
        try:
            with st.spinner("Composing with every model..."):
                out = {"Original (real music)": (piece(song, "")[:length], 0, REAL_C)}
                for name in MODEL_INFO:
                    toks, seed_n = make_tokens(name, song, length, temp)
                    out[name] = (toks, seed_n, GEN_C)
            st.session_state["cmp"] = (out, bpm)
        except Exception as e:
            st.error(f"Comparison failed: {e}")
    if "cmp" in st.session_state:
        out, bpm_used = st.session_state["cmp"]
        names = list(out)
        for row in [names[i:i + 2] for i in range(0, len(names), 2)]:
            cols = st.columns(2)
            for col, name in zip(cols, row):
                with col:
                    st.subheader(name)
                    toks, seed_n, color = out[name]
                    show_result(toks, seed_n, bpm_used, "cmp_" + name, color, compact=True)
        st.info("Look at **Avg note jump**: real music is about 2.9. The LSTM and bigram stay close to it, "
                "while Naive Bayes (no memory of earlier notes) jumps around like random notes.")

# ----------------------------------------------------------------- tab 3
with tab3:
    st.subheader("1. How well do the models predict the next note?")
    res = read_json("outputs/results.json")
    if res:
        st.dataframe({"Model": [r["model"] for r in res["rows"]],
                      "Test perplexity (lower is better)": [r["perplexity"] for r in res["rows"]],
                      "Next-note accuracy (%)": [r["accuracy"] for r in res["rows"]]}, hide_index=True)
        st.caption(f"Every model is scored on the same {res['n_targets']} held-out notes. Single run on a small dataset (64 pieces).")
    else:
        st.info("Run `python src\\compare_models.py` to fill this table.")

    st.subheader("2. Does the generated music behave like real music?")
    gm = read_json("outputs/generation_metrics.json")
    if gm:
        st.dataframe({"Source": list(gm), "Avg note jump (semitones)": [v["jump"] for v in gm.values()],
                      "Pitch-class distance (lower = closer)": [v["pc"] for v in gm.values()],
                      "Distinct 4-note patterns": [v["d4"] for v in gm.values()]}, hide_index=True)
        st.caption("Real music jumps about 2-3 semitones between notes. Random notes jump about 16.")
    else:
        st.info("Run `python src\\evaluate.py` to fill this table.")

    for path, cap in [("outputs/lstm_loss.png", "LSTM training curve"), ("outputs/mlp_loss.png", "Vanilla NN training curve")]:
        if os.path.exists(path):
            st.subheader(cap)
            st.image(path)
    st.caption("Validation loss rises after a few epochs (overfitting), so the best checkpoint is kept.")

# ----------------------------------------------------------------- tab 4
with tab4:
    st.markdown("""
**Idea.** Music is a sequence, like a sentence. A model reads the notes so far and predicts the next one; repeating this builds a new melody.
Based on the Stanford CS229 (2018) project *Music Composition with Machine Learning*.

**Data.** 64 classical pieces (Bach chorales, Beethoven, Mozart, Haydn) from the music21 corpus, about 39,000 note events.
Every piece is moved to the same key (C major / A minor). Each note event becomes a token such as `67_1.0` (pitch 67, one beat).

**Models**
1. **LSTM** - a neural network with memory of earlier notes (1 layer, 256 units).
2. **Vanilla NN (MLP)** - a plain neural network that sees the previous 32 notes at once (1024 hidden units, as in the Stanford paper).
3. **Bigram Markov** - predicts the next note from the current note only (baseline).
4. **Naive Bayes-like** - picks a melody note from the bass note alone, as in the Stanford paper.

**Creativity (temperature).** The model gives a probability for every possible next note. Low temperature picks the likeliest ones (safe, repetitive); high temperature takes more risks.

**Limitations.** Small dataset, not piano-only, the LSTM overfits, one training run, no listening survey. The encoder-decoder model from the paper is not implemented.
""")
