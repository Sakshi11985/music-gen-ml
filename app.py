"""Live demo.  Run:  streamlit run app.py"""
import base64, pickle, random, streamlit as st, streamlit.components.v1 as components
from src_path import *  # noqa  (adds src/ to sys.path)
from baseline_nb import NaiveBayesLike
from utils import tokens_to_midi

st.set_page_config(page_title="ML Music Generator", page_icon="🎵")
st.title("🎵 Generating Music with Machine Learning")
D = pickle.load(open("data/processed.pkl", "rb"))
songs = list(D["test"].keys())

model_name = st.radio("Model", ["LSTM", "Naive Bayes-like"], horizontal=True)
song = st.selectbox("Seed / guide song (from held-out test split)", songs)
length = st.slider("Length (note events)", 50, 400, 150)
temp = st.slider("Temperature (LSTM only)", 0.3, 1.5, 1.0, 0.1)

if st.button("Generate"):
    guide = D["test"][song]
    if model_name == "LSTM":
        from generate import load_model, generate
        m, vocab = load_model()
        toks = generate(m, vocab, guide[:16], length, temp)
    else:
        toks = NaiveBayesLike().fit(D["train"]).generate(guide[:length])
    path = "outputs/demo.mid"
    tokens_to_midi(toks, path)
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    components.html(f"""
      <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.5.0"></script>
      <midi-player src="data:audio/midi;base64,{b64}" sound-font visualizer="#pr"></midi-player>
      <midi-visualizer type="piano-roll" id="pr"></midi-visualizer>""", height=300)
    st.download_button("Download MIDI", open(path, "rb").read(), "generated.mid")
