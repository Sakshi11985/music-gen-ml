# Generating Music with Machine Learning

# Music Generation with Machine Learning

A comparative study of machine-learning models for automatic music generation using classical MIDI compositions.

## Research Statement

**How effectively can different machine-learning models learn musical sequence patterns from classical music and generate new music that preserves the statistical and structural characteristics of the original compositions?**

## Objective

To compare different machine-learning approaches for generating new musical sequences from classical MIDI data and evaluate the generated music.

## Background

This project builds upon the 2018 Stanford CS229 project **"Music Composition with Machine Learning"**.

The original project explored:
- Naive Bayes
- LSTM RNN
- Encoder-Decoder models

Our project extends the approach by comparing multiple models and using additional evaluation metrics.

## Dataset

- 64 classical compositions
- Bach, Beethoven, Mozart and Haydn
- `music21` corpus
- ~39,000 note events
- MIDI-based representation
- Each event contains pitch and duration

## Models

- **LSTM** — 1 layer, 256 hidden units
- **Vanilla Neural Network (MLP)** — previous 32 notes
- **Bigram Markov Model** — uses the previous note
- **Naive Bayes-like Model** — uses bass/chord information

## Pipeline

```text
Classical MIDI
      ↓
Preprocessing
      ↓
Note/Event Tokens
      ↓
Train Models
      ↓
Next-Note Prediction
      ↓
Music Generation
      ↓
Evaluation
