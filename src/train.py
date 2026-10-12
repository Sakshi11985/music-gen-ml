"""Train the LSTM.

Usage:
    python src/train.py --data data/processed.pkl --epochs 30
"""

import argparse
import math
import os
import pickle
import random

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import MusicLSTM


def set_seed(seed):
    """Make training reproducible."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_windows(songs, tok2id, seq_len):
    X, Y = [], []

    for toks in songs.values():

        ids = [
            tok2id.get(t, 0)
            for t in toks
        ]

        for i in range(
            0,
            len(ids) - seq_len - 1,
            max(seq_len // 2, 1)
        ):
            X.append(
                ids[i:i + seq_len]
            )

            Y.append(
                ids[i + 1:i + seq_len + 1]
            )

    if not X:
        return (
            torch.empty((0, seq_len), dtype=torch.long),
            torch.empty((0, seq_len), dtype=torch.long)
        )

    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long)
    )


def run_epoch(model, X, Y, loss_fn, opt, bs, dev):
    train = opt is not None

    model.train(train)

    if len(X) == 0:
        return float("nan"), float("nan")

    if train:
        perm = torch.randperm(len(X))
    else:
        perm = torch.arange(len(X))

    tot_loss = 0.0
    tot_hit = 0
    tot_n = 0

    with torch.set_grad_enabled(train):

        for i in range(0, len(X), bs):

            idx = perm[i:i + bs]

            x = X[idx].to(dev)
            y = Y[idx].to(dev)

            logits, _ = model(x)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            if train:
                opt.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )

                opt.step()

            n = y.numel()

            tot_loss += loss.item() * n
            tot_n += n

            tot_hit += (
                logits.argmax(-1) == y
            ).sum().item()

    return (
        tot_loss / tot_n,
        tot_hit / tot_n
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data",
        default="data/processed.pkl"
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    ap.add_argument(
        "--seq_len",
        type=int,
        default=50
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=64
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=0.005
    )

    ap.add_argument(
        "--hidden",
        type=int,
        default=256
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    ap.add_argument(
        "--ckpt",
        default="checkpoints/lstm.pt"
    )

    a = ap.parse_args()

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------

    set_seed(a.seed)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    dev = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ---------------------------------------------------------
    # Load processed dataset
    # ---------------------------------------------------------

    with open(a.data, "rb") as f:
        D = pickle.load(f)

    # ---------------------------------------------------------
    # Create train / validation / test windows
    # ---------------------------------------------------------

    Xtr, Ytr = make_windows(
        D["train"],
        D["tok2id"],
        a.seq_len
    )

    Xva, Yva = make_windows(
        D["val"],
        D["tok2id"],
        a.seq_len
    )

    Xte, Yte = make_windows(
        D["test"],
        D["tok2id"],
        a.seq_len
    )

    print(
        f"windows  "
        f"train {len(Xtr)}  "
        f"val {len(Xva)}  "
        f"test {len(Xte)}  "
        f"| vocab {len(D['vocab'])} "
        f"| device {dev}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = MusicLSTM(
        len(D["vocab"]),
        hidden=a.hidden
    ).to(dev)

    # ---------------------------------------------------------
    # Optimizer and loss
    # ---------------------------------------------------------

    opt = torch.optim.Adam(
        model.parameters(),
        lr=a.lr,
        betas=(0.9, 0.999)
    )

    loss_fn = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    tr_hist = []
    va_hist = []

    best = float("inf")

    os.makedirs(
        os.path.dirname(a.ckpt) or ".",
        exist_ok=True
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    for ep in range(
        1,
        a.epochs + 1
    ):

        trl, tra = run_epoch(
            model,
            Xtr,
            Ytr,
            loss_fn,
            opt,
            a.batch,
            dev
        )

        val, vaa = run_epoch(
            model,
            Xva,
            Yva,
            loss_fn,
            None,
            a.batch,
            dev
        )

        tr_hist.append(trl)
        va_hist.append(val)

        print(
            f"epoch {ep:3d} "
            f"| train loss {trl:.4f} "
            f"acc {tra:.3f} "
            f"| val loss {val:.4f} "
            f"acc {vaa:.3f}"
        )

        # Save the best validation checkpoint.
        if val < best:

            best = val

            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab": D["vocab"],
                    "hidden": a.hidden,
                    "seq_len": a.seq_len,
                    "seed": a.seed,
                },
                a.ckpt
            )

    # ---------------------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------------------

    ck = torch.load(
        a.ckpt,
        map_location=dev
    )

    model.load_state_dict(
        ck["model"]
    )

    # ---------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------

    tel, tea = run_epoch(
        model,
        Xte,
        Yte,
        loss_fn,
        None,
        a.batch,
        dev
    )

    print(
        f"\nTEST  "
        f"loss {tel:.4f}  "
        f"perplexity {math.exp(tel):.2f}  "
        f"next-note accuracy {tea:.3f}"
    )

    # ---------------------------------------------------------
    # Training curve
    # ---------------------------------------------------------

    plt.figure()

    plt.plot(
        tr_hist,
        label="train"
    )

    plt.plot(
        va_hist,
        label="val"
    )

    plt.xlabel("epoch")
    plt.ylabel("cross-entropy")
    plt.legend()
    plt.title("LSTM loss")

    plt.savefig(
        "outputs/lstm_loss.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved outputs/lstm_loss.png"
    )


if __name__ == "__main__":
    main()
