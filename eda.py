"""EDA for the speech-to-text project.

Usage:
    python eda.py --pred outputs/pred_test-clean_base.en.csv \
        outputs/pred_noisy_snr10_base.en.csv outputs/pred_noisy_snr0_base.en.csv

Outputs (in outputs/): eda.png, wer_by_condition.png, eda_stats.csv,
wer_by_length.csv, error_examples.csv
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

REF_COLS = ["true_transcript", "reference", "ref", "text", "transcript_true"]
HYP_COLS = ["pred_transcript", "predicted_transcript", "prediction", "hypothesis", "hyp", "pred", "transcript_pred"]


def find_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Could not find the {label} column. Columns are: {list(df.columns)}\n"
                     f"Edit REF_COLS / HYP_COLS at the top of this file.")


def words(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def edit_distance(ref, hyp):
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)]


def duration(path):
    try:
        return sf.info(path).duration
    except Exception:
        return np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", default=sorted(glob.glob("outputs/pred_*.csv")))
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    frames = []
    for p in args.pred:
        df = pd.read_csv(p)
        ref_col, hyp_col = find_col(df, REF_COLS, "reference"), find_col(df, HYP_COLS, "prediction")
        cond = df["condition"].iloc[0] if "condition" in df.columns else Path(p).stem
        ref_w = [words(t) for t in df[ref_col]]
        hyp_w = [words(t) for t in df[hyp_col].fillna("")]
        errs = [edit_distance(r, h) for r, h in zip(ref_w, hyp_w)]
        frames.append(pd.DataFrame({
            "condition": cond,
            "filename": df["filename"] if "filename" in df.columns else range(len(df)),
            "duration_s": [duration(a) for a in df["audio_path"]] if "audio_path" in df.columns else np.nan,
            "ref_words": [len(r) for r in ref_w],
            "hyp_words": [len(h) for h in hyp_w],
            "errors": errs,
            "ref": df[ref_col].values,
            "hyp": df[hyp_col].fillna("").values,
        }))
    data = pd.concat(frames, ignore_index=True)
    data["wer"] = data.errors / data.ref_words.clip(lower=1)
    order = list(dict.fromkeys(data.condition))

    # ---- Summary stats per condition
    stats = data.groupby("condition", sort=False).agg(
        clips=("wer", "size"),
        mean_duration_s=("duration_s", "mean"),
        mean_ref_words=("ref_words", "mean"),
        mean_hyp_words=("hyp_words", "mean"),
        mean_clip_wer=("wer", "mean"),
        median_clip_wer=("wer", "median"),
        pct_perfect_clips=("errors", lambda s: 100 * (s == 0).mean()),
    ).round(3)
    stats["corpus_wer"] = (data.groupby("condition", sort=False).errors.sum()
                           / data.groupby("condition", sort=False).ref_words.sum()).round(4)
    stats.to_csv(out / "eda_stats.csv")
    print(stats.to_string(), "\n")

    # ---- WER by clip length (is short = bad?)
    data["length_bucket"] = pd.cut(data.ref_words, [0, 5, 15, 30, 1000],
                                   labels=["1-5 words", "6-15", "16-30", "31+"])
    by_len = data.pivot_table(index="length_bucket", columns="condition", values="wer",
                              aggfunc="mean", observed=False)[order].round(3)
    counts = data[data.condition == order[0]].length_bucket.value_counts().sort_index()
    by_len.insert(0, "clips", counts)
    by_len.to_csv(out / "wer_by_length.csv")
    print("Mean clip WER by clip length:\n", by_len.to_string(), "\n")

    # ---- Error examples: 10 worst clips per condition
    examples = (data.sort_values("wer", ascending=False).groupby("condition", sort=False).head(10)
                [["condition", "filename", "wer", "ref", "hyp"]])
    examples.to_csv(out / "error_examples.csv", index=False)

    # ---- Figure 1: EDA panels
    base = data[data.condition == order[0]]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    if base.duration_s.notna().any():
        ax[0, 0].hist(base.duration_s.dropna(), bins=20, color="#4C72B0")
    ax[0, 0].set(title="Clip duration (s)", xlabel="seconds", ylabel="clips")
    ax[0, 1].hist(base.ref_words, bins=20, color="#55A868")
    ax[0, 1].set(title="Transcript length (words)", xlabel="words", ylabel="clips")
    for cond in order:
        s = data[data.condition == cond]
        ax[1, 0].scatter(s.ref_words, s.wer, s=14, alpha=0.6, label=cond)
    ax[1, 0].set(title="Clip WER vs transcript length", xlabel="reference words", ylabel="clip WER")
    ax[1, 0].legend()
    ax[1, 1].boxplot([data[data.condition == c].wer for c in order])
    ax[1, 1].set_xticklabels(order)
    ax[1, 1].set(title="Clip WER distribution", ylabel="clip WER")
    plt.tight_layout()
    plt.savefig(out / "eda.png", dpi=150)
    plt.close()

    # ---- Figure 2: corpus WER by condition
    plt.figure(figsize=(6, 4))
    vals = stats.corpus_wer.loc[order] * 100
    bars = plt.bar(order, vals, color=["#55A868", "#DD8452", "#C44E52"][:len(order)] or None)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center")
    plt.ylabel("WER (%)")
    plt.title("WER by condition")
    plt.tight_layout()
    plt.savefig(out / "wer_by_condition.png", dpi=150)

    print(f"Saved eda.png, wer_by_condition.png, eda_stats.csv, wer_by_length.csv, error_examples.csv in {out}/")