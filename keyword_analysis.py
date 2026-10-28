"""Keyword detection analysis: how does WER translate into task performance?

A clip counts as "keyword present" if the word appears in the transcript.
Ground truth = keyword in the true transcript, prediction = keyword in the ASR transcript.

Usage:
    python keyword_analysis.py --pred outputs/pred_test-clean_base.en.csv \
        outputs/pred_noisy_snr10_base.en.csv outputs/pred_noisy_snr0_base.en.csv

Optional:
    --keywords river king ...   use your own list instead of auto-picking
    --top-k 20                  how many keywords to auto-pick
"""
import argparse
import glob
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STOP = set("""a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her
here hers him his how i if in into is it its just me more most my no nor not now of off on once only or other our
out over own same she should so some such than that the their them then there these they this those through to too
under until up very was we were what when where which while who whom why will with would you your said""".split())

REF_COLS = ["true_transcript", "reference", "ref", "text", "transcript_true"]
HYP_COLS = ["pred_transcript", "predicted_transcript", "prediction", "hypothesis", "hyp", "pred", "transcript_pred"]


def norm_words(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def find_col(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"Could not find the {label} column. Columns are: {list(df.columns)}\n"
                     f"Edit REF_COLS / HYP_COLS at the top of this file.")


def pick_keywords(ref_texts, k):
    """Top-k content words by number of clips they appear in."""
    doc_freq = Counter()
    for t in ref_texts:
        doc_freq.update({w for w in norm_words(t) if w not in STOP and len(w) >= 4})
    return [w for w, _ in doc_freq.most_common(k)]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", nargs="+", default=sorted(glob.glob("outputs/pred_*.csv")))
    ap.add_argument("--keywords", nargs="+", default=None)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    runs = {}
    for p in args.pred:
        df = pd.read_csv(p)
        cond = df["condition"].iloc[0] if "condition" in df.columns else Path(p).stem
        runs[cond] = df

    first = next(iter(runs.values()))
    ref_col = find_col(first, REF_COLS, "reference")
    hyp_col = find_col(first, HYP_COLS, "prediction")

    # Same keyword list for every condition, chosen from the true transcripts
    keywords = [k.lower() for k in args.keywords] if args.keywords else pick_keywords(first[ref_col], args.top_k)
    (out / "keywords.txt").write_text("\n".join(keywords))
    print(f"Keywords ({len(keywords)}): {', '.join(keywords)}\n")

    summary, per_word = [], []
    for cond, df in runs.items():
        refs = [set(norm_words(t)) for t in df[ref_col]]
        hyps = [set(norm_words(t)) for t in df[hyp_col].fillna("")]
        TP = FP = FN = 0
        for kw in keywords:
            tp = sum(kw in r and kw in h for r, h in zip(refs, hyps))
            fp = sum(kw not in r and kw in h for r, h in zip(refs, hyps))
            fn = sum(kw in r and kw not in h for r, h in zip(refs, hyps))
            TP, FP, FN = TP + tp, FP + fp, FN + fn
            _, rec, _ = prf(tp, fp, fn)
            per_word.append({"condition": cond, "keyword": kw, "occurrences": tp + fn,
                             "found": tp, "missed": fn, "false_alarms": fp, "recall": round(rec, 3)})
        p, r, f = prf(TP, FP, FN)
        summary.append({"condition": cond, "clips": len(df), "TP": TP, "FP": FP, "FN": FN,
                        "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})

    summ = pd.DataFrame(summary)
    words = pd.DataFrame(per_word)
    summ.to_csv(out / "keyword_metrics.csv", index=False)
    words.to_csv(out / "keyword_per_word.csv", index=False)
    print(summ.to_string(index=False))

    # Which keywords break first? (lowest recall in the noisiest condition)
    worst_cond = summ.sort_values("recall").iloc[0]["condition"]
    hardest = words[words.condition == worst_cond].sort_values("recall").head(5)
    print(f"\nHardest keywords under '{worst_cond}':")
    print(hardest[["keyword", "occurrences", "found", "missed", "recall"]].to_string(index=False))

    # Plot: precision / recall / F1 per condition
    ax = summ.set_index("condition")[["precision", "recall", "f1"]].plot(kind="bar", figsize=(7, 4), ylim=(0, 1.05))
    ax.set_ylabel("score")
    ax.set_title("Keyword detection vs noise level")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out / "keyword_pr.png", dpi=150)
    print(f"\nSaved {out}/keyword_metrics.csv, keyword_per_word.csv, keywords.txt, keyword_pr.png")