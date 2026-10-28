import argparse
import re
from pathlib import Path
import pandas as pd
import jiwer

ABBREV = {"mr": "mister", "mrs": "missus", "dr": "doctor"}

def normalize(text):
    t = str(text).lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    words = [w.strip("'") for w in t.split()]          
    words = [ABBREV.get(w, w) for w in words if w]
    return " ".join(words)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="outputs/pred_test-clean_base.en.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.pred).fillna("")
    refs = [normalize(t) for t in df["true_transcript"]]
    hyps = [normalize(t) for t in df["pred_transcript"]]

    # Overall WER across all clips, with error breakdown
    out = jiwer.process_words(refs, hyps)
    print(f"Clips: {len(df)}")
    print(f"WER: {out.wer:.2%}")
    print(f"Substitutions: {out.substitutions}  Deletions: {out.deletions}  Insertions: {out.insertions}")

    # Per-clip WER, to find the worst cases
    df["ref_norm"], df["hyp_norm"] = refs, hyps
    df["wer"] = [jiwer.wer(r, h) for r, h in zip(refs, hyps)]
    df.sort_values("wer", ascending=False).to_csv("outputs/per_clip_wer.csv", index=False)

    print("\nWorst 5 clips:")
    for r in df.sort_values("wer", ascending=False).head(5).itertuples():
        print(f"\nWER {r.wer:.0%}\n  REF: {r.ref_norm}\n  HYP: {r.hyp_norm}")