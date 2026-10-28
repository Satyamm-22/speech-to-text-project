import argparse, time
from pathlib import Path
import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel
from librespeech_dataset import load_split

def transcribe_df(df, model_name="base.en"):
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    preds = []
    for i, row in enumerate(df.itertuples(), 1):
        start = time.time()
        audio, sr = sf.read(row.audio_path, dtype="float32")
        segments, _ = model.transcribe(audio, language="en")
        text = " ".join(s.text.strip() for s in segments).lower()
        preds.append(text)
        print(f"[{i}/{len(df)}] {time.time()-start:.1f}s | {text[:60]}")
    out = df.copy()
    out["pred_transcript"] = preds
    out["model"] = model_name
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.manifest:
        df = pd.read_csv(args.manifest)
        tag = args.tag or "custom"
    else:
        df = load_split(args.split, n=args.n)
        tag = args.split
    result = transcribe_df(df, args.model)

    Path("outputs").mkdir(exist_ok=True)
    out_path = f"outputs/pred_{tag}_{args.model}.csv"
    result.to_csv(out_path, index=False)
    print("Saved", out_path)


