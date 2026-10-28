"""Test the Whisper pipeline on your own voice or your own audio file.

Mic mode (interactive, good for a live demo):
    python live_test.py
    python live_test.py --seconds 6 --model base.en

File mode:
    python live_test.py --file my_clip.wav --truth "the sentence you said"

If you type what you actually said, the result is scored (WER) and appended to
outputs/pred_own_voice_<model>.csv, which eda.py and keyword_analysis.py can read.
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel

SR = 16000


def words(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def wer(ref, hyp):
    ref, hyp = words(ref), words(hyp)
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(len(ref), 1)


def record(seconds, path):
    import sounddevice as sd
    print(f"Recording {seconds}s... speak now")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio.squeeze(), SR)


def transcribe(model, path):
    t0 = time.time()
    # Read with soundfile and pass an array, so faster-whisper skips its PyAV decoder
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:  # simple linear resample for files not recorded at 16 kHz
        audio = np.interp(np.linspace(0, len(audio), int(len(audio) * SR / sr), endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    segments, _ = model.transcribe(audio, language="en", beam_size=5)
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments)
    conf = float(np.mean([np.exp(s.avg_logprob) for s in segments])) if segments else 0.0
    return text, conf, time.time() - t0


def log_result(args, path, truth, text, conf):
    Path("outputs").mkdir(exist_ok=True)
    row = {"filename": path.stem, "audio_path": str(path), "true_transcript": truth,
           "pred_transcript": text, "condition": args.tag, "confidence": round(conf, 3)}
    out = Path(f"outputs/pred_{args.tag}_{args.model}.csv")
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="use an existing audio file instead of the mic")
    ap.add_argument("--truth", default="", help="what is actually said in --file")
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--tag", default="own_voice")
    ap.add_argument("--save-dir", default="data/own_voice")
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    if args.file:
        path = Path(args.file)
        text, conf, secs = transcribe(model, path)
        print(f"\nTranscript ({secs:.1f}s, confidence {conf:.2f}):\n  {text}")
        if args.truth:
            print(f"WER: {wer(args.truth, text):.1%}")
            log_result(args, path, args.truth, text, conf)
        raise SystemExit

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(save_dir.glob("*.wav")))
    while True:
        if input("\nPress Enter to record (q to quit): ").strip().lower() == "q":
            break
        n += 1
        path = save_dir / f"clip_{n:03d}.wav"
        record(args.seconds, path)
        text, conf, secs = transcribe(model, path)
        print(f"Transcript ({secs:.1f}s, confidence {conf:.2f}):\n  {text}")
        truth = input("What did you actually say? (Enter to skip scoring): ").strip()
        if truth:
            print(f"WER: {wer(truth, text):.1%}")
            log_result(args, path, truth, text, conf)