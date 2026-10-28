"""Live demo: speak (or give a recording) -> see what Whisper heard -> type what you actually said -> WER.

    python demo.py                         # press Enter to record from the mic, OR drag a file in and press Enter
    python demo.py --file clip.opus        # use a saved recording instead (wav, flac, opus, m4a, mp3)
    python demo.py --read                  # read a random sentence from data/manifest.csv (no typing needed)

Output per clip:
    Model heard: "hello good evening"   (confidence 0.55)
    What did you actually say?  <you type it>
    SAID / HEARD comparison + WER

Every scored clip is appended to outputs/pred_<tag>_<model>.csv (eda.py / keyword_analysis.py can read it).
"""
import argparse
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
GREEN, RED, YEL, MAG, BOLD, DIM, END = "\033[92m", "\033[91m", "\033[93m", "\033[95m", "\033[1m", "\033[2m", "\033[0m"
FALLBACK = ["there is no fear of that sir", "now then let's enter the city and enjoy the grand feast",
            "i don't know all of them but i know a few", "the floor more than anything else showed the great age"]


def words(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def align(ref, hyp):
    """Word alignment -> list of (op, ref_word, hyp_word); op is ok / sub / del / ins."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0], d[0, :] = range(n + 1), range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
    return ops[::-1]


def load_audio(path):
    """Mono float32 at 16 kHz; uses ffmpeg for formats soundfile can't read (m4a, mp3)."""
    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception:
        if not shutil.which("ffmpeg"):
            raise SystemExit(f"Can't read {path.suffix} files. Run: brew install ffmpeg (or convert to .wav).")
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "c.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ar", str(SR), "-ac", "1", str(wav)],
                           check=True, capture_output=True)
            audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = np.interp(np.linspace(0, len(audio), int(len(audio) * SR / sr), endpoint=False),
                          np.arange(len(audio)), audio).astype("float32")
    return audio


def record_until_enter(path):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Microphone mode needs sounddevice. Run: python -m pip install sounddevice\n"
                         "Or score a saved recording: python demo.py --file data/friend/clip_1.opus")
    chunks = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        callback=lambda indata, *_: chunks.append(indata.copy())):
        input(f"{RED}{BOLD}  ● RECORDING{END}  read the sentence, then press Enter to stop ")
    audio = np.concatenate(chunks).squeeze() if chunks else np.zeros(0, dtype="float32")
    sf.write(str(path), audio, SR)
    return audio


def transcribe(model, audio):
    t0 = time.time()
    segments, _ = model.transcribe(audio, language="en", beam_size=5)
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments)
    conf = float(np.mean([np.exp(s.avg_logprob) for s in segments])) if segments else 0.0
    return text, conf, time.time() - t0


def show_result(ref, hyp, conf, secs):
    """Print reference vs heard, aligned word by word, then the WER. Returns (errors, n_ref)."""
    ops = align(words(ref), words(hyp))
    colors = {"ok": GREEN, "sub": RED, "del": YEL, "ins": MAG}
    cols = []
    for op, r, h in ops:
        w = max(len(r), len(h), 1)
        cols.append((colors[op], (r or "·").ljust(w), (h or "·").ljust(w), w))

    print(f"\n{BOLD}{'═' * 64}{END}")
    # wrap long sentences into blocks of ~60 characters
    block, width = [], 0
    blocks = []
    for c in cols:
        if width + c[3] + 1 > 60 and block:
            blocks.append(block)
            block, width = [], 0
        block.append(c)
        width += c[3] + 1
    blocks.append(block)
    for b in blocks:
        print(f"  {DIM}SAID   {END} " + " ".join(f"{c}{r}{END}" for c, r, _, _ in b))
        print(f"  {DIM}HEARD  {END} " + " ".join(f"{c}{h}{END}" for c, _, h, _ in b) + "\n")
    print(f"  {GREEN}correct{END}  {RED}wrong word{END}  {YEL}missed{END}  {MAG}extra{END}")

    sub, dele, ins = (sum(o == k for o, _, _ in ops) for k in ("sub", "del", "ins"))
    n_ref = max(len(words(ref)), 1)
    errors = sub + dele + ins
    wer = errors / n_ref
    color = GREEN if wer < 0.1 else YEL if wer < 0.3 else RED
    print(f"\n  {BOLD}WER: {color}{wer:.1%}{END}   "
          f"({sub} wrong, {dele} missed, {ins} extra, out of {n_ref} words)   "
          f"{DIM}confidence {conf:.2f}, transcribed in {secs:.1f}s{END}")
    print(f"{BOLD}{'═' * 64}{END}")
    return errors, n_ref


def log_row(out, path, truth, text, conf, tag):
    row = {"filename": path.stem, "audio_path": str(path), "true_transcript": truth,
           "pred_transcript": text, "condition": tag, "confidence": round(conf, 3)}
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


def sentence_pool(manifest):
    try:
        df = pd.read_csv(manifest)
        col = next(c for c in ("true_transcript", "reference", "text") if c in df.columns)
        pool = [s for s in df[col].dropna().unique() if 6 <= len(words(s)) <= 14]
        return pool or FALLBACK
    except Exception:
        return FALLBACK


def mic_available():
    try:
        import sounddevice  # noqa: F401
        return True
    except ImportError:
        return False


def heard_then_ask(model, audio, truth=None):
    """Transcribe, show what the model heard, then ask what was actually said."""
    text, conf, secs = transcribe(model, audio)
    print(f"\n{BOLD}Model heard:{END} \"{text}\"   {DIM}(confidence {conf:.2f}){END}")
    if truth is None:
        truth = input("\nWhat did you actually say? (Enter to skip WER): ").strip()
    return text, conf, secs, truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="use a saved recording instead of the mic")
    ap.add_argument("--truth", default=None, help="what is said in --file (otherwise you are asked)")
    ap.add_argument("--read", action="store_true", help="show a random sentence to read aloud (no typing needed)")
    ap.add_argument("--sentence", default=None, help="read this specific sentence aloud")
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--model", default="base.en")
    ap.add_argument("--tag", default="own_voice")
    ap.add_argument("--save-dir", default="data/own_voice")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    print(f"Loading Whisper {args.model} (pretrained, no training)...")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    Path("outputs").mkdir(exist_ok=True)
    out = Path(f"outputs/pred_{args.tag}_{args.model}.csv")

    # ---- File mode (also the fallback when no microphone package is installed)
    if args.file or not mic_available():
        if not args.file:
            print(f"\n{YEL}Microphone mode needs sounddevice (python -m pip install sounddevice).{END}")
            args.file = input("Meanwhile, give me the path to an audio file (drag it here): ")
        path = Path(args.file.strip().strip("'\"").replace("\\ ", " "))
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        text, conf, secs, truth = heard_then_ask(model, load_audio(path), args.truth)
        if truth:
            show_result(truth, text, conf, secs)
            log_row(out, path, truth, text, conf, args.tag)
        return

    # ---- Microphone mode
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(save_dir.glob("*.wav")))
    pool = sentence_pool(args.manifest) if args.read else []
    random.shuffle(pool)
    total_err = total_words = rounds = 0

    while True:
        sentence = args.sentence or (pool.pop() if args.read and pool else None)
        if sentence:
            print(f"\n{BOLD}Read this sentence aloud:{END}\n\n    {BOLD}\"{sentence[0].upper() + sentence[1:].lower()}\"{END}\n")
        cmd = input("  Press Enter to record, or drag an audio file in here (q to quit): ").strip()
        if cmd.lower() == "q":
            break
        if cmd:  # a file path was given instead of recording
            path = Path(cmd.strip("'\"").replace("\\ ", " "))
            if not path.exists():
                print(f"  File not found: {path}")
                continue
            audio = load_audio(path)
        else:
            n += 1
            path = save_dir / f"clip_{n:03d}.wav"
            audio = record_until_enter(path)
            if len(audio) < SR * 0.5:
                print("  Recording too short, try again.")
                continue
        text, conf, secs, truth = heard_then_ask(model, audio, sentence)
        if truth:
            err, n_ref = show_result(truth, text, conf, secs)
            log_row(out, path, truth, text, conf, args.tag)
            total_err, total_words, rounds = total_err + err, total_words + n_ref, rounds + 1
        args.sentence = None  # --sentence applies to the first round only

    if rounds:
        print(f"\n{BOLD}Session: {rounds} clips, overall WER {total_err / total_words:.1%}{END}   saved to {out}")


if __name__ == "__main__":
    main()