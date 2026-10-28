"""Streamlit app: give it your audio + what you actually said -> transcript, WER, accuracy, precision, confidence.

Run:
    python -m pip install streamlit
    streamlit run app.py
"""
import re
import shutil
import subprocess
import tempfile
import time
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

SR = 16000
MODELS = ["tiny.en", "base.en", "small.en"]
LOW_CONF = 0.5
AUDIO_TYPES = ["wav", "flac", "ogg", "opus", "mp3", "m4a"]


# ---------------------------------------------------------------- text + metrics
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


def compute_metrics(ref_text, tokens):
    """tokens = [(word, probability), ...] as heard by the model. Returns None if the reference is empty."""
    ref, hyp = words(ref_text), [t for t, _ in tokens]
    if not ref:
        return None
    rows, k = [], 0
    for op, r, h in align(ref, hyp):
        p = None
        if op != "del":
            p = tokens[k][1]
            k += 1
        rows.append({"op": op, "ref": r, "heard": h, "conf": p})
    H = sum(r["op"] == "ok" for r in rows)
    S = sum(r["op"] == "sub" for r in rows)
    D = sum(r["op"] == "del" for r in rows)
    I = sum(r["op"] == "ins" for r in rows)
    N = len(ref)
    wer = (S + D + I) / N
    precision = H / (H + S + I) if (H + S + I) else 0.0   # of the words the model wrote, how many were right
    recall = H / N                                          # of the words you said, how many it got right
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"rows": rows, "N": N, "H": H, "S": S, "D": D, "I": I, "wer": wer,
            "accuracy": max(0.0, 1 - wer), "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------- audio + model
def load_audio(path):
    """Mono float32 at 16 kHz; falls back to ffmpeg for m4a/mp3 if soundfile can't read the file."""
    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"Can't read {Path(path).suffix} files. Install ffmpeg (brew install ffmpeg) "
                               f"or upload a .wav/.flac/.opus file.")
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


@st.cache_resource(show_spinner="Loading Whisper model (first time downloads it)...")
def get_model(name):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe(model, audio):
    t0 = time.time()
    segments, _ = model.transcribe(audio, language="en", beam_size=5, word_timestamps=True)
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments)
    seg_conf = float(np.mean([np.exp(s.avg_logprob) for s in segments])) if segments else 0.0
    tokens = [(tok, float(w.probability)) for s in segments for w in (s.words or []) for tok in words(w.word)]
    if not tokens:  # word-level scores unavailable: fall back to the segment score
        tokens = [(t, seg_conf) for t in words(text)]
    conf = float(np.mean([p for _, p in tokens])) if tokens else 0.0
    return {"text": text, "tokens": tokens, "confidence": conf, "secs": time.time() - t0}


# ---------------------------------------------------------------- display helpers
COLORS = {"ok": "rgba(46,160,67,.30)", "sub": "rgba(248,81,73,.35)",
          "del": "rgba(210,153,34,.40)", "ins": "rgba(163,113,247,.35)"}
LABELS = {"ok": "correct", "sub": "wrong word", "del": "missed", "ins": "extra"}


def alignment_html(rows):
    cells = []
    for r in rows:
        c = COLORS[r["op"]]
        cells.append(
            '<div style="display:inline-block;margin:0 6px 10px 0;text-align:center;font-family:monospace">'
            f'<div style="background:{c};padding:2px 8px;border-radius:4px 4px 0 0">{escape(r["ref"] or "·")}</div>'
            f'<div style="background:{c};padding:2px 8px;border-radius:0 0 4px 4px;'
            f'border-top:1px solid rgba(128,128,128,.35)">{escape(r["heard"] or "·")}</div></div>')
    legend = " ".join(f'<span style="background:{COLORS[k]};padding:1px 8px;border-radius:4px">{v}</span>'
                      for k, v in LABELS.items())
    return f'<div style="line-height:1.2">{"".join(cells)}</div><div style="margin-top:4px">{legend}</div>'


def pct(x):
    return f"{x:.1%}"


# ---------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="Speech-to-Text Evaluator", page_icon="🎙️", layout="wide")
    st.title("🎙️ Speech-to-Text Evaluator")
    st.caption("A pretrained Whisper model transcribes your audio, then the transcript is compared "
               "with what you actually said.")

    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Whisper model", MODELS, index=1)
        st.caption("tiny = fastest · base = balanced · small = most accurate but slower. "
                   "The first use of a model downloads it.")
        tag = st.text_input("Condition label", "app")
        save = st.checkbox("Also append result to outputs/pred_<label>_<model>.csv", value=False,
                           help="Same format as your other prediction files, so eda.py and "
                                "keyword_analysis.py can read it.")

    left, right = st.columns(2)
    with left:
        st.subheader("1. Your audio")
        sources = ["Upload a file"] + (["Record with microphone"] if hasattr(st, "audio_input") else [])
        source = st.radio("Audio source", sources, horizontal=True, label_visibility="collapsed")
        audio_file = (st.file_uploader("Audio file", type=AUDIO_TYPES, label_visibility="collapsed")
                      if source == sources[0] else st.audio_input("Record"))
        if audio_file is not None:
            st.audio(audio_file)
    with right:
        st.subheader("2. What you actually said")
        reference = st.text_area("Reference text", height=160, label_visibility="collapsed",
                                 placeholder="Type exactly what is said in the recording. "
                                             "Leave empty to see only the transcript.")

    if st.button("Transcribe & evaluate", type="primary"):
        if audio_file is None:
            st.warning("Add an audio file or record something first.")
        else:
            suffix = Path(getattr(audio_file, "name", "") or "rec.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_file.getvalue())
            try:
                audio = load_audio(tmp.name)
                if len(audio) < SR * 0.3:
                    st.error("The audio is too short.")
                else:
                    with st.spinner("Transcribing..."):
                        res = transcribe(get_model(model_name), audio)
                    metrics = compute_metrics(reference, res["tokens"]) if reference.strip() else None
                    st.session_state["result"] = {
                        "res": res, "metrics": metrics, "reference": reference.strip(), "model": model_name,
                        "duration": len(audio) / SR, "name": getattr(audio_file, "name", "recording.wav")}
                    if save and metrics:
                        Path("outputs").mkdir(exist_ok=True)
                        out = Path(f"outputs/pred_{tag}_{model_name}.csv")
                        row = {"filename": Path(st.session_state["result"]["name"]).stem,
                               "audio_path": st.session_state["result"]["name"],
                               "true_transcript": reference.strip(), "pred_transcript": res["text"],
                               "condition": tag, "confidence": round(res["confidence"], 3)}
                        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
                        st.toast(f"Saved to {out}")
            except Exception as e:
                st.error(f"Could not process the audio: {e}")
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    result = st.session_state.get("result")
    if not result:
        return
    res, metrics = result["res"], result["metrics"]

    st.divider()
    st.subheader("What the model heard")
    st.info(res["text"] or "(nothing detected)")

    if metrics:
        cols = st.columns(6)
        for col, (label, value, hint) in zip(cols, [
            ("WER", pct(metrics["wer"]), "Word error rate. Lower is better."),
            ("Accuracy", pct(metrics["accuracy"]), "1 - WER."),
            ("Precision", pct(metrics["precision"]), "Of the words the model wrote, the share that were right."),
            ("Recall", pct(metrics["recall"]), "Of the words you said, the share the model got right."),
            ("F1", pct(metrics["f1"]), "Balance of precision and recall."),
            ("Confidence", pct(res["confidence"]), "Average probability Whisper gave to its own words."),
        ]):
            col.metric(label, value, help=hint)
        st.caption(f"{metrics['N']} reference words · {metrics['H']} correct · {metrics['S']} wrong · "
                   f"{metrics['D']} missed · {metrics['I']} extra   |   "
                   f"{result['duration']:.1f}s of audio transcribed in {res['secs']:.1f}s with {result['model']}")

        st.subheader("Word by word")
        st.caption("Top line = what you said. Bottom line = what the model heard.")
        st.markdown(alignment_html(metrics["rows"]), unsafe_allow_html=True)

        table = pd.DataFrame([{"Said": r["ref"], "Heard": r["heard"], "Result": LABELS[r["op"]],
                               "Confidence": None if r["conf"] is None else round(r["conf"], 3)}
                              for r in metrics["rows"]])
        tab1, tab2 = st.tabs(["Comparison table", "Confidence per heard word"])
        with tab1:
            st.dataframe(table, hide_index=True)
            st.download_button("Download table (CSV)", table.to_csv(index=False), "word_comparison.csv", "text/csv")
        with tab2:
            heard = pd.DataFrame([{"word": f"{i + 1}. {r['heard']}", "confidence": r["conf"],
                                   "result": LABELS[r["op"]]}
                                  for i, r in enumerate(metrics["rows"]) if r["op"] != "del"])
            if len(heard):
                st.bar_chart(heard, x="word", y="confidence")
                low = heard[heard.confidence < LOW_CONF]
                wrong_low = low[low.result != "correct"]
                st.caption(f"{len(low)} heard words had confidence below {LOW_CONF:.0%}; "
                           f"{len(wrong_low)} of them were also wrong or extra. "
                           "Low confidence is a useful warning sign for mistakes.")
    else:
        st.metric("Confidence", pct(res["confidence"]),
                  help="Average probability Whisper gave to its own words.")
        st.caption(f"{result['duration']:.1f}s of audio transcribed in {res['secs']:.1f}s. "
                   "Type what you actually said above and run again to get WER, accuracy and precision.")

    with st.expander("How are these numbers calculated?"):
        st.markdown(
            "Your reference text and the model's transcript are lower-cased, stripped of punctuation and aligned "
            "word by word. Each word is **correct**, **wrong** (substituted), **missed** (deleted) or **extra** "
            "(inserted).\n\n"
            "- **WER** = (wrong + missed + extra) / words you said\n"
            "- **Accuracy** = 1 − WER (never below 0)\n"
            "- **Precision** = correct / (correct + wrong + extra), i.e. correct / words the model wrote\n"
            "- **Recall** = correct / words you said\n"
            "- **F1** = harmonic mean of precision and recall\n"
            "- **Confidence** = average probability Whisper assigned to each word it output "
            "(it is the model's own belief, not a measure of correctness)\n\n"
            "Contractions and number words count as errors if they are written differently "
            "(\"there's\" vs \"there is\", \"5\" vs \"five\").")


if __name__ == "__main__":
    main()