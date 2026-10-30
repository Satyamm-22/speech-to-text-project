
# 🎙️ Speech-to-Text Project (ID409)

How good is a speech recognition model *really*? And what happens to it when the audio gets noisy?

This project puts a pretrained [Whisper](https://github.com/openai/whisper) model (run through [faster-whisper](https://github.com/SYSTRAN/faster-whisper)) to the test. We feed it clean audio, deliberately messy audio, and even our own voices, then measure how many words it gets wrong and what that means in practice.

No training involved. Whisper is used as-is, so everything here is about **evaluating and understanding** the model, not building a new one.

---

## What you can do with it

- **Transcribe** clips from the LibriSpeech dataset and save the results to CSV.
- **Measure** accuracy with Word Error Rate (WER), including a breakdown of substitutions, deletions and insertions.
- **Add noise** to the audio at a chosen signal-to-noise ratio and see how quickly the model falls apart.
- **Explore** the results with plots and stats: WER by clip length, the worst clips, error examples.
- **Test keyword detection**, which asks a more practical question: even if WER goes up, can the model still catch the words that matter?
- **Try it yourself** with a microphone, an audio file, or a small web app.

---

## Getting started

### 1. Install the dependencies

```bash
pip install -r requirements.txt
```

This installs `faster-whisper`, `jiwer`, `pandas`, `matplotlib`, `sounddevice` and `soundfile`.

A couple of extras you may need:

- **Streamlit**, only if you want the web app: `pip install streamlit`
- **ffmpeg**, only if you want to use `.mp3` or `.m4a` files (for example `brew install ffmpeg` on macOS). WAV, FLAC, OGG and Opus work without it.

### 2. Run the pipeline

Everything below is run from the project's root folder.

**Step 1: transcribe some clean audio**

```bash
python transcribe.py --split test-clean --n 100 --model base.en
```

The first run downloads the LibriSpeech `test-clean` split into `data/` (it's a big file, so give it a moment) and picks a random sample of clips. Results land in `outputs/pred_test-clean_base.en.csv`.

**Step 2: score it**

```bash
python evaluate.py --pred outputs/pred_test-clean_base.en.csv
```

You'll get the overall WER, the number of substitutions, deletions and insertions, and the five worst clips. A per-clip breakdown is saved to `outputs/per_clip_wer.csv`.

**Step 3: make it harder with noise**

```bash
python add_noisy.py --n 100 --snr 10
python add_noisy.py --n 100 --snr 0
```

This copies the same clips, mixes in Gaussian noise at the SNR you pick (lower means noisier), and writes a new manifest such as `data/manifest_noisy_snr10.csv`. Then transcribe the noisy version:

```bash
python transcribe.py --manifest data/manifest_noisy_snr10.csv --tag noisy_snr10 --model base.en
python evaluate.py --pred outputs/pred_noisy_snr10_base.en.csv
```

**Step 4: look at the big picture**

```bash
python eda.py --pred outputs/pred_test-clean_base.en.csv \
                     outputs/pred_noisy_snr10_base.en.csv \
                     outputs/pred_noisy_snr0_base.en.csv
```

This produces charts and tables in `outputs/`: clip duration and transcript length distributions, WER by condition, WER by clip length, and the ten worst clips per condition.

**Step 5: check keyword detection**

```bash
python keyword_analysis.py --pred outputs/pred_test-clean_base.en.csv \
                                  outputs/pred_noisy_snr10_base.en.csv \
                                  outputs/pred_noisy_snr0_base.en.csv
```

WER treats every word equally, but in real applications some words matter more than others. This script picks the most common content words (or uses your own list with `--keywords`) and checks whether the model's transcript still contains them. You get precision, recall and F1 for each noise level, plus the keywords that break first.

---

## Try it on your own voice

### The web app

```bash
streamlit run app.py
```

Upload an audio file or record from your browser, then type what you actually said. The app shows you:

- what the model heard
- WER, accuracy, precision, recall, F1 and Whisper's own confidence
- a word-by-word comparison, colour-coded as correct, wrong, missed or extra
- a chart of confidence per word, so you can see whether low confidence tends to flag real mistakes

You can choose between three model sizes in the sidebar: `tiny.en` (fastest), `base.en` (balanced) and `small.en` (most accurate, slower).

### The terminal demo

```bash
python demo.py                 # record from your mic
python demo.py --read          # get a random sentence to read out loud
python demo.py --file clip.wav # use a saved recording
```

It plays out like a little game: it shows what the model heard, you type what you really said, and it prints a colour-coded comparison with the WER. Every scored clip is saved so the analysis scripts can pick it up later.

### The quick mic test

```bash
python live_test.py
python live_test.py --file my_clip.wav --truth "the sentence you said"
```

A lighter version of the demo, handy for fast checks.

---

## How the numbers work

Both the reference text and the model's output are lower-cased and stripped of punctuation. They're then lined up word by word, and each word is labelled as correct, a substitution (wrong word), a deletion (missed) or an insertion (extra).

| Metric | What it means |
| --- | --- |
| **WER** | (substitutions + deletions + insertions) ÷ number of words you said. Lower is better. |
| **Accuracy** | 1 − WER, never below zero. |
| **Precision** | Of the words the model wrote, how many were right. |
| **Recall** | Of the words you said, how many the model got right. |
| **F1** | The balance between precision and recall. |
| **Confidence** | The average probability Whisper gave its own words. It's the model's belief, not proof it was right. |

One thing to keep in mind: small differences in how things are written count as errors. "there's" vs "there is" or "5" vs "five" will both be marked wrong, even though a human would say the model understood perfectly.

---

## What's in the repo

| File | What it does |
| --- | --- |
| `librespeech_dataset.py` | Downloads LibriSpeech splits and builds a table of clips and their true transcripts |
| `transcribe.py` | Runs Whisper over a set of clips and saves predictions to CSV |
| `evaluate.py` | Calculates overall and per-clip WER |
| `add_noisy.py` | Creates noisy copies of the audio at a chosen SNR |
| `eda.py` | Plots and stats: WER by condition and by clip length, worst clips |
| `keyword_analysis.py` | Keyword detection precision, recall and F1 across noise levels |
| `app.py` | Streamlit app for testing your own audio |
| `demo.py` | Interactive terminal demo with colour-coded results |
| `live_test.py` | Quick microphone or file test |
| `data/` | Downloaded audio, manifests and recordings |
| `outputs/` | Predictions, metrics, plots and tables |
| `requirements.txt` | Python dependencies |

---

## Good to know

- **Everything runs on CPU** with int8 quantisation, so no GPU is needed. Larger models will just be slower.
- **English only.** The `.en` Whisper models are used and the language is fixed to English.
- **Model weights download on first use**, so the first run of each model size takes a little longer.
- Prediction files follow one shared format (`filename`, `audio_path`, `true_transcript`, `pred_transcript`, `condition`), which is why every script can read the output of every other script.

---

## Credits

- [OpenAI Whisper](https://github.com/openai/whisper) and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for the speech recognition
- [LibriSpeech](https://www.openslr.org/12) for the audio and reference transcripts
- [jiwer](https://github.com/jitsi/jiwer) for WER calculation

---
