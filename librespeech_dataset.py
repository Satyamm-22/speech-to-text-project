import tarfile, random, urllib.request
from pathlib import Path
import pandas as pd

BASE_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
DATA_DIR = Path("data")

def download_split(split="test-clean"):
    """Download and extract one split. Skips steps already done."""
    DATA_DIR.mkdir(exist_ok=True)
    root = DATA_DIR / "LibriSpeech" / split
    if root.exists():
        return root
    tar_path = DATA_DIR / f"{split}.tar.gz"
    if not tar_path.exists():
        print(f"Downloading {split}...")
        urllib.request.urlretrieve(BASE_URL.format(split=split), tar_path)
    print(f"Extracting {split}...")
    with tarfile.open(tar_path) as t:
        t.extractall(DATA_DIR)
    return root

def load_split(split="test-clean", n=None, seed=42):
    """Return a DataFrame: filename, audio_path, true_transcript, condition."""
    root = download_split(split)
    rows = []
    for trans in root.rglob("*.trans.txt"):
        for line in trans.read_text().splitlines():
            utt_id, text = line.split(" ", 1)
            rows.append({
                "filename": utt_id,
                "audio_path": str(trans.parent / f"{utt_id}.flac"),
                "true_transcript": text.lower(),
                "condition": "clean" if "clean" in split else "other",
            })
    random.Random(seed).shuffle(rows)
    return pd.DataFrame(rows[:n] if n else rows)

if __name__ == "__main__":
    df = load_split("test-clean", n=100)
    df.to_csv(DATA_DIR / "manifest.csv", index=False)
    print(df.head())