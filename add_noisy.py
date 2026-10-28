import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from librespeech_dataset import load_split

def add_noise(audio, snr_db, rng):
    noise_power = np.mean(audio ** 2) / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), audio.shape)
    return np.clip(audio + noise, -1, 1).astype("float32")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--snr", type=float, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    tag = f"noisy_snr{int(args.snr)}"
    out_dir = Path("data") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_split(args.split, n=args.n)     
    new_paths = []
    for row in df.itertuples():
        audio, sr = sf.read(row.audio_path, dtype="float32")
        path = out_dir / f"{row.filename}.wav"
        sf.write(path, add_noise(audio, args.snr, rng), sr)
        new_paths.append(str(path))

    df["audio_path"] = new_paths
    df["condition"] = tag
    df.to_csv(f"data/manifest_{tag}.csv", index=False)
    print("Saved", f"data/manifest_{tag}.csv")