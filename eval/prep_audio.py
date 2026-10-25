"""Check recordings and noise files, and write 16 kHz mono 16-bit copies for the VAD eval.

    .venv/Scripts/python -m eval.prep_audio --clean data/vad_dev_tmp/clean --noise data/noise \
        --out-clean data/vad_dev --out-noise data/noise_16k

Per file: container/subtype, sample rate, channels, duration, peak, clipped samples, noise floor
(10th percentile of 10 ms frame energy) and speech level (95th percentile). Originals are never
written; copies go to the --out folders (a clip already in the target format is copied byte for byte).

Flags (clean clips): CLIPPED (any sample at full scale), QUIET (95th percentile below
QUIET_DBFS), NOISY (floor less than NOISY_MARGIN_DB below the speech level), GATED (floor at
digital silence: the recorder suppresses noise between words, so pauses are exact zeros).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio_io import load_wav
from app.config import SAMPLE_RATE

QUIET_DBFS = -40.0
NOISY_MARGIN_DB = 25.0
GATED_DBFS = -90.0


def stats(path: Path) -> dict:
    info = sf.info(str(path))
    raw, _ = sf.read(str(path), dtype="float32", always_2d=True)
    x = load_wav(path)  # 16 kHz mono float
    fr = x[: len(x) // 160 * 160].reshape(-1, 160)
    db = 10 * np.log10(np.mean(fr**2, axis=1) + 1e-12)
    floor, level = np.percentile(db, (10, 95))
    return {"file": path.name, "format": f"{info.format}/{info.subtype}", "sr": info.samplerate,
            "channels": info.channels, "duration_s": round(info.duration, 2),
            "peak_dbfs": round(float(20 * np.log10(np.abs(raw).max() + 1e-12)), 1),
            "clipped": int((np.abs(raw) >= 0.999).sum()), "floor_db": round(float(floor), 1),
            "level_db": round(float(level), 1),
            "ok_format": info.samplerate == SAMPLE_RATE and info.channels == 1 and info.subtype == "PCM_16"
                         and info.format == "WAV"}


def flags(s: dict) -> list[str]:
    out = []
    if s["clipped"]:
        out.append("CLIPPED")
    if s["level_db"] < QUIET_DBFS:
        out.append("QUIET")
    if s["floor_db"] > GATED_DBFS and s["level_db"] - s["floor_db"] < NOISY_MARGIN_DB:
        out.append("NOISY")
    if s["floor_db"] <= GATED_DBFS:
        out.append("GATED")
    return out


def convert(src: Path, dst: Path, ok_format: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if ok_format:
        shutil.copyfile(src, dst)
    else:
        sf.write(str(dst), load_wav(src), SAMPLE_RATE, subtype="PCM_16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path, required=True)
    ap.add_argument("--noise", type=Path)
    ap.add_argument("--out-clean", type=Path, required=True)
    ap.add_argument("--out-noise", type=Path)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print("| file | format | Hz | ch | s | peak dBFS | clipped | floor dB | speech p95 dB | flags | action |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    jobs = [(p, args.out_clean, True) for p in sorted(args.clean.glob("*.wav"))]
    if args.noise:
        jobs += [(p, args.out_noise, False) for p in sorted(args.noise.glob("*.wav"))]
    for src, out_dir, is_clean in jobs:
        s = stats(src)
        f = flags(s) if is_clean else []
        convert(src, out_dir / src.name, s["ok_format"])
        action = "copied" if s["ok_format"] else "converted to 16 kHz mono PCM_16"
        print(f"| {s['file']} | {s['format']} | {s['sr']} | {s['channels']} | {s['duration_s']} | {s['peak_dbfs']} | "
              f"{s['clipped']} | {s['floor_db']} | {s['level_db']} | {' '.join(f) or '-'} | {action} |")


if __name__ == "__main__":
    main()
