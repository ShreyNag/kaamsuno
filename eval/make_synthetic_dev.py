"""Synthetic stand-in for the VAD dev set until Aastha's labelled clips arrive.

Speech: Windows SAPI TTS phrases (romanised Hinglish read by an en-US voice, so
pronunciation is poor but the signal is real speech-like audio). Each phrase is
synthesised separately, trimmed, then concatenated with known gaps, so the
labels are exact at the phrase level.

Noise: procedural drilling / generator / traffic / hammering / crowd (TTS babble).

Writes data/vad_dev_synth/{clip_XX.wav, vad_labels.csv} and data/noise_synth/*.wav.
Windows only (uses System.Speech via PowerShell).
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import DATA_DIR, EVAL, SAMPLE_RATE

SR = SAMPLE_RATE
OUT_CLIPS = DATA_DIR / "vad_dev_synth"
OUT_NOISE = DATA_DIR / "noise_synth"

PHRASES = [
    "umm", "teesri manzil pe", "dhai sau bori cement kam hai", "matlab", "tower B ke paas",
    "crane ka hydraulic leak ho raha hai", "aa", "jaise ki", "do log ko chot lagi hai",
    "scaffolding hil raha hai", "employee ID four five seven two", "bas itna hi",
    "JCB ka engine band ho gaya", "sariya ka stock khatam", "toh", "paanchvi manzil",
    "mixer kharab hai subah se", "woh", "aur rait bhi kam hai", "fire extinguisher nahi mil raha",
]
VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop"]

_PS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$jobs = Get-Content -Raw -Path $args[0] | ConvertFrom-Json
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($j in $jobs) {
  $s.SelectVoice($j.voice); $s.Rate = $j.rate
  $s.SetOutputToWaveFile($j.path, $fmt); $s.Speak($j.text)
}
$s.SetOutputToNull()
"""


def tts_batch(jobs: list[dict]) -> list[np.ndarray]:
    """Synthesise [{text, voice, rate}] -> list of float32 arrays at 16 kHz."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for i, j in enumerate(jobs):
            j["path"] = str(d / f"{i}.wav")
        (d / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
        (d / "tts.ps1").write_text(_PS_SCRIPT, encoding="utf-8")
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(d / "tts.ps1"), str(d / "jobs.json")],
            check=True,
        )
        return [sf.read(j["path"], dtype="float32")[0] for j in jobs]


def trim(x: np.ndarray, thr_db: float = -45.0) -> np.ndarray:
    """Cut leading/trailing near-silence (TTS pads each utterance)."""
    env = 20 * np.log10(np.abs(x) + 1e-9)
    idx = np.flatnonzero(env > thr_db)
    return x[idx[0] : idx[-1] + 1] if len(idx) else x[:0]


def make_clips(rng: np.random.Generator, n_clips: int = 10) -> None:
    OUT_CLIPS.mkdir(parents=True, exist_ok=True)
    plans = []
    jobs = []
    for c in range(n_clips):
        k = rng.integers(4, 8)
        idx = rng.choice(len(PHRASES), size=k, replace=False)
        voice, rate = VOICES[c % 2], int(rng.integers(-2, 2))
        plans.append([(len(jobs) + n, rng.uniform(0.25, 1.8)) for n in range(k)])
        jobs += [{"text": PHRASES[i], "voice": voice, "rate": rate} for i in idx]
    audio = [trim(a) for a in tts_batch(jobs)]

    rows = []
    for c, plan in enumerate(plans):
        gain = 10 ** (rng.uniform(-14, -2) / 20)  # soft and loud speakers
        parts, t = [np.zeros(int(rng.uniform(0.5, 1.2) * SR))], 0.0
        t = len(parts[0]) / SR
        name = f"clip_{c:02d}.wav"
        for j, gap in plan:
            seg = audio[j] / (np.abs(audio[j]).max() + 1e-9) * 0.7 * gain
            rows.append((name, round(t, 3), round(t + len(seg) / SR, 3)))
            parts += [seg, np.zeros(int(gap * SR))]
            t += (len(seg) + int(gap * SR)) / SR
        parts.append(np.zeros(int(rng.uniform(0.5, 1.0) * SR)))
        x = np.concatenate(parts)
        x += rng.normal(0, 10 ** (-70 / 20), len(x))  # mic self-noise, not digital silence
        sf.write(OUT_CLIPS / name, x.astype(np.float32), SR, subtype="PCM_16")

    with open(OUT_CLIPS / "vad_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "start_s", "end_s"])
        w.writerows(rows)


# --- Procedural noise --------------------------------------------------------
def _colored(rng: np.random.Generator, n: int, exponent: float) -> np.ndarray:
    """1/f^exponent noise (0 white, 1 pink, 2 brown)."""
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    x = np.fft.irfft(spec / f ** (exponent / 2), n)
    return x / np.std(x)


def _bursts(rng: np.random.Generator, n: int, on: tuple[float, float], off: tuple[float, float]) -> np.ndarray:
    env, i, state = np.zeros(n), 0, True
    while i < n:
        d = int(rng.uniform(*(on if state else off)) * SR)
        env[i : i + d] = 1.0 if state else 0.05
        i, state = i + d, not state
    k = int(0.03 * SR)
    return np.convolve(env, np.ones(k) / k, mode="same")


def drilling(rng, n):
    t = np.arange(n) / SR
    f0 = 140 + 8 * np.sin(2 * np.pi * 0.7 * t)  # motor whine with slight wobble
    phase = 2 * np.pi * np.cumsum(f0) / SR
    tone = sum(np.sin(h * phase) / h**0.7 for h in range(1, 25))
    grind = _colored(rng, n, 0) * (0.6 + 0.4 * np.sin(2 * np.pi * 30 * t))  # broadband bit noise
    return (tone / np.std(tone) + 0.8 * grind) * _bursts(rng, n, (1.5, 4.0), (0.5, 2.0))


def generator(rng, n):
    t = np.arange(n) / SR
    hum = sum(np.sin(2 * np.pi * 25 * h * t + rng.uniform(0, 6.28)) / h**0.5 for h in range(1, 40))
    return hum / np.std(hum) + 0.7 * _colored(rng, n, 1.5)


def traffic(rng, n):
    t = np.arange(n) / SR
    rumble = _colored(rng, n, 1.8) * (1 + 0.5 * np.sin(2 * np.pi * 0.1 * t))
    horn = np.zeros(n)
    for _ in range(max(1, n // (8 * SR))):
        s, d = rng.integers(0, n - SR), int(rng.uniform(0.3, 0.8) * SR)
        tt = np.arange(d) / SR
        horn[s : s + d] += 0.8 * (np.sign(np.sin(2 * np.pi * 420 * tt)) + np.sign(np.sin(2 * np.pi * 520 * tt)))
    return rumble + horn


def hammering(rng, n):
    x, i = np.zeros(n), int(rng.uniform(0, 0.5) * SR)
    tt = np.arange(int(0.25 * SR)) / SR
    while i < n:
        f = rng.uniform(900, 2500)
        hit = np.exp(-tt * 35) * (np.sin(2 * np.pi * f * tt) + 0.5 * rng.normal(size=len(tt)))
        m = min(len(hit), n - i)
        x[i : i + m] += hit[:m] * rng.uniform(0.6, 1.0)
        i += int(rng.uniform(0.35, 1.2) * SR)
    return x + 0.05 * _colored(rng, n, 1)


def crowd(rng, n):
    jobs = [{"text": " ".join(rng.choice(PHRASES, 6)), "voice": VOICES[i % 2], "rate": int(rng.integers(-3, 4))}
            for i in range(8)]
    out = np.zeros(n)
    for a in tts_batch(jobs):
        a = np.resize(trim(a), n)  # loop each talker to fill the duration
        out += np.roll(a, rng.integers(0, n)) / (np.std(a) + 1e-9)
    return out


def make_noise(rng: np.random.Generator, seconds: float = 60.0) -> None:
    OUT_NOISE.mkdir(parents=True, exist_ok=True)
    n = int(seconds * SR)
    for name in EVAL.noise_types:
        x = globals()[name](rng, n)
        x = x / np.abs(x).max() * 0.5
        sf.write(OUT_NOISE / f"{name}.wav", x.astype(np.float32), SR, subtype="PCM_16")


if __name__ == "__main__":
    rng = np.random.default_rng(EVAL.seed)
    make_clips(rng)
    make_noise(rng)
    print(f"wrote {OUT_CLIPS} and {OUT_NOISE}")
