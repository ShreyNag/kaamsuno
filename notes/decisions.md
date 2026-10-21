# Decisions log

One line per non-obvious choice, with the reason. Newest at the bottom of each section.

## Setup (Phase 0, built 2026-10-21 because nothing existed)

- **Repo root = project folder, not a `voice-reporter/` subfolder.** Nothing else lives here, so one level less to type.
- **Synthetic dev set (`data/vad_dev_synth`, `data/noise_synth`) until Aastha's clips arrive.** Windows SAPI TTS phrases (en-US voices reading romanised Hinglish) with exact phrase-level labels, plus procedural drill/generator/traffic/hammering/crowd. Every script picks `data/vad_dev` + `data/noise` automatically once they exist (`app/audio_io.dev_dirs`). **No numbers from the synthetic set go in the final submission.**

- **Pipecat pinned to 1.7.0, not latest (1.13.0).**
  - pipecat-gnani 0.5.12 does `from pipecat.services.settings import _NotGiven`. Pipecat 1.8.0 renamed that to `NotGiven`, so on any Pipecat from 1.8.0 on, importing the plugin fails.
  - The plugin only declares `pipecat-ai>=0.0.50`, so pip happily installs the broken combination.
  - 1.7.0 has everything we use: turn strategies, `LocalSmartTurnAnalyzerV3`, `vad_analyzer` on the user aggregator, and observers. The only exception is the observer's `observe_every_push` flag; we dedupe by frame id instead.
- **TTS = timbre-v2.5 + Poorvi + language "auto".**
  - The installed SDK only accepts `timbre-v2.0` / `timbre-v2.5`. `timbre-v3.0` raises `ValueError` at construction.
  - `language` is accepted only on v2.5, from {auto, hi-IN, en-IN, …}. `hi-en` is rejected.
  - "auto" still needs a live listen to see how it reads romanised Hinglish.
- **Gnani STT plugin exposes only `language`, `format`, `itn_native_numerals`, `sample_rate`.**
  - Missing: `x-min-silence-ms`, `x-vad-threshold`, `x-min-speech-ms`, `x-bias-list`, `x-bias-score`, `x-partials`. Headers are built inline in `GnaniSTTService._connect`.
  - Plan (Day 3, not done yet): a small subclass in our repo that overrides `_connect` to add these headers. No fork.
- **Hop log = JSON lines via a Pipecat observer** (`app/logging_utils.py`), one file per session in `logs/`. An observer, so pipeline code doesn't change when we add hops.
- **`bot.py` forces UTF-8 stdout.** The Pipecat runner's banner crashes on the default Windows cp1252 console.

## VAD features (Day 1)

- **Frames: 25 ms / 10 ms hop, trailing partial frame dropped.** Dropping it keeps offline output identical to streaming, which can't see the future.
- **Frame energy on the unwindowed frame, in dBFS.** Simpler to reason about than windowed energy; the window only matters for the spectrum.
- **Running RMS (causal EMA, tau 1 s) is NOT applied to the gate.** The gate (percentiles of log energy), flatness, peak concentration and ZCR are all scale-invariant already, so normalising by a running RMS changes nothing for them. And normalising the energy itself would make long speech look like silence as the EMA catches up. Running RMS is computed, plotted, and available; the soft/loud-speaker problem is handled by the adaptive gate.
- **Absolute floor of -60 dBFS on top of the adaptive gate.** In a quiet room, the adaptive gate alone would open on mic self-noise fluctuations; -60 dBFS is well below any real phone speech.
- **Gate threshold has a minimum margin of 3 dB above the floor.** In steady noise with no speech, peak ≈ floor, so `floor + 0.3*(peak-floor)` sits inside the noise and the gate flickers on half the frames.
- **Gate warm-up uses an expanding window** (first 3 s use whatever history exists), not a fixed delay. Live calls must work from the first second.
- **Flatness is computed on a periodogram smoothed over 5 FFT bins (~156 Hz).** The raw single-frame periodogram of white noise has flatness ≈ exp(-γ) ≈ 0.56, not ~1, because each bin is chi-squared(2). Smoothing gets white noise to >0.8 while 156 Hz is still narrower than most formant structure.
- **Peak concentration uses mean power per band (a density), not band sum.** With octave-like bands, the summed 4–8 kHz band holds half the energy of white noise, so white noise scored 0.5 ("concentrated"). As a density, white noise scores ~1/6.
- **ZCR rule (Feature D):** `vote = (ZCR < 0.12)  OR  (ZCR > 0.25 AND full-band flatness < 0.5)`.
  - Low ZCR means voiced speech.
  - High ZCR means either a fricative ("s", "sh") or broadband hiss. Fricatives are tilted towards 4–8 kHz, so they are *not* flat over 0–8 kHz; hiss is. So high ZCR only counts as speech when the full band is not flat.
  - Full-band flatness (0–8 kHz) is used here instead of Feature B's 100–4000 Hz, because fricative energy lives above 4 kHz.
  - The thresholds are first guesses; Day 2 tunes them.
