# Day 1: what each feature does on noise (before any decision logic)

**Caveat: synthetic data.** Speech is Windows TTS; noise is procedural. My "drill" is a 140 Hz motor whine with 24 harmonics, plus broadband grinding, in 1.5–4 s bursts. My "generator" is 25 Hz firing harmonics up to ~1 kHz, plus brown noise, running continuously. Real recordings will differ, so re-run `python -m eval.plots` and `python -m eval.feature_stats` when Aastha's audio lands.

Plots: `eval/out/plots/clip_XX__{clean,drilling_5dB,generator_5dB}.png`
Rates: `eval/out/feature_vote_rates.md`. Each cell is the fraction of speech frames voted speech / the fraction of non-speech frames voted speech.

| condition | A gate | B flatness | C peak conc | D ZCR rule |
|---|---|---|---|---|
| clean | 0.98 / 0.01 | 0.92 / 0.01 | 0.84 / 0.01 | 0.68 / 0.01 |
| drilling 5 dB | 0.82 / **0.45** | 0.72 / 0.08 | 0.81 / **0.93** | 0.36 / 0.12 |
| generator 5 dB | 0.76 / 0.14 | 1.00 / **1.00** | 0.89 / **1.00** | 0.93 / **1.00** |
| traffic 5 dB | 0.88 / 0.47 | 0.94 / 0.93 | 0.93 / 1.00 | 0.92 / 0.99 |
| crowd 5 dB | 0.81 / 0.17 | 0.98 / 1.00 | 0.87 / 0.92 | 0.76 / 0.67 |

**Clean:** every feature separates speech cleanly. Mic self-noise is white, so it is flat (about 0.85), has low concentration and a high ZCR. Speech is the opposite on all three.

**Drilling:**
- **Gate (A):** fails on *intermittent* noise. The 3 s window's 10th percentile lands on the drill-off gaps, so when the drill restarts it clears the threshold. 45% of noise frames pass the gate.
- **Flatness (B):** the most useful feature here. Drill sits around 0.45, speech dips to 0.1–0.3, and the 0.35 threshold lands between them.
- **Peak concentration (C):** useless here. The drill's harmonics put most of its energy into the 0–250 Hz band, so it looks "concentrated".
- **ZCR (D):** drill ZCR is about 0.2, between the voiced and fricative thresholds. So it mostly abstains, which is good, but it also misses half the speech.

**Generator:**
- **Votes B, C and D:** fail completely, voting speech on 100% of generator frames. The generator is low-frequency and harmonic, which is exactly what voiced speech looks like to these features. Flatness from 100 Hz upwards is close to 0 because energy below 1 kHz dominates. Concentration is about 0.8 in the 0–250 Hz band. ZCR is about 0.03, so it reads as voiced.
- **Gate (A):** the only feature that discriminates here, and only by level. At 5 dB the peak–floor gap is about 8 dB, so the threshold is barely above the noise.

**Traffic** (brown rumble) behaves like the generator. **Crowd** is real speech babble, so only level can separate it (expected; Silero will have the same problem).

## What this means for Day 2: needs your OK

As configured, "gate AND 2 of 3 votes" on a generator collapses to "gate only", because all three votes are always on. The common cause is that energy below roughly 300 Hz dominates every spectral feature, while speech intelligibility lives in roughly 300–3400 Hz. The fix stays inside the planned design (the same four features with configurable bands); it only changes which frequencies they look at:

1. High-pass at about 300 Hz (or apply pre-emphasis) before ZCR, so a low-frequency hum can't make every frame look "voiced"
2. Raise flatness `lo_hz` from 100 Hz to about 300 Hz
3. Compute peak concentration over the 250–4000 Hz bands only
4. Compute the gate on speech-band (300–3400 Hz) energy instead of full-band energy, which should help most at 5 dB on low-frequency noise. Separately, use a longer or lower-percentile floor so that drill-off gaps don't drag the floor down

I'd do 1–3 as part of Day 2 tuning, and try 4 as a measured A/B in the comparison table.
