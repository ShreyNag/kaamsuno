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

## LLM

- **Groq model = `openai/gpt-oss-20b`** (2026-10-22). `llama-3.3-70b-versatile` returned 404 model_not_found; our key's chat models are only gpt-oss-20b, gpt-oss-120b and qwen3.8-27b. Picked 20b for latency and JSON-mode support; 120b is the fallback if the interpreter's extraction quality is poor.

## Gate 1 live-test fixes (2026-10-22)

### Root cause 1: the bot was a plain chatbot
- `interpreter.py`, `dialogue.py` and the `/report` page **did not exist**. Days 3–4 had never been built, so `bot.py` was still Gnani's example: STT -> chat LLM (`GroqLLMService` + "site-safety assistant" system prompt) -> TTS. Every turn got a free-form LLM reply.
- **Fix: no chat LLM in the pipeline.** `mic -> VADProcessor -> PrismaSTTService -> UserTurnProcessor -> ReportProcessor -> TTS`. Every Prisma segment goes to the interpreter, which only updates state. At turn end, `dialogue.next_question` picks one template question. The interpreter LLM returns JSON only, and nothing it writes is ever spoken.
- **Interpreter calls run concurrently and merge in segment order.** A call only sees the report as it was when it was sent (plus the earlier segment texts), so latency doesn't stack up. Ordered commit keeps corrections deterministic.
- **The merge, not the LLM, enforces the report rules.** An update is dropped if its evidence string doesn't occur in the transcript (normalised: case/punctuation), if its value is null, or if it's outside the field's enum. A changed value pushes the old one to `history`.
- **Critical-field priority (placeholder):** incident_type, location, equipment_or_material, severity, employee_id. This is the schema's field order. Asking for an employee ID first felt wrong for a worker reporting an incident. Aastha's schema.json will override it.
- **Missing beats low-confidence.** We ask about any missing critical field before re-confirming a low-confidence one (< 0.6).
- **Report page = `/report/{id}` (HTML) + `/report/{id}/events` (SSE), `/report/latest` redirects.** Routes are added to the Pipecat runner's own FastAPI `app` (`pipecat.runner.run.app`), so it's the same server and port.

### Root cause 2: "the bot speaks the instant I stop"
Measured on the last 10 turns in `logs/` (`python -m eval.latency_report`):
- speech_end -> turn_end was **~790 ms on 8 of 10 turns**, and ~3000 ms on the other 2.
- **What ran (Pipecat 1.7 defaults):** `bot.py` passed no turn strategies, so the user aggregator used `UserTurnStrategies()` = start [VAD, Transcription] / stop [Smart Turn v3]. Smart Turn **was** loaded and running. `TURN_MODE` / `FIXED_TIMEOUT_MS` in config were never read, so `fixed_timeout` didn't exist.
- **790 ms decoded:** Silero's stop_secs is 0.2 s. Then Smart Turn said COMPLETE at the first 200 ms pause. Gnani never marks transcripts `finalized`, so Pipecat then waited out its STT safety net: `DEFAULT_TTFS_P99` (1.0 s) - stop_secs (0.2 s) = 0.8 s. The ~3000 ms turns are the ones Smart Turn called incomplete, which hit its 3 s silence fallback.
- **So the cut-off came from Smart Turn's verdict, not from wiring.** It said "complete" on mid-sentence pauses such as "mujhe aur chijen chaahiye jaise ki". On the test fixture it gives **p(complete)=0.98 after "mujhe dedh sau bori cement" (a 1.5 s mid-sentence pause)**, and the threshold knob can't fix a 0.98. Treat Smart Turn v3 as unreliable for Hinglish hesitation until measured; this is result (b)'s job.
- **Replies were not triggered per TranscriptionFrame.** The LLM ran only on turn stop. One real double-reply did exist: `TranscriptionUserTurnStartStrategy` let a Prisma segment arriving *after* turn end open a new turn, which then closed on the fallback timer (session 010428, turn 12: two turn_end events, two replies).
- **x-min-silence-ms=250 was never sent.** The plugin doesn't expose it (see Setup), so Prisma ran on its 500 ms default.
- **Fixes:**
  - **Only VAD starts a turn.** Transcripts never start or end one; a late segment just updates the report.
  - **`fixed_timeout`** = `SpeechTimeoutUserTurnStopStrategy(user_speech_timeout = FIXED_TIMEOUT_MS - stop_secs)`. Its timer starts at the VAD stop, which is already stop_secs late. So the turn ends FIXED_TIMEOUT_MS after the last speech.
  - **`smart_turn`** = `TurnAnalyzerUserTurnStopStrategy(ThresholdSmartTurn)`. It ends only on a Smart Turn COMPLETE (P > `SMART_TURN_THRESHOLD`) or its own `SMART_TURN_STOP_SECS` silence fallback.
  - **`wait_for_transcript=False` and `supports_ttfs=False` on Prisma**, which removes Pipecat's blind 0.8 s STT wait. The reporter instead waits for Prisma explicitly (below).
  - Tested by `tests/test_turn_modes.py` on `tests/fixtures/mid_sentence_pause.wav`.
- **Prisma tail wait at turn end.** Prisma sends `processing` when its VAD closes a segment and `transcript` when the text is ready. `PrismaSTTService` counts the two. At turn end the reporter waits until (last speech + min_silence_ms + 150 ms) and no segment is in flight, capped at `tail_timeout_ms` (1.5 s). Then it drains the interpreter (3 s cap) and asks. The question is never based on a report missing the last phrase.
- **A turn end can overtake a transcript.** `UserStoppedSpeakingFrame` is a SystemFrame, so it can be processed before a TranscriptionFrame still queued in the reporter. The tail wait covers this.
- **Prisma headers now sent** (`app/stt.py`): x-min-silence-ms, x-min-speech-ms, x-vad-threshold, x-partials, x-bias-list (JSON array, single words only), x-bias-score.
  - `PrismaSTTService._connect` is a copy of the 0.5.12 method with the extra headers. It's a subclass, not a fork, and it's pinned with the plugin.
  - Prisma's `connected` reply echoes `partials_enabled: true, partial_interval_ms: 450`. We haven't seen a partial message type on the wire yet; the plugin ignores unknown types.
- **Hop log fixes.**
  - Every VAD/turn event was logged twice: a broadcast frame is two frames with different ids, linked by `broadcast_sibling_id`. Now deduped.
  - The turn counter advanced on every BotStartedSpeaking (2–4× per reply); it now advances once per reply.
  - New events: `speech_end.stop_secs`, `turn_verdict` (Smart Turn p_complete), `interp_request/response`, `stt_tail_done`, `question`, `report_update`.
- **VAD_MODE=composite is still Silero.** The custom/composite Pipecat wrappers (Day 3) aren't built yet; startup warns about it.
- **Interpreter latency on Groq gpt-oss-20b (reasoning_effort=low): 1.0–4.5 s per call (3 calls).** It runs during speech, so most of that is hidden. But a call still in flight at turn end can hit the 3 s drain cap and the question goes out without it. Watch `interp_response.ms` in the logs.
