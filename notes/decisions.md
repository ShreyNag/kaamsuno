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

## Phase 2 Day 1: turn controller (2026-10-25)

- **One stop strategy for all three modes.** Pipecat 1.7 ends turns through `BaseUserTurnStopStrategy`s inside the `UserTurnProcessor`. The processor hands every frame, in both directions, to its strategies. So a strategy sees VAD start/stop, input audio, Prisma segments and `BotStarted/StoppedSpeaking`, which is every input the controller needs. `app/turn/strategy.py` (`PolicyTurnStopStrategy`) is that strategy. It's a thin adapter around a pure state machine (`app/turn/controller.py`) that has no Pipecat and no clock, so simulated timelines can test it. `fixed_timeout` and `smart_turn` run through the same controller rather than Pipecat's built-in strategies, so every mode writes the same per-pause log and the modes can be compared on the same data. Rejected alternatives: a custom `BaseTurnAnalyzer` (it never sees transcripts or the dialogue context), and a processor that holds back `UserStoppedSpeakingFrame` (it fights the aggregator's own turn state).
- **Default `TURN_MODE=smart_turn_fillers`** (agreed with Shrey). `fixed_timeout` and `smart_turn` stay fully working for the evaluation.
- **Smart Turn's buffer is no longer cleared on COMPLETE, and it's trimmed to ~8.5 s instead.** Upstream `analyze_end_of_turn` clears the buffer when it says COMPLETE. We may keep waiting after a COMPLETE (a filler), and if the worker resumes, the next prediction should still see the whole turn. The buffer is cleared at the real turn end. Upstream only trims between turns, so `ThresholdSmartTurn.append_audio` keeps the last `max_duration_secs` (8 s) plus the pre-speech margin, which is all the model reads anyway.
- **Smart Turn says COMPLETE on the "matlab" fixture too (p=0.98 at the 1.5 s pause).** This is the same failure as Gate 1, and why hesitation/continuation waits ignore Smart Turn.
- **The Prisma segment budget counts from the VAD stop, not from the last speech** (agreed with Shrey). The default is **680 ms** = p90 of last speech → segment arrival (**883 ms**; p50 800 ms; n=92; all old logs) minus the VAD stop delay (200 ms).
  - Those logs ran with Prisma's default 500 ms min-silence, because our 250 ms header wasn't being sent before Gate 1. With 250 ms the lag should be ~250 ms shorter, so 680 is conservative.
  - Every `pause` record now has `segment_lag_ms`. Re-derive the p90 from the new live logs and lower the budget.
  - A budget from the last speech of 400 ms would almost always expire before the words arrive. Then Smart Turn's 0.98 would end the turn on "matlab".
- **A segment arriving after the decision re-decides** (`revisions` in the pause log). "bas" can still shorten the wait, and "umm" can still extend it, within the 2500 ms cap.
- **A segment that arrives before the VAD stop, but after the last speech, is the pause's tail.** Prisma's own VAD can close the segment before ours reports the stop.
- **VAD stop delay is already short.** Silero `stop_secs` = 0.2 s (`VAD_STOP_SECS`). The custom VAD's offline hangover (300 ms) isn't used live (see the VAD wrapper entry). The long waits belong to the turn policy now.
- **Lower caps at turn end, configurable:**
  - Prisma tail wait 1.5 s → **0.8 s** (`PRISMA_TAIL_TIMEOUT_MS`). In `smart_turn_fillers` the controller already waited for the segment before ending the turn, so this rarely runs.
  - Interpreter drain 3 s → **1.2 s** (`INTERPRETER_DRAIN_TIMEOUT_S`). After it, the question is asked from the report as it stands. A late result still merges and shows on the page; it just doesn't change the question already asked.
- **Worst-case last speech → question (before TTS), by mode:**

  | mode | turn wait | Prisma tail | interpreter | before (old caps) | now |
  |---|---|---|---|---|---|
  | smart_turn_fillers | 2.5 s cap | 0.8 s | 1.2 s | 7.0 s | 4.5 s |
  | smart_turn | 3.0 s silence fallback | 0.8 s | 1.2 s | 7.5 s | 5.0 s |
  | fixed_timeout | 0.7 s | 0.8 s | 1.2 s | 5.2 s | 2.7 s |

  Timbre's first byte comes on top.
- **Prisma partials: none have reached the pipeline.** Zero `stt_partial` events in all logs. The plugin turns `transcript` with `is_final: false` into an interim frame, and none came. But it silently drops any message *type* it doesn't know. `PrismaSTTService` now logs the first message of every type with its keys (`stt_msg_type`) and the counts at session end (`stt_msg_counts`). The next live session answers whether Prisma sends partials under another type. Until then, Day 2 interprets per final segment.
- **TTS latency: the old "turn_end → tts_first_byte ≈ 1.3 s" was not Timbre.** Those pre-Gate-1 sessions had the Groq chat LLM between turn end and TTS.
  - Transport is the plugin's persistent WebSocket (`GnaniTTSService`).
  - A `TTSSpeakFrame` goes to Timbre as one request; our questions are 1–2 short sentences.
  - New hops measure Timbre alone: `question → tts_first_byte`, and Pipecat's own TTS TTFB metric (`tts_ttfb` event). `latency_report.py` reports both, per turn mode.
- **The custom VAD runs live as a Pipecat `VADAnalyzer` (`app/vad/pipecat_vad.py`); `VAD_MODE=silero|custom|composite` all work.**
  - `voice_confidence` returns the share of the chunk's 10 ms frames that are speech candidates (gate AND ≥2 of 3 votes). Threshold `CUSTOM_VAD_CONFIDENCE`=0.5; chunks are 20 ms.
  - Pipecat's start/stop counters (`VAD_START_SECS`/`VAD_STOP_SECS` = 0.2 s) replace our offline `Hysteresis` (enter 30 ms / hangover 300 ms). The `stop_secs` on `VADUserStoppedSpeakingFrame` is then the true stop delay, which the turn controller subtracts to find the last speech.
  - Checked on the turn fixture: one stop per phrase, implying a last speech within 20 ms of the label. No mid-word stops, so no need for 0.3 s.
  - `min_volume=0` for custom. Our adaptive gate handles level, and Pipecat's volume floor would cut soft speakers.
  - **composite** = Silero's confidence, zeroed unless our VAD calls ≥50% of the chunk speech (Silero AND ours), on Silero's 512-sample chunks.
- **Streaming VAD was 1.8× slower than real time (36.7 ms per 20 ms chunk); fixed to 1.3 ms.** `StreamingVAD.push` recomputed the rolling gate percentiles over the whole 3 s history every chunk and kept the last two. `features.adaptive_threshold_last_db` computes only the new frames. It's the same math (expanding window = percentile of the values present), checked against the full version, and the streaming-equals-offline tests still pass. Composite costs 2.5 ms per 32 ms chunk.
- **VAD comparison tool (`eval/vad_eval.py`) built ahead of real data.** The Phase 1 comparison had never been run; `eval/out/vad_results.md` didn't exist.
  - Temporary dataset: Shrey's 12 clean recordings in `data/vad_dev_tmp/clean/`, labelled by `eval/auto_label.py`. Energy threshold = floor + max(10 dB, 0.2·(peak−floor)); word gaps < 0.3 s bridged; padding 0.15 s before / 0.2 s after. On the synthetic clips this catches 100% of true speech frames and marks 20–40% of true non-speech as speech, all next to speech. Precision and false alarm near segment edges are therefore pessimistic, equally for every VAD.
  - Noise: real CC0 clips in `data/noise/` plus generated white, pink and 50 Hz hum (with harmonics). SNR measured on labelled speech (`mix_noise.py`, seeded per clip/noise/SNR).
  - One knob per VAD, tuned leave-one-clip-out on F2 (recall weighted 2×); only held-out clips are scored.
  - Two views from the same confidences: **raw** (per-chunk decision) and **deployed** (through Pipecat's start/stop counters, i.e. what the turn logic sees). The offline replay equals the live Pipecat analyzers frame for frame (custom and composite, tested). That needed the live chunking rule: a 10 ms frame counts in the chunk where its last sample arrives.
  - `--clips` / `--noise` are arguments, so Aastha's labelled real clips rerun with one command. `eval/out/` is gitignored: results stay local.

## Live test 1 failure (2026-10-25): the bot heard nothing

- **Root cause: `x-partials: true` switches Prisma to a different message protocol, and pipecat-gnani 0.5.12 drops it.** `eval/prisma_probe.py` streamed the fixtures straight to Prisma with each header combination:
  - none, `x-min-silence-ms`, or + bias list: `speech_start` + `transcript`.
  - + `x-partials`: `speech_start` + `partial_transcript` (~every 450 ms) + `final_transcript`.

  The transcription text is identical either way. The plugin only reads `type == "transcript"`, so all 25 finals in the three live sessions were dropped: no captions, no segments, no interpreter calls, empty report. The headers came in with the Gate 1 commit and were never tested live, which is our miss. Prisma's `processing` message (our old "segment in flight" signal) doesn't exist in either protocol.
- **Fix: `PrismaSTTService` replaces the plugin's receive loop and reads both protocols.** `transcript`/`final_transcript` → `TranscriptionFrame`; `partial_transcript` → `InterimTranscriptionFrame` (a separate class, so the reporter and turn controller ignore partials until Day 2 uses them). Unknown types are logged (`stt_unknown_type`); errors are logged in full with close codes. "Prisma still owes words" is now tracked as utterances with a `speech_start` but no final.
- **Partials stay on.** They arrive 1–2 s before the final, which Day 2 (interpreter on partials) and the turn controller's tail wait both want. Every header can be switched off on its own: `PRISMA_SEND_MIN_SILENCE`, `_MIN_SPEECH`, `_VAD_THRESHOLD`, `_PARTIALS`, `_BIAS` = 0.
- **Startup self-test (`PRISMA_SELF_TEST`, on by default).** Before serving, the bot streams 2.5 s of `matlab_pause.wav` to Prisma with its exact headers, then silence, and exits with a clear error if no final transcript arrives within 5 s. It costs ~4.3 s at boot. The silence matters: Prisma only closes a segment on silence it receives, so the first version (which stopped sending after the clip) failed without headers, where Prisma's default min-silence is longer.
- **Second root cause: Aastha's schema loaded silently as "27 fields, 0 critical".** Her KaamSuno v1.1 format marks required fields in a top-level `required_before_submission` list, not per field. Our loader read `critical` per field, found none, and `next_question` returned "complete" on every turn. That's the repeated "Shukriya, saari jaankari mil gayi" line.
  - `parse_schema` now recognises her format (and our legacy one) and raises `SchemaFormatError` for anything else, at boot (`app/bot.py` loads the schema before serving).
  - Also an error: a required field that doesn't exist, an enum without `allowed`, or no critical fields after parsing.
- **Critical = `required_before_submission` minus `config.NOT_ASKED_REQUIRED`.** That removes `reporter_confirmation` (read-back) and `site_name` (from `SITE_NAME` in config: one site per demo), leaving location, incident_type, description and urgency.
  - Ask order is `config.ASK_ORDER`: location, incident_type, asset_or_equipment (equipment_failure only), description, quantity + unit (material_shortage only), urgency. A filled field is never asked, so urgency isn't asked when the words already made it clear.
- **Guard: the bot never says "complete" while a required field is empty.** The order is: emergency → opening question if nothing is captured yet → missing (ask order, then any other critical field) → conflict → low confidence → complete. A final check refuses "complete" if a critical field is empty. Tested per field, and with an ask order that forgets fields.
- **System fields are filled by code, never by the LLM (`config.SYSTEM_FIELDS`):**
  - at session start: incident_id = session id, reported_at, site_name, status=draft, reporter_confirmation=not_obtained
  - on every save: transcript, missing_fields, conflicts
- **Her validation rules, as far as code can check them:**
  - enums are checked against `allowed`
  - emergency is never downgraded
  - a quantity whose evidence isn't just digits keeps the phrase in `spoken_number_raw`
  - CONFLICT_FIELDS (location, quantity, unit, asset, material, date/time): a different value without a clear correction (the LLM's new `correction` flag) or refinement keeps both, and the dialogue asks which is right; "a correction replaces a value only after the worker's intent is clear"
  - a location's evidence must contain a place word (`config.LOCATION_CUES`, Latin and Devanagari)

  All her rules are also in the interpreter prompt.
- **Found by the live end-to-end replay (`tests/test_e2e_replay.py`, RUN_LIVE=1):**
  - (a) the LLM set `location = "crane hydraulic pipe area"` with evidence "crane hydraulic pipe". The evidence check passed because those words were said. This is why the place-word check exists.
  - (b) "it is urgent" → urgency `emergency`. Fixed in the prompt ("urgent/jaldi" = high; emergency = someone hurt or in immediate danger), not by downgrading in code: a missed injury costs more than a false alert.

  After both fixes the replay passed 3/3: location question asked, fields saved to `reports/<session>.json`.
- **Interpreter text normalisation was broken for Devanagari.** `re.sub(r"[^\w]+")` treats Devanagari vowel signs as non-word characters, so "मंजिल" → "म ज ल". Evidence matching still worked because both sides were mangled the same way, but any word-level check (place words, fillers) failed. It now strips only Unicode punctuation and symbols.
- **Aastha's fillers.json (entries format) is mapped by `function`** (`tail_classifier.FUNCTION_TO_CLASS`), per the integration plan:
  - hesitation / pause request → hesitation; continuation, self-correction, reformulation → continuation; possible turn completion → completion; confirmation/acknowledgement → confirmation; uncertainty → uncertainty. Everything else is ignored for turn-taking.
  - A spelling variant takes its seed's class ("Variant associated with 'umm'").
  - Her 776 generated candidates are off by default (`FILLERS_INCLUDE_GENERATED`).
  - Result with her file + ours: hesitation 78, continuation 96, completion 39, confirmation 39, uncertainty 43 (`python -m app.turn.tail_classifier` prints them with examples).
- **Two files of ours, in her format, marked for her review:**
  - `research/fillers_extra_ours.json`: 35 Latin tail words her seeds lack ("ke", "ki", "jaise", "kyunki", "because", "the", "aa", "woh", "kya bolte hain", "yaani", "bas itna hi"…). Grammatical words that end an unfinished sentence matter most for the tail rule.
  - `research/fillers_devanagari_extra.json`: 41 Devanagari core cues. Her file has 69 Devanagari entries, mostly regional, and `hi-IN` returns Devanagari.
- **A whole turn of just "haan" right after a confirmation question is a short answer, even though her data lists "haan"/"achha" as hesitation.** Her rule is "don't end a turn on 'haan', 'achha'… if the utterance appears unfinished". A lone "haan" answering "…sahi?" is finished. Without a confirmation question it stays hesitation (long wait).
- **Uncertainty cues lower confidence:** if a segment contains one ("shayad", "pata nahi", "lagbhag"…), the interpreter caps the confidence of every value from that segment below LOW_CONFIDENCE, so the dialogue confirms them. This follows her rule "when a worker is unsure about a critical number, ID, location or hazard, ask a targeted follow-up".
- **Question texts come from Aastha's dialogue_scripts.json** (`app/interpreter/templates.py`):
  - Generated scripts grouped by intent and language (hi, hinglish, en; `pa_hi` ignored for now), rotated per (key, language). Intent → key: opening, missing_location → location, missing_quantity → quantity, missing_asset_id → asset_or_equipment, readback, unclear_audio.
  - Hand-written DIA001–020 are not used as templates: each answers one specific utterance ("Where on site is the conveyor located?"). They're conversation tests instead.
  - The bot replies in the worker's style: Devanagari → hi; otherwise by the share of common English words (≥ 60% en, ≤ 10% hi, else hinglish), from what they said since the last question.
  - Keys her scripts don't cover (incident_type, description, urgency, unit, confirm/conflict slots) have our hi/en versions; the Hinglish text is the placeholder. Fallback order: language → hinglish → placeholder.
- **Fixes for Aastha (dialogue_scripts.json v1.1):**
  - The generated `bot_response` is a copy-paste error for `filler_pause`, `confirmation`, `rejection` and `emergency` in DIA027–030, DIA037–040, DIA047–050 and DIA057–060. Emergency says "detail clear nahi hui, repeat karo"; confirmation repeats the read-back line.
  - Ours for now: filler_pause → say nothing; rejection → "kaunsi detail galat hai?"; confirmation → "report bhej diya" (Phase 3); emergency → DIA014's pattern (safe distance, supervisor alerted, urgent report) in hi/hinglish/en.
  - `missing_location` asks for the *site* as well ("Kaunsi site aur exact area hai?"), but the site comes from config in our setup. Ask her for area-only variants.
- Still missing from her file, so placeholders remain: confirmation templates with slots (`{digits}`, `{value} {unit}`), the yes/no answer lists, the backchannel list and the emergency trigger words.

## Live test 2 (2026-10-25, fixed_timeout): "report taiyar hai" on every turn; barge-in

- **What the log showed:**
  - By turn 2 the report held location, incident_type=material_shortage, description, material and **urgency=high**, but the urgency was guessed: its evidence, "cement kam pad gaya hai", says nothing about urgency.
  - After "dedh sau bori cement" everything asked was filled, so "complete" was correct by the old rules. With no read-back state it repeated on every later turn.
  - The quantity question was asked twice in a row (the worker's speech in between produced no transcript).
  - `reporter_name_or_id = "employee id"`: the label stored as the value.
  - Prisma closed the session after 60 s without speech (`timeout_reason: idle`); the plugin reconnected on the next audio.
- **Counts as missing: null, empty, `UNKNOWN_VALUES` ("unknown", "pata nahi"…) or `inferred`.** The interpreter now returns `stated` per value. Code also marks urgency as inferred when its evidence contains no urgency word (`config.URGENCY_CUES`). An inferred value is kept and shown, but still asked, and never overwrites a stated one. A value made only of label words ("employee id") is dropped.
- **Dialogue manager (stateful, `dialogue.DialogueManager`):**
  - Required fields first (location, incident_type, description, urgency), then the type table in config (`TYPE_FIELDS`): equipment_failure → asset_or_equipment, plus one ID question if the value has no digit; material_shortage → material, quantity, unit; safety_hazard / injury_or_near_miss → people_at_risk.
  - Low-confidence values and conflicts are confirmed.
  - `MAX_QUESTIONS` = 3 caps everything **except missing required fields**, so the cap never forces a read-back while a required field could still be asked. Each field is asked at most twice; the second time rephrased ("_repeat").
  - **The same line is never spoken twice in a row:** rephrase or stay silent.
- **Minimal read-back pulled forward from Phase 3** (sending stays in Phase 3):
  - Format: her readback template split as intro + question, with the values in between, urgency last; a missing required field is read as "… nahi pata".
  - Then: yes → "report confirm ho gayi" (not "bhej diya"), `reporter_confirmation` = confirmed / corrected_then_confirmed. No → "kaunsi detail galat hai?". A changed field → re-read only the change ("Theek hai, location Bay 3. Sahi hai?"). New information after confirming → re-read it.
  - During the read-back a different value is always a correction (`Interpreter.assume_correction`), never a conflict.
  - Status: draft → awaiting_confirmation. "submitted" is Phase 3.
- **Barge-in: not broken by today's turn strategy.**
  - Every turn start still broadcasts an `InterruptionFrame` (enable_interruptions defaults to True on the VAD start strategy, unchanged since Gate 1).
  - The SmallWebRTC output awaits each chunk, so an interruption cuts within one chunk.
  - `tests/test_barge_in.py`: speech during bot audio → `InterruptionFrame` at the bot's output in **228–244 ms** in all three modes (real Silero, our strategy, the reporter).
  - Most likely what was heard: the bot *was* interrupted, but ~1.5 s later, at the end of the worker's turn, it said the same "taiyar hai" line again. That's fixed by the read-back and the no-repeat rule.
  - The hop log now records `interruption`, `bot_started` and `bot_stopped`, and `latency_report` reports barge-in reaction time (speech start during bot audio → bot audio stops) and any barge-in without an interruption frame. The next live log will confirm it.

## Live test 3 (2026-10-25, smart_turn_fillers): "clear nahi hai" loop, read-back corrections, cut-ins

- **"Clear nahi hai, phir batao" on most turns (root cause: short answers had no question context).**
  - Replayed the live short answers through Groq with the old prompt:
    - "abhi turant" → `{}`
    - "bhed do mein" → date_or_time "in two hours"
    - "location hai day teen" → location kept by the LLM, then dropped by our place-word rule ("location" wasn't a place word)
    - "dedh sau bori" → 150 bags, which worked
  - A second cause: the LLM saw the *guessed* urgency "high" as already set, so "abhi turant" and "urgency urgent hai" changed nothing, and the guess was never confirmed.
  - The old second ask of any field was "_repeat", whatever the worker had said.
  - Prisma also writes "Bay" as "be", "ve", "b", "bhed" or "day".
- **Fixes:**
  - The interpreter now gets `bot_just_asked` ({field, reason, question}) with every segment.
  - The LLM sees only stated values; guessed ones are sent as null.
  - When location was the question asked, the place-word rule is waived; the same goes for the urgency-cue rule. "location" and "jagah" are place words.
  - Prompt rules 10–11: short answers, read-back corrections, and Prisma's usual mishearings.
  - A field is asked at most twice. The second ask is "_repeat" only when nothing but fillers was heard, otherwise "_reask" ("Bas location bata dijiye"). After that the dialogue moves on: the field goes into `dialogue.unconfirmed` and `uncertainties`, and is read back as "… nahi pata".
- **Read-back:** every reply goes through the interpreter.
  - A field changed or re-stated → re-read only that field.
  - Yes → confirmed.
  - The reply names a field ("location galat hai", `config.FIELD_WORDS`) → "sahi location kya hai?", and the answer comes back as a change.
  - No → "kaunsi detail galat hai?".
  - "haan ya nahi" only when the reply has none of these.
- **Cut-ins:** `eval/replay_pauses.py` replays a log's pauses through the current controller. There were 3 audible cut-ins:
  - After "umm": Prisma sent no text at all, Smart Turn said 0.005, and the old 1200 ms wait cut in.
  - After "saat": a number with ID context, and the old 1500 ms wait cut in at a 1.9 s pause.
  - After "saal" (Prisma's spelling of "saat"): classed "other" and cut in at 1200 ms.
- **Turn fixes:**
  - No text by the segment budget means wait up to the cap (`no_text`); text arriving later re-decides.
  - Partial transcripts stand in for a late final, and a partial after the decision re-decides.
  - Waits are now: hesitation and continuation 3000 ms, number 2000 ms, cap 3500 ms.
  - **Deviation from the request:** ID read-outs ("ID chaar saat …") get their own class `id_in_progress` at 3000 ms. With 2000 ms, the required 2.5 s pause after "chaar saat" would end the turn. Quantities keep 2000 ms.
  - `NUMBER_SOUNDALIKES` ("saal", "no", …) count as digits only mid read-out.
- **Not fixed:** two early turn ends after complete sentences (Smart Turn 0.97–0.98; the worker added "ID C twelve" after 0.6–0.9 s). They were silent: the pending question is cancelled when the worker resumes, so the bot never spoke. Nothing in the text says more is coming.

## VAD comparison on real recordings (2026-10-25)

- **Data:** s01–s12 (one speaker), drilling, generator and traffic (CC0). `hammering.wav`, `crowd.wav` and `LICENSES.md` were not in `data/noise/`. White, pink and 50 Hz hum are generated.
  - `eval/prep_audio.py` checked every file and wrote 16 kHz mono 16-bit copies to `data/vad_dev/` and `data/noise_16k/`; the originals are untouched.
  - The noise files were 48 kHz / 44.1 kHz stereo, and `traffic.wav` is an MP3 with a .wav name.
- **Recording quality:** no clip is clipped (s12 peaks at −1.2 dBFS). None is quiet by the −40 dBFS rule; the quietest are s03 (p95 −37.8) and s01, not s06/s10.
  - **Every clip is noise-gated:** the floor between words is digital silence (−97 dBFS), so the recorder suppressed noise.
  - **The scripted 1.5–2 s pauses are 0.3–0.6 s of silence in the audio** (s02: a held "ummm", then about 0.3 s). Possibly the app trims silence.
  - s09 ends mid-speech.
  - These clips don't exercise long-pause turn-taking. Record that with the gate and noise suppression off.
- **Labeller (`eval/auto_label.py`):**
  - The threshold is never below `peak − 30 dB`. A floor at digital silence otherwise made clicks and breaths at −60 to −80 dBFS "speech".
  - Padding inside a gap is capped at gap/4 per side, so every pause of 300 ms or more stays non-speech. Gaps under 300 ms are bridged.
  - Plots are in `eval/out/plots/real/`.
- **Custom tuning:** five knobs (gate α, flatness max, peak-concentration min, min votes, chunk share), 72 combinations, leave-one-clip-out. Features are computed once per mix and only the votes are redone; the result is identical to `custom_vad` at the config values.
- **Result (held-out, raw view):**

  | VAD | F1 | recall | false alarm |
  |---|---:|---:|---:|
  | Silero | 0.92 | 0.86 | 1.3 % |
  | WebRTC | 0.90 | 0.87 | 23.6 % |
  | custom | 0.79 | 0.67 | 11.2 % |
  | composite | 0.69 | 0.52 | 0.1 % |

  - Every fold picked the most permissive end of each grid (Silero 0.3; custom α 0.2, min votes 1, share 0.5). The F2 objective and the padded labels favour recall.
  - Custom beats Silero on false alarms only in white noise and hum (0.000 vs 0.007–0.015). It loses on recall in every condition, and on false alarms in pink noise (40 %), generator (17–20 %), traffic (8–10 %) and drilling (4–5 %).
  - Composite cuts false alarms to about 0, but loses a third of Silero's recall: 0.52 vs 0.86 raw, 0.55 vs 0.84 deployed.
- **Recommendation: Silero for the live bot** (`VAD_MODE=silero`). The config default is still `composite`; not switched without asking.

## VAD decision (2026-10-31)

- **`VAD_MODE=silero` is the default** (agreed with Shrey), from the real-recording comparison: Silero F1 0.92 / false alarm 1.3 %, WebRTC 0.90 / 23.6 %, custom 0.79 (recall 0.67), composite 0.69 (recall 0.52). `custom` and `composite` stay buildable and in `eval/vad_eval.py` for the comparison.
- **Silero's threshold is its own knob, `SILERO_VAD_CONFIDENCE` = 0.5** (was `VAD_CONFIDENCE` = 0.7; renamed because it never applied to custom). Composite uses it too. Tuning picked 0.3 in every fold, but the auto-labels are padded and the F2 objective weights recall 2×, so it is biased low; 0.5 is Silero's own default and a middle ground between 0.3 and the old 0.7. Recheck after the re-recorded pause clips.

## Groq free tier: pacing the live interpreter (2026-10-31)

- **We stay on the free tier.** Limits are read from Groq's headers on every response (`x-ratelimit-*`) and from 429s (`retry-after`, or "try again in …" in the body), never hard-coded (`app/interpreter/groq.py`, `GroqPacer`). Every response logs `groq_limits` (limits, remaining, reset, prompt/completion tokens) and every 429 logs `groq_429` in the hop log.
- **What our key returned (gpt-oss-20b, 2026-10-31):** 1,000 requests/day, 8,000 tokens/minute. A call is ~1,600 tokens (1,435 prompt + ~170 out); Groq reports no cached prompt tokens. **Tokens/minute is the binding limit: ~5 calls/minute sustained.**
- **Live policy (`Interpreter`):**
  - At most one call in flight per session. Segments that arrive during a call go out together in the next one, at least `LLM_MIN_GAP_S` (1.5 s) after the previous call ended. Segments after a new bot question are not merged with older ones (a short answer is read against its question).
  - Turn end (`drain`) sends what is left at once, skipping the gap. If the interpreter is still busy at `INTERPRETER_DRAIN_TIMEOUT_S`, the dialogue asks from the report as it is (`interp_late` in the hop log); the late result still merges.
  - Each call sends only the stated report values (no system fields), the bot's last question and the new text. `earlier_segments` is gone from the payload and the prompt; evidence must come from the new text.
  - A 429 keeps the segments and retries them with the next ones after retry-after (`LLM_429_DEFAULT_WAIT_S` = 5 s when Groq gives none). `GroqPacer.wait_s` also holds a call when the headers say the token bucket can't take it, so 429s should be rare. Nothing crashes; the SDK's own retries are off so they can't hide 429s.
  - A side effect: a correction said within one batch ("teesri manzil… nahi, paanchvi") reaches the LLM as one text, so only the final value is stored and the history has no earlier value. Corrections across calls keep their history.
- **Calls per turn (`eval/groq_budget.py`, replaying the real segment and turn-end times of 11 logged sessions; `eval/out/groq_budget.md`):** 147 calls for 107 worker turns = **1.37 calls per turn**. Merging saved nothing on these logs: Prisma's final segments arrive further apart than call + gap.
  - A 10-turn session is ~14 calls ≈ 22k tokens. **It stays under the limit if those calls average ≥ ~9 s apart** (the bucket starts full: 8,000 + refill). At one call every 6 s, 5 of 14 calls are held, up to 12 s. Of the logged sessions, the three fastest (≥ 8 calls in a minute) would have had 1–2 calls held 2–12 s.
  - Requests/day allow ~70 such sessions a day, enough for testing and the demo, as long as batch evals don't run the same day.
  - If holds show up live, the next lever is a shorter system prompt (1,435 tokens, mostly the schema); not done, it needs an eval rerun.
- **Batch evals (`eval/llm_compare.py`):** DIA001–020 + 50 holdout (not all 200). Calls go one at a time, paced by the same `GroqPacer`; 429s wait retry-after (`LLM_EVAL_MAX_TRIES` = 8). The cache is written after every call, so an interrupted run resumes. Limit events go to `eval/out/llm_compare_limits.jsonl`.
- **First throttled eval run (gpt-oss-20b, new small payload):** 69 calls in 7.6 min, 65 pacer waits (380 s in all), six 429s (each retried after Groq's retry-after of 1–2 s), **0 errors** (was 5 unrecovered errors before). Next action 60/68 (was 56/68), field checks 24/56 (was 27/56), p50 843 ms. The six 429s came with ~1,900 tokens still "remaining" for ~1,600-token calls: Groq reserves room for the completion, so the pacer now holds until remaining ≥ last call × `LLM_TOKEN_MARGIN` (1.3).
- prisma_probe (variant e) and the live e2e replay (`RUN_LIVE=1`, 2/2) pass after the change.

## VAD rerun with the new noises; pause clips still missing (2026-10-31)

- **The re-recorded pause clips are not here.** `data/vad_dev_tmp/clean/s02, s03, s05, s07, s09, s12` are byte-identical to the 2026-10-25 copies in `data/vad_dev/` (same MD5). All six still fail both checks:
  - floor −96 to −97 dBFS (digital silence; the rule is above about −90)
  - longest labelled pause 0.19–0.63 s (planned 1.5–2 s): s02 0.19, s03 0.63, s05 0.39, s07 0.52, s09 0.38, s12 0.41
  - `audios.zip` is an empty 22-byte zip; no newer recordings anywhere under Desktop/Downloads/Documents/Music. Likely lost in the restart before they were copied in.
  - **Not used.** The turn-taking test (pause clips through the turn controller) is blocked on them.
- **Noises:** `hammering.wav` (44.1 kHz stereo, 12 clipped samples) and `crowd.wav` (48 kHz mono) arrived with `LICENSES.md`; 16 kHz mono copies are in `data/noise_16k/`. hammering and traffic are CC BY-NC: eval only, not in the demo video.
- **Interim rerun:** the 6 non-pause clips (s01, s04, s06, s08, s10, s11; `data/vad_dev_nopause/`), all 5 real noises + white/pink/hum, 10 and 5 dB, held-out only. Results in `eval/out/vad_nopause/`; `eval/out/vad_results.md` still holds the 12-clip run until the pause clips arrive.

  | VAD (raw, held-out) | F1 | recall | false alarm | before (12 clips, 3 real noises) |
  |---|---:|---:|---:|---|
  | Silero | 0.94 | 0.90 | 2.7 % | 0.92 / 0.86 / 1.3 % |
  | WebRTC | 0.92 | 0.89 | 25.5 % | 0.90 / 0.87 / 23.6 % |
  | custom | 0.82 | 0.72 | 20.0 % | 0.79 / 0.67 / 11.2 % |
  | composite | 0.72 | 0.56 | 0.8 % | 0.69 / 0.52 / 0.1 % |

  - The ranking is unchanged; Silero is still best on F1 and recall in every condition.
  - Silero's false alarms rose because of the new noises: crowd 12.5 % (10 dB) / 19.3 % (5 dB) and hammering 8.4 % at 5 dB; 0–1.6 % elsewhere. Crowd is babble (real voices), so part of that is the VAD hearing speech, as LICENSES.md says; report it apart.
  - Custom does worst on hammering at 5 dB (69 % false alarms): impulsive hits pass its gate.
  - Tuning again picked Silero 0.3 in every fold; the live value stays 0.5 (see the VAD decision).

## incident_type inferred, not asked (2026-10-31)

- **Why it was asked (session 20261031-181558):** the worker gave material, quantity, urgency and location, and the bot still asked "Kya hua hai?". No Groq call ever returned `incident_type` (the logged field lists; replayed requests match the logged sizes exactly). The prompt's rule 1, "Never guess", kept the LLM from choosing a category the worker never named. Had it returned one, it would be `"stated": false` → `inferred` → `is_known` false → asked as missing.
- **Fix:**
  - `config.INFERRED_OK_FIELDS` = incident_type, assigned_team: a guessed value counts as filled. They're categories picked from the description by design.
  - `config.VAGUE_VALUES`: incident_type "other" is asked as if missing. Below `LOW_CONFIDENCE` (0.6) it is confirmed as before.
  - The prompt (rule 1) tells the LLM to fill both as soon as the problem is clear, with `"stated": false`, 0.9 when it clearly fits.
  - The no-guessing rule stays for location, quantity, unit, asset ID, time and urgency.
- **Two more from the live test with the brief's sentence:**
  - Urgency: Groq flagged "urgent hai" → "high" as `"stated": false`. Urgency is now stated iff its evidence has an urgency word (the existing `URGENCY_CUES` rule), whatever the LLM's flag says.
  - A correction (`"correction": true`) always counts as stated. Groq flagged "nahi, chauthi manzil" `stated: false` at the read-back and the correction was dropped.
- **Hop log keeps each call's request JSON and raw reply** (`LLM_LOG_PAYLOADS`, on by default); the system prompt is logged once per session as a hash. Before this, a bad turn could only be explained by replaying it.
- **Open:**
  - ~~The LLM always marks its `description` summary `stated: false`, so the bot still asks "Thoda aur bataiye" after a complete report.~~ Fixed 2026-11-10: `description` is in `INFERRED_OK_FIELDS` (agreed with Shrey). Its evidence must still be in the transcript.
  - The live e2e test `test_live_short_answers_...` fails: the read-back comes without quantity ever being asked (material_shortage). Not investigated yet.

## Turn taking: tone AND words before a fast reply (2026-10-31)

- **Rule** (`app/turn/turn_policy.py`; every value in `config.TURN_POLICY`):

  | tail / signal | wait after the last speech |
  |---|---:|
  | completion words ("bas", "itna hi", "ho gaya", "that's it"), short answer to "... sahi?" | 400 ms |
  | incomplete ending: hesitation, connector, postposition | 3,500 ms |
  | number in progress (number word at the end, or after "ID"/"number"/"machine") | 3,000 ms |
  | Smart Turn ≥ 0.8 AND text arrived AND no incomplete ending | 400 ms (the only other fast path) |
  | text not arrived, Smart Turn 0.5–0.8, or no verdict | 2,500 ms |
  | Smart Turn < 0.5, words fine | 3,000 ms (**not in the brief**; tone says "not finished") |
  | no text AND Smart Turn < 0.5 (a lone "umm": Prisma sends nothing) | 3,000 ms (the longer of the two) |
  | hard cap | 4,000 ms |

  Speech resuming at any point cancels the pause, as before.
- **Keyword rule** (`app/turn/tail_classifier.py`): lists in config (`TURN_HESITATION`, `TURN_CONNECTORS`, `TURN_POSTPOSITIONS`, `TURN_COMPLETION`, `TURN_ID_CUES`), Latin and Devanagari, unioned with Aastha's fillers.json.
  - The tail is the last 1–3 words of the turn's text, not of the newest segment, so "ke" | "paas" split by Prisma still counts.
  - Postpositions are a new class: none were in any list before. That is why "teesri manzil pe" and "yaar 2nd floor pay" fell through to Smart Turn alone.
  - "pay"/"pai" are listed as Prisma's spellings of "pe".
  - "do"/"ek" (also give / a) count as numbers only after another number or an ID cue: "paani bhijava do" is a finished request. "machine"/"no" need an ID-like word after them ("kharab hai machine" is finished).
- **Why the old filler rule missed the live cut-ins** (20261031-181558, -181658, -181858):
  - "yaar 2nd floor pay" (ST 0.01): no postposition list → "other" → 1,200 ms; the worker resumed at 1.9 s. **Now held** (postposition, 3,500 ms).
  - "sunbe jittade", "ek saath do paanch ko paani chaahiye", "mujhe naa dhaai sau … nahi", "dedh sau bori cement chaahiye": the words read as finished and Smart Turn said 0.89–0.98. There was no filler to catch, and **the new rule still replies fast**: both signals say "finished". The worker then added more ("aur 3rd floor pe chaahiye", 0.7–1.3 s later). Catching these needs something other than the tail, e.g. a minimum report state before a fast reply.
- **Speed cost** (`eval/turn_delay.py` → `eval/out/turn_delay.md`; 13 sessions with Smart Turn verdicts):
  - End-of-turn delay p50 585 → 673 ms, p95 2,054 → 3,500 ms.
  - Cut-ins 16 → 9; 7 of the 16 live cut-ins are now held. The 9 left are "finished-sounding" sentences like those above.
- **Tests:**
  - `tests/test_turn_eager.py`: synthetic timelines, real config and lists, Smart Turn 0.98/0.65/none, 1/2/2.5 s pauses, final, partial-only or late text. Pauses after "aur", "ke paas", "umm", "chaar saat", "teesri manzil pe" (and Devanagari) never end the turn. "mixer kharab hai" (ST 0.98) and "bas itna hi" end ≤ 0.5 s after the speech when Prisma's text arrives at +0.45 s.
  - `tests/test_turn_modes.py`: the same on the real matlab_pause.wav recording (VAD, Smart Turn and turn strategy as the bot runs them; text injected).
  - The re-recorded s02–s12 pause clips are still missing, so they are not in these tests.

## After the confirmation: "aur kuch?", sending, a second issue (2026-11-10)

- **"Haan" to the read-back → "…confirm ho gayi. Aur kuch batana hai?"** (phase `confirmed`). Then:
  - no / "bas" → send → "Report supervisor ko bhej di" (status `submitted`). A failed or unconfigured send says so ("save ho gayi, lekin … nahi bhej paaye"); it never claims a send that didn't happen.
  - new details for the same report → re-read only them, then "aur kuch?" again.
  - a new issue (cue words "ek aur", "dusri", "another"…, or the incident type changed) → the confirmed report is sent, a new report `<session>-2` starts, and this turn's words are re-interpreted against it (one extra Groq call). The bot says "Pehli report … bhej di. Ab nayi report." plus its first question.
  - anything else → one prompt ("nahi toh bas boliye"), then send. The worker can't get stuck in "aur kuch?".
- **The confirmed report is snapshotted at "haan".** During the read-back the interpreter treats every different value as a correction, so a new issue's words overwrite the old report before we know it's a new issue. The snapshot is what gets sent.
- After sending, more speech with new content starts a new report; otherwise the bot stays silent.
- The page keeps one URL per session (`/report/<session>`) and shows the current report; each report has its own file `reports/<incident_id>.json`.
- **Supervisor channel = Telegram** (`app/notify.py`): one `sendMessage` POST, plain text (no Markdown, so no escaping), ≤ 4,000 chars. Order: urgency, type, location, item/asset, quantity + unit, description, report ID, then site/time, read-back state, unclear fields and the worker's words. `True` only when Telegram returns `ok`; the bot token is never logged. `python -m app.notify` sends a sample to check the setup.

## Emergency path (2026-11-10)

- **Trigger words OR the LLM flag.** `config.EMERGENCY_TRIGGERS` (sparks/chingari, aag/fire/dhuan, chot/injured/khoon/behosh, gas leak, karant/"current laga"/shock, collapse/dab gaya/fas gaya/gir gaya, blast; Latin and Devanagari), matched as whole words in `Interpreter.submit`, so urgency=emergency is set before any LLM call (stated, confidence 1.0, evidence = the trigger) and is never downgraded. The LLM can still flag one the words miss. Bare "current" is not a trigger ("current nahi hai" is a power cut).
- **The alert goes at once,** when the segment arrives (or the LLM's result merges), not at turn end: `send("emergency")`, once per report. The Telegram message says details are still being collected.
- **Safety line, then details.** At turn end the bot waits up to 3 s for the alert, then says "…Maine supervisor ko alert bhej diya hai" only if Telegram accepted it, otherwise "…supervisor ko abhi phone kariye". The first missing question follows in the same breath (location first), and the dialogue continues as usual: read-back, "aur kuch?", and the full report is sent as a second message. Before, an emergency skipped every question and went straight to the read-back.
- **DIA014 passes:** with the scripted worst cases (Groq returns no urgency, or "high") and once live on Groq (gpt-oss-20b, 2026-11-10): action `emergency`, urgency `emergency`. On 2026-10-31 Groq had returned nothing for urgency and the bot asked for the location.

## Demo polish: shorter turn waits (2026-11-10)

- **hesitation / continuation / postposition 3.5 → 2.5 s, number in progress 3.0 → 2.5 s, cap 4.0 → 3.0 s** (agreed with Shrey: long silences look bad on video). ID read-outs stay at 3.0 s (= the cap); the "tone says not finished" wait stays 3.0 s.
- **Cost, replayed on the same 18 logged sessions** (`eval/turn_delay.py`, 2026-10-31 + 2026-11-10 logs): cut-ins 18 (old waits) → 19 (new); p95 end-of-turn delay 3.5 → 3.0 s; p50 unchanged (564 ms).
- **The original brief's "a 2.5 s pause after 'umm' must not end the turn" no longer holds**: the wait is now exactly 2.5 s. Tests now hold pauses under 2.5 s (`test_pause_shorter_than_the_wait_does_not_end_the_turn`, the eager timelines at 1.0/2.0/2.4 s) and document that 2.5 s after "umm" ends the turn (`test_a_2_5_s_pause_after_umm_now_ends_the_turn`). ID read-outs still hold a 2.5 s pause.
- **"Aur kuch?" never sends on a ramble** (2026-11-10, replaces "prompt once, then send"). Sent only on a clear, short answer (≤ 5 words with nahi/bas/khatam/bhej/"that's all") or after `AUR_KUCH_SILENCE_S` (6 s) of silence, counted from when the bot stops speaking (Pipecat's `BotStoppedSpeakingFrame`) or, when the bot stayed silent, from the end of the worker's turn. An unclear reply gets one more prompt, then the bot stays quiet and listens. Any speech cancels the timer.
- **Plain words only** (`app/plain.py`, 2026-11-10): field names and enum values go through `field_word` / `say_value` wherever they're spoken (read-back, re-read, low-confidence and conflict questions, re-asks, "… nahi pata") or sent (Telegram, English). "material_shortage" is read back as "material ki kami" (supervisor: "Material shortage"), urgency "high" as "bahut urgent" ("Urgent"), "asset_or_equipment" as "machine". "other" and "unknown" are never said: they're left out of the read-back, and Telegram shows "Type: see description" / "Urgency: not given yet". `tests/test_plain_words.py` checks every enum value of the schema in all three languages.
- **WhatsApp as a second channel** (2026-11-10, optional): `NOTIFY_CHANNEL=telegram|whatsapp` (default telegram). `WhatsAppSender` posts the same `format_message` text to Twilio's Messages API (sandbox number works), cut to Twilio's 1,600-character limit; `True` only on a 2xx with a message sid, otherwise logged with Twilio's error code and the bot says it couldn't send. The Telegram sender is unchanged. Tested with a mocked Twilio only: no Twilio keys were set when this was written. Sandbox catch: the supervisor's phone must send "join <code>" to the sandbox number first, or Twilio returns error 63007/63015.
