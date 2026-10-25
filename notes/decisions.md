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
