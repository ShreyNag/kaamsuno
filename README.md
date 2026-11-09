# KaamSuno: a voice incident reporter for construction and factory sites

A worker reports a problem by speaking in Hindi, English or Hinglish, the way they would to a supervisor. KaamSuno
fills a structured incident report live while they talk. It asks only for what is missing or unclear, reads the
report back, takes corrections, and sends the confirmed report to the supervisor on Telegram (or WhatsApp). An
emergency ("wire se sparks aa rahe hain", "chot lagi hai") alerts the supervisor at once, before any questions.

Built for Gnani AI's Great Indian AI Internship Challenge on Gnani's **Prisma** (streaming STT) and **Timbre**
(TTS), with Pipecat.

```
"Teesri manzil pe dedh sau bori cement kam pad gaya hai, urgent hai"
  -> location 3rd floor · material shortage · cement · 150 bags · urgent      (filled live, nothing asked)
Bot: "Main report read-back kar raha hoon: 3rd floor, material ki kami, cement, 150 bags, bahut urgent. ..."
Worker: "nahi, chauthi manzil"   -> Bot: "Theek hai, location chauthi manzil. Sahi hai?"   -> "haan"
Bot: "Report confirm ho gayi. Aur kuch batana hai?"   -> "nahi"   -> Telegram: 📋 Incident report ...
```

## How it works

```
 mic (WebRTC)
   │
   ▼
 VAD ── Silero (default) | custom (adaptive energy gate + 3 spectral votes) | composite
   │
   ▼
 Prisma STT (streaming, partial + final segments) ──────────────┐
   │                                                            │ every final segment
   ▼                                                            ▼
 Turn controller (smart_turn_fillers)                 Interpreter (Groq gpt-oss-20b, JSON only)
   Smart Turn v3 tone  +  last 1-3 words                 emergency trigger words -> urgency=emergency at once
   ("umm", "aur", "ke paas", "chaar saat" → wait)        merge rules in code: evidence must be in the transcript,
   fast reply only if tone AND words say "finished"      no guessed location/quantity/ID, never downgrade
   │                                                     an emergency, corrections keep history
   ▼  turn end                                                  │
 Dialogue manager (templates only, no free LLM text) ◄─────────┘ report state
   emergency line → missing fields → read-back → corrections → "aur kuch?" → send
   │                                     │
   ▼                                     ▼
 Timbre TTS → speaker            Telegram / WhatsApp (Twilio) · reports/<id>.json · live page /report/<id>
```

- **No chat LLM speaks.** The LLM only extracts field updates as JSON. Everything the bot says is a template from
  the research team's dialogue scripts, in the worker's language (Hindi, Hinglish or English).
- **The report rules are enforced in code,** not only in the prompt (`app/interpreter/interpreter.py`).
- **Turn taking** (`app/turn/`) is a pure state machine tested on simulated timelines. All three modes
  (`fixed_timeout`, `smart_turn`, `smart_turn_fillers`) run through it, so they can be compared on the same logs.
- **Every choice and its reason** is in `notes/decisions.md`.

## Run it

Needs Python 3.12 and Windows, macOS or Linux.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # Pipecat is pinned to 1.7.0 (pipecat-gnani 0.5.12 breaks on 1.8+)
copy .env.example .env                            # then fill in the keys below
```

| `.env` key | for |
|---|---|
| `GNANI_API_KEY` | Prisma STT and Timbre TTS |
| `GROQ_API_KEY` | the interpreter (the free tier is enough; calls are paced from Groq's rate-limit headers) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | the supervisor's chat (@BotFather → `/newbot`; message the bot, then read `chat.id` from `https://api.telegram.org/bot<TOKEN>/getUpdates`) |
| optional: `NOTIFY_CHANNEL=whatsapp` + `TWILIO_*`, `SUPERVISOR_WHATSAPP_TO` | WhatsApp through the Twilio sandbox instead |

```powershell
.venv\Scripts\python -m app.notify                    # check the supervisor channel: sends a sample report
.venv\Scripts\python -m app.bot -t webrtc             # ~4 s self-test against Prisma, then serves on :7860
start http://localhost:7860/client                    # talk to the bot
start http://localhost:7860/report/latest             # the live report (open after the client connects)
```

**Or with Docker** (Docker Desktop; tested on Windows 11 with WSL 2): fill in `.env`, then

```powershell
docker compose up --build        # first build ~4 min; then open the same two pages as above
docker compose down              # stop
```

The image is Python 3.12.9-slim, CPU only. The Silero and Smart Turn models come inside the Pipecat wheel and are
loaded once at build time. WebRTC audio from the host browser reaches the container through Docker Desktop's NAT
(peer-reflexive ICE), so no UDP ports need publishing. If a browser won't connect, use the native route above.

Reports are saved to `reports/<incident_id>.json`. Each session's event log (every hop, with timings) goes to
`logs/<session>.jsonl`. Knobs such as `TURN_MODE`, `VAD_MODE` and the turn waits are in `app/config.py`, and each
can be overridden from the environment.

```powershell
.venv\Scripts\python -m pytest -q                                   # 449 offline tests, ~4.5 min
$env:RUN_LIVE="1"; .venv\Scripts\python -m pytest tests\test_e2e_replay.py   # 3 tests against live Prisma + Groq
```

## Results

All numbers are measured on our own recordings and logs. **Interim** means the data set is small or the policy
changed after the measurement; see the limits below.

**VAD** (`eval/vad_eval.py`; frame-level scores on held-out clips; noise mixed at 10 and 5 dB SNR; each VAD's
threshold tuned leave-one-clip-out). *Interim: one speaker, auto-labelled.*

| VAD | F1 | recall | false alarm | F1 (6 clips, 5 real noises incl. crowd and hammering) |
|---|---:|---:|---:|---:|
| **Silero** (live default) | **0.92** | 0.86 | 1.3 % | **0.94** (false alarm 2.7 %) |
| WebRTC | 0.90 | 0.87 | 23.6 % | 0.92 (25.5 %) |
| custom (adaptive gate + flatness, peak, ZCR votes) | 0.79 | 0.67 | 11.2 % | 0.82 (20.0 %) |
| composite (Silero AND custom) | 0.69 | 0.52 | 0.1 % | 0.72 (0.8 %) |

12 clips × drilling, generator, traffic, white, pink and 50 Hz hum. Our custom VAD beats Silero on false alarms
only in steady hum and white noise. Crowd babble is the hardest case: Silero's false alarms are 12–19 %, and part
of that is real background speech.

**Turn taking** (`eval/turn_delay.py`: the current policy replayed on every logged pause). *Interim.*

| | logged sessions | cut-ins (worker interrupted) then → now | end-of-turn delay p50 / p95 |
|---|---:|---:|---|
| tone + words rule, 3.5 s waits (2026-10-31) | 13 | 16 → 9 | 673 / 3,500 ms |
| current: 2.5 s waits, 3 s cap (demo setting) | 18 | 25 → 19 | 568 / 3,000 ms |

On the same 18 sessions, shortening the waits from 3.5 s to 2.5 s costs one cut-in (18 → 19).

**Latency** (`eval/latency_report.py`, live sessions, `smart_turn_fillers`). *Interim: most of these turns ran an
older turn policy; to be refreshed after the final rehearsals.*

| hop | n | p50 | p95 |
|---|---:|---:|---:|
| last speech → turn end | 104 | 557 ms | 3,004 ms |
| turn end → question chosen (Prisma tail + interpreter) | 88 | 665 ms | 1,210 ms |
| question → Timbre first audio byte | 83 | 250 ms | 364 ms |
| last speech → bot audio starts | 82 | 1,676 ms | 4,165 ms |

With `fixed_timeout` (700 ms), last speech → bot audio is 1,318 / 2,689 ms (n = 22), but it cuts workers off
mid-sentence.

**Barge-in** (the worker talks over the bot → the bot's audio stops): **39 ms p50, 67 ms p95** live (n = 10, every
one with an interruption), and 228–244 ms in the end-to-end test on real audio (`tests/test_barge_in.py`).

**Interpreter** (`eval/llm_compare.py`, gpt-oss-20b on Groq's free tier, DIA001–020 plus 50 held-out scenarios):
right next action on 60/68, interpreter call p50 843 ms / p95 1,415 ms. *Interim: run before the incident-type
and description fixes; the emergency item DIA014 has passed since (live check on Groq).*

**Tests:** 449 offline (VAD streaming = offline, turn timelines, dialogue flows, report rules, plain-word output,
Telegram/WhatsApp failure handling) plus 3 live end-to-end tests against Prisma and Groq, all passing.

## Limits (honest)

- **Small, one-speaker evaluation.** The VAD numbers come from 12 recordings by one speaker, auto-labelled. The
  recorder's noise gate removed most silence, and the long-pause clips we re-recorded were lost before use. So
  long-pause turn taking is tested on synthetic timelines and live logs, not on a labelled set.
- **Synthetic data was used only during development** (Windows TTS phrases and generated noise). No number above
  comes from it.
- **Background voices.** Crowd babble raises Silero's false alarms to 12–19 %. A nearby talker can open a turn or
  interrupt the bot; nothing yet tells the worker's voice from others.
- **Remaining cut-ins.** 19 cut-ins remain when the 18 logged sessions are replayed. In those, the sentence sounds finished in
  both tone and words ("dedh sau bori cement chaahiye") and the worker then adds more. A 2.5 s pause after "umm"
  now ends the turn (the demo trade-off).
- **Numbers.** Spoken numbers rely on Prisma's `transcribe` format and the LLM. The number normaliser tested
  against the 1,800 number pairs was not built.
- **Free-tier LLM.** About 5 interpreter calls a minute (8k tokens/min); a very fast talker can make the bot
  ask from a slightly stale report.
- **Emergency words are a keyword list** (sparks, aag, chot, gas leak, karant…). It is cautious by design, so
  expect some false alarms.
- WhatsApp sending is tested against a mocked Twilio only.

## Credits

- **Gnani.ai:** Prisma streaming STT and Timbre TTS, through `pipecat-gnani`.
- **Pipecat** (Daily): the real-time pipeline, WebRTC transport and the **Smart Turn v3** end-of-turn model.
- **Silero VAD.** WebRTC VAD (`webrtcvad-wheels`) for the comparison.
- **Groq** (free tier) serving OpenAI's open-weight **gpt-oss-20b**.
- **Research pack** (report schema, dialogue scripts, filler lists, number pairs, scenarios): Aastha, in `research/`.
- **Noise clips** (Freesound; evaluation only, not redistributed, not in the demo video): "hammering.wav" by
  jpberger (CC BY-NC 3.0), "Drilling-001.wav" by tosha73 (CC BY 4.0), "Paris_Porte_Maillot_Traffic.mp3" by
  Greg Baumont (CC BY-NC 4.0), plus CC0 clips by craigsmith (crowd) and felix.blume (generator).
