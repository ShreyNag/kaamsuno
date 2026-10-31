# Groq free-tier budget (live interpreter)

Limits from Groq's headers: 8000 tokens/min, 1000 requests/day. A call costs ~1600 tokens, so at most 5 calls in any minute (plus whatever is left in the bucket).

Policy: one call in flight, merge segments, min gap 1.5 s, flush at turn end. Replayed on 11 logged sessions.

Turns = worker turns with words in them (turn ends on noise alone make no call). Held = calls the pacer would hold for the token bucket (full at the start of the session).

| session | turns | segments | calls before (1/segment) | calls now | calls/turn | peak calls in 60 s | held | max hold s | min | interp p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261022-002642-41f54e | 1 | 7 | 7 | 7 | 7.00 | 7 ⚠ | 0 | 0.0 | 0.8 | 800 |
| 20261022-005428-33f441 | 27 | 34 | 34 | 34 | 1.26 | 10 ⚠ | 16 | 12.0 | 5.6 | 800 |
| 20261022-010216-5ee11b | 2 | 3 | 3 | 3 | 1.50 | 3 | 0 | 0.0 | 0.2 | 800 |
| 20261022-010428-02a4af | 7 | 9 | 9 | 9 | 1.29 | 9 ⚠ | 0 | 0.0 | 1.0 | 800 |
| 20261022-012259-603d66 | 32 | 43 | 43 | 43 | 1.34 | 8 ⚠ | 0 | 0.0 | 8.8 | 800 |
| 20261025-132137-6a01b0 | 7 | 10 | 10 | 10 | 1.43 | 6 ⚠ | 0 | 0.0 | 2.1 | 666 |
| 20261025-152347-ae4841 | 7 | 9 | 9 | 9 | 1.29 | 7 ⚠ | 0 | 0.0 | 1.3 | 676 |
| 20261025-155234-3c9570 | 6 | 6 | 6 | 6 | 1.00 | 6 ⚠ | 0 | 0.0 | 0.6 | 652 |
| 20261025-155355-d70b60 | 10 | 10 | 10 | 10 | 1.00 | 10 ⚠ | 1 | 1.8 | 1.0 | 711 |
| 20261025-155519-f798e8 | 4 | 8 | 8 | 8 | 2.00 | 8 ⚠ | 2 | 12.0 | 0.3 | 726 |
| 20261025-155613-19941d | 4 | 8 | 8 | 8 | 2.00 | 8 ⚠ | 2 | 9.8 | 0.4 | 672 |

**All sessions: 147 calls for 107 turns = 1.37 calls/turn** (was 1.37 with one call per segment).

10-turn session: ~14 calls = ~21,981 tokens; requests/day allow ~72 such sessions. ⚠ = a minute over 5 calls: the pacer holds the next call until the bucket refills, and the dialogue asks from the report as it is.

10 turns, 14 calls, one call every N seconds (the bucket starts full):

| a call every | calls held | longest hold s |
|---:|---:|---:|
| 6 s | 5 | 12.0 |
| 9 s | 0 | 0.0 |
| 12 s | 0 | 0.0 |
| 15 s | 0 | 0.0 |
