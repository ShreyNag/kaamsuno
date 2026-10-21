"""tests/fixtures/mid_sentence_pause.wav: one sentence with a 1.5 s pause in the middle.

    "mujhe dedh sau bori cement" <1.5 s pause> "chahiye teesri manzil pe"

Speech timings go in mid_sentence_pause.json next to it. Windows only (SAPI TTS);
the WAV is committed, so tests don't need to regenerate it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.make_synthetic_dev import SR, tts_batch, trim

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
LEAD_S, PAUSE_S, TAIL_S = 0.6, 1.5, 4.0  # tail > Smart Turn's 3 s silence fallback


def main() -> None:
    a, b = (trim(x) for x in tts_batch([
        {"text": "mujhe dedh sau bori cement", "voice": "Microsoft Zira Desktop", "rate": 0},
        {"text": "chahiye teesri manzil pe", "voice": "Microsoft Zira Desktop", "rate": 0},
    ]))
    a, b = (x / np.abs(x).max() * 0.5 for x in (a, b))
    z = lambda s: np.zeros(int(s * SR), dtype=np.float32)  # noqa: E731
    x = np.concatenate([z(LEAD_S), a, z(PAUSE_S), b, z(TAIL_S)])
    x += np.random.default_rng(0).normal(0, 10 ** (-70 / 20), len(x)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    sf.write(OUT / "mid_sentence_pause.wav", x, SR, subtype="PCM_16")
    t1 = LEAD_S + len(a) / SR
    t2 = t1 + PAUSE_S
    meta = {"speech": [[LEAD_S, round(t1, 3)], [round(t2, 3), round(t2 + len(b) / SR, 3)]],
            "pause": [round(t1, 3), round(t2, 3)], "duration": round(len(x) / SR, 3)}
    (OUT / "mid_sentence_pause.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    main()
