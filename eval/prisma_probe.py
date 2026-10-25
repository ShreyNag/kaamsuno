"""Stream WAVs straight to Prisma (no Pipecat) and print every raw message, per header combination.

    .venv/Scripts/python -m eval.prisma_probe                       # all variants, all tests/fixtures/*.wav
    .venv/Scripts/python -m eval.prisma_probe --variants a e --wav tests/fixtures/matlab_pause.wav

Variants (language and x-format as the bot sends them in every variant):
    a  no extra headers (Phase 1)
    b  + x-min-silence-ms
    c  + x-bias-list / x-bias-score
    d  + x-partials
    e  everything the bot sends now (prisma_headers())

Audio goes out as 20 ms int16 PCM chunks at real-time pace (what Pipecat sends),
followed by 2 s of silence; then we listen 3 s more. Costs a few seconds of Prisma
credit per file per variant.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from pipecat_gnani._common import GNANI_STT_WS_URL
from pipecat_gnani._sdk import _generate_request_id, sdk_headers, ws_header_kwargs
from websockets.asyncio.client import connect

from app import config
from app.audio_io import load_wav
from app.stt import prisma_headers

SR, CHUNK = config.SAMPLE_RATE, 320


def variant_headers(v: str) -> dict[str, str]:
    full = prisma_headers()
    pick = {
        "a": [],
        "b": ["x-min-silence-ms"],
        "c": ["x-min-silence-ms", "x-bias-list", "x-bias-score"],
        "d": ["x-min-silence-ms", "x-bias-list", "x-bias-score", "x-partials"],
        "e": list(full),
    }[v]
    return {k: full[k] for k in pick if k in full}


async def probe(wav: Path, extra: dict[str, str], verbose: bool) -> dict:
    headers = {
        "x-api-key-id": config.GNANI_API_KEY,
        "lang_code": config.LANGUAGE_CODE,
        "x-sample-rate": str(SR),
        "x-api-request-id": _generate_request_id(),
        "x-format": config.PRISMA.output_format,
        **sdk_headers(),
        **extra,
    }
    pcm = (np.clip(load_wav(wav), -1, 1) * 32767).astype(np.int16).tobytes() + bytes(SR * 2 * 2)  # + 2 s silence
    types: Counter[str] = Counter()
    finals: list[str] = []
    other: list[str] = []
    t0 = time.monotonic()
    async with connect(GNANI_STT_WS_URL, **ws_header_kwargs(headers), close_timeout=5) as ws:
        first = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if verbose:
            print(f"    [{0:5.2f}s] {json.dumps(first, ensure_ascii=False)[:300]}")

        async def send():
            for i, off in enumerate(range(0, len(pcm), CHUNK * 2)):
                await asyncio.sleep(max(0.0, t0 + i * CHUNK / SR - time.monotonic()))
                await ws.send(pcm[off : off + CHUNK * 2])

        async def recv():
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        types["<binary>"] += 1
                        continue
                    d = json.loads(msg)
                    kind = d.get("type", "?")
                    types[kind] += 1
                    if verbose:
                        print(f"    [{time.monotonic() - t0:5.2f}s] {json.dumps(d, ensure_ascii=False)[:300]}")
                    if kind in ("transcript", "final_transcript"):
                        finals.append(d.get("text") or d.get("output") or "")
                    elif kind == "error":
                        other.append(json.dumps(d, ensure_ascii=False))
            except Exception as e:  # noqa: BLE001 - report whatever closes the socket
                other.append(f"recv ended: {type(e).__name__}: {e}")

        rx = asyncio.create_task(recv())
        await send()
        await asyncio.sleep(3.0)
        rx.cancel()
        close = (ws.close_code, ws.close_reason)
    return {"connected": first, "types": dict(types), "finals": finals, "errors": other, "close": close}


async def main_async(args) -> None:
    wavs = [Path(w) for w in args.wav] or sorted(Path("tests/fixtures").glob("*.wav"))
    summary = []
    for v in args.variants:
        extra = variant_headers(v)
        print(f"\n=== variant {v}: extra headers {sorted(extra)}")
        for w in wavs:
            print(f"  -- {w.name}")
            try:
                r = await probe(w, extra, args.verbose)
            except Exception as e:  # noqa: BLE001
                r = {"types": {}, "finals": [], "errors": [f"{type(e).__name__}: {e}"], "close": None, "connected": None}
            print(f"    connected config: {json.dumps((r['connected'] or {}).get('config'), ensure_ascii=False)[:300]}")
            print(f"    message types: {r['types']}")
            print(f"    finals: {r['finals']}")
            if r["errors"]:
                print(f"    errors: {r['errors']}")
            summary.append((v, w.name, r["types"], len(r["finals"])))
    print("\n=== summary: variant, file, message types, n transcripts the plugin would read ('transcript' only)")
    for v, name, types, n in summary:
        print(f"  {v}  {name:28s} {types}  final-ish={n}  plugin-readable={types.get('transcript', 0)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list("abcde"), choices=list("abcde"))
    ap.add_argument("--wav", nargs="*", default=[])
    ap.add_argument("-v", "--verbose", action="store_true", help="print every raw message")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
