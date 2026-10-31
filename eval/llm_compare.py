"""Interpreter LLM comparison: next-action accuracy, field checks and call latency per model.

    .venv/Scripts/python -m eval.llm_compare                                   # 20b vs 120b
    .venv/Scripts/python -m eval.llm_compare --models openai/gpt-oss-20b --fresh

Items: Aastha's hand-written DIA001-020 plus 50 holdout scenarios from research/scenarios.jsonl
(the 4 holdout items with a real utterance outside the two templated categories, and every third
templated equipment_failure / quality_issue item). Each item runs like a live session: segments
through the Interpreter (real Groq calls, the production prompt, the bot's last question as
context), then one DialogueManager decision. That decision, as an action name ("ask_location",
"confirm_quantity", "emergency", "reread_location", ...), is scored against the actions our dialogue
policy should take for the item (`gold`: a set, since several moves can be right). Field checks
score the extraction itself.

Gold is in OUR policy's terms, not Aastha's labels: e.g. DIA005 "paanch ya pandrah bag" is
"clarify_quantity" in her set, but our dialogue asks the location first when it's missing, so
ask_location is right here and the quantity check is a field check. DIA010/011 ("keep_listening")
are turn-taking items: the turn classifier decides them, no LLM, so they're reported apart.

Calls are cached in data/llm_cache/<model>.json (prompt hash -> reply, ms): a re-run is free and
reports the latency measured when the call was made, and an interrupted run resumes; --fresh ignores
the cache. Uncached calls are throttled from Groq's rate-limit headers (free tier, see `cached`). Results go to
eval/out/llm_compare.md and .csv.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app import config
from app.interpreter.dialogue import DialogueManager, Question
from app.interpreter.groq import GroqPacer, RateLimited, groq_complete
from app.interpreter.interpreter import Interpreter, build_system_prompt, is_known
from app.interpreter.schema import PLACEHOLDER_TEMPLATES, load_schema
from app.turn.tail_classifier import TailClassifier

CACHE_DIR = config.DATA_DIR / "llm_cache"
OUT_DIR = config.ROOT / "eval" / "out"
MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")


@dataclass
class Item:
    id: str
    segments: list[str]  # the last one is scored; earlier ones set the scene (one turn each)
    gold: set[str]
    checks: dict[str, str] = field(default_factory=dict)  # field -> regex the value must match ("" = not known)
    note: str = ""


def dia_items() -> list[Item]:
    """DIA001-020 with gold actions in our policy's terms (see module docstring)."""
    loc_first = {"ask_location"}
    return [
        Item("DIA001", ["Machine band ho gayi hai."], loc_first),
        Item("DIA002", ["Pump kaam nahi kar raha."], loc_first),
        Item("DIA003", ["The conveyor has stopped."], loc_first),
        Item("DIA004", ["Machine ID shayad one zero seven hai."], loc_first | {"confirm_asset_or_equipment"},
             {"asset_or_equipment": r"107"}, "uncertain ID: kept, low confidence"),
        Item("DIA005", ["Paanch ya pandrah bag chahiye."], loc_first | {"opening"}, {}, "ambiguous quantity"),
        Item("DIA006", ["Twenty chahiye."], loc_first | {"opening"}),
        Item("DIA007", ["Kal hua tha, ya parson shayad."], loc_first | {"opening"}),
        Item("DIA008", ["It stopped at 10. No, maybe 12."], loc_first | {"opening"}),
        Item("DIA009", ["Nahi, 15 nahi, 50 kg."], loc_first | {"opening"}, {"quantity": r"^50$", "unit": r"(?i)kg|kilo"}),
        Item("DIA010", ["Umm... matlab... aur ek issue hai."], {"keep_listening"}, note="turn-taking"),
        Item("DIA011", ["Bas... ek aur cheez."], {"keep_listening"}, note="turn-taking"),
        Item("DIA012", ["Report: pump P07, Bay 2, oil leak. Submit?"], {"ask_urgency", "readback"},
             {"location": r"(?i)bay\s*2", "asset_or_equipment": r"(?i)p\W*07"}),
        Item("DIA013", ["Bay 2 mein pump P07 band ho gaya hai, bahut urgent hai.", "Nahi, Bay 2 nahi, Bay 3."],
             {"reread_location"}, {"location": r"(?i)bay\s*3"}, "correction at the read-back"),
        Item("DIA014", ["Wire se sparks aa rahe hain aur log paas hain."], {"emergency"}, {"urgency": r"emergency"}),
        Item("DIA015", ["The motor stopped; I don't know why."], loc_first),
        Item("DIA016", ["Naam nahi dena."], {"opening"}, {"reporter_name_or_id": ""}),
        Item("DIA017", ["Noise bahut hai, dobara bolta hoon."], {"opening", "repeat"}),
        Item("DIA018", ["I have to go, just send it."], {"opening"}),
        Item("DIA019", ["Machine band aa ji, gate de kol."], {"ask_urgency", "confirm_location", "ask_incident_type"},
             {"location": r"(?i)gate"}, "Punjabi 'de kol' = near"),
        Item("DIA020", ["Cement kam hai, around twenty bags."], loc_first,
             {"material": r"(?i)cement", "quantity": r"^20$", "unit": r"(?i)bag"}),
    ]


PLACES = ("Bay 2", "Bay 3", "Gate 1", "north side", "basement level", "road section 4", "store room", "mixing area",
          "plant room", "loading dock", "floor 5", "water tank")


def holdout_items(path: Path = config.RESEARCH_DIR / "scenarios.jsonl", n: int = 50) -> list[Item]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    hold = [r for r in rows if r["split"] == "holdout"]
    fixed = {  # the holdout items with a real utterance outside the two templated categories
        "SCN0001": ({"ask_location"}, {"asset_or_equipment": r"(?i)p\W*0?07|p\W*zero\W*seven"}),
        "SCN0006": ({"ask_urgency", "ask_incident_type"}, {"location": r"(?i)bay\s*3"}),
        "SCN0011": ({"ask_location"}, {}),
        "SCN0016": ({"ask_incident_type", "ask_location", "confirm_location"}, {}),
    }
    items = [Item(r["scenario_id"], [r["worker_utterance"]], *fixed[r["scenario_id"]], note=r["category"])
             for r in hold if r["scenario_id"] in fixed]
    templated = [r for r in hold if r["category"] in ("equipment_failure", "quality_issue")]
    for r in templated[:: 3][: n - len(items)]:
        u = r["worker_utterance"]
        place = next(p for p in PLACES if p.lower() in u.lower())
        asset_id = re.search(r"\b([A-Z])(\d\d)\b", u)
        checks = {"location": r"(?i)" + r"\W*".join(map(re.escape, place.split())),
                  "asset_or_equipment": rf"(?i){asset_id[1]}\W*{asset_id[2]}",
                  "incident_type": "^" + r["category"] + "$",
                  "urgency": ""}  # never stated in these: must stay unknown (or a marked guess)
        items.append(Item(r["scenario_id"], [u], {"ask_urgency"}, checks, r["category"]))
    return items


def action(q: Question) -> str:
    if q.reason == "missing":
        return f"ask_{q.field}"
    if q.reason == "missing_id":
        return "ask_asset_id"
    if q.reason == "low_confidence":
        return f"confirm_{q.field}"
    if q.reason == "reread":
        return f"reread_{q.field}"
    return {"conflict": "resolve_conflict"}.get(q.reason, q.reason)


def cached(model: str, fresh: bool):
    """The production completion function, behind a per-model disk cache that also keeps the latency.

    Free tier: calls go one at a time, paced by GroqPacer from Groq's own rate-limit headers (it waits
    for the token bucket instead of hitting 429s); a 429 waits its retry-after and retries. The cache
    is written after every call, so an interrupted run resumes where it stopped. Every limit event goes
    to eval/out/llm_compare_limits.jsonl."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{model.replace('/', '_')}.json"
    cache = {} if fresh or not path.exists() else json.loads(path.read_text(encoding="utf-8"))
    limits_log = (OUT_DIR / "llm_compare_limits.jsonl").open("a", encoding="utf-8")

    def log(event, **kw):
        limits_log.write(json.dumps({"t": round(time.time(), 3), "model": model, "event": event, **kw}) + "\n")
        limits_log.flush()

    pacer = GroqPacer(log=log)
    live = groq_complete(model, max_retries=0, pacer=pacer)  # waits happen here, never inside the latency
    calls: list[dict] = []

    async def complete(messages):
        key = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
        if key not in cache:
            err = None
            for attempt in range(config.LLM_EVAL_MAX_TRIES):  # rate limits and network errors: retried, never cached
                wait, why = pacer.wait_s()
                if wait > 0:
                    log("eval_wait", seconds=round(wait, 2), limit=why)
                    await asyncio.sleep(wait + 0.2)
                t0 = time.perf_counter()
                try:
                    reply = await live(messages)
                except RateLimited as e:  # pacer already holds the retry-after
                    err = str(e)[:200]
                    continue
                except Exception as e:
                    err = str(e)[:200]
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                cache[key] = {"reply": reply, "ms": round((time.perf_counter() - t0) * 1000), "error": None}
                path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                break
            else:  # scored as an empty update, counted in "errors"
                calls.append({"reply": "{}", "ms": None, "error": err})
                return "{}"
        calls.append(cache[key])
        return cache[key]["reply"]

    return complete, calls


def check(report, name: str, pattern: str) -> bool:
    e = report.get(name)
    if e is None:
        return True  # the schema has no such field
    if pattern == "":
        return not is_known(e)
    return is_known(e) and re.search(pattern, str(e["value"])) is not None


async def run_item(item: Item, schema, system: str, complete) -> tuple[str, dict[str, bool]]:
    interp = Interpreter(schema, complete, system_prompt=system, uncertainty_cue=None)
    dm, q, known = DialogueManager(), None, {}
    for seg in item.segments:
        interp.submit(seg)
        await interp.drain(timeout=None)  # rate-limit waits must not cut a call short
        now = {f.name: interp.report[f.name]["value"] for f in schema.llm_fields if is_known(interp.report[f.name])}
        changed = {n: v for n, v in now.items() if known.get(n) != v}
        known = now
        q = dm.decide(interp.report, schema, PLACEHOLDER_TEMPLATES, seg, changed, interp.take_touched())
        interp.assume_correction = dm.phase in ("readback", "confirmed")
        if not q.silent:
            interp.asked = {"field": q.field, "reason": q.reason, "question": q.text}
    return action(q), {n: check(interp.report, n, p) for n, p in item.checks.items()}


def turn_item(item: Item, clf: TailClassifier) -> str:
    cls = clf.classify(item.segments[-1]).tail_class
    return "keep_listening" if cls in ("hesitation", "continuation", "postposition", "number_in_progress",
                                       "id_in_progress") \
        else f"reply ({cls})"


async def evaluate(model: str, items: list[Item], fresh: bool) -> list[dict]:
    schema = load_schema()
    system = build_system_prompt(schema)
    complete, calls = cached(model, fresh)
    clf = TailClassifier()
    rows = []
    for item in items:
        n0 = len(calls)
        if item.note == "turn-taking":
            act, checks = turn_item(item, clf), {}
        else:
            act, checks = await run_item(item, schema, system, complete)
        item_calls = calls[n0:]
        rows.append({"model": model, "id": item.id, "set": "DIA" if item.id.startswith("DIA") else "holdout",
                     "note": item.note, "action": act, "gold": "|".join(sorted(item.gold)),
                     "ok": act in item.gold, "checks": checks, "checks_ok": all(checks.values()),
                     "ms": [c["ms"] for c in item_calls if c["ms"] is not None], "errors": [c["error"] for c in item_calls if c["error"]]})
        print(f"{model} {item.id}: {act} {'OK' if act in item.gold else 'want ' + '|'.join(sorted(item.gold))}"
              f" checks {checks}", flush=True)
    return rows


def summary(rows: list[dict], models: list[str]) -> str:
    out = ["| model | set | n | next action ok | field checks ok | calls | interp p50 ms | interp p95 ms | errors |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in models:
        for s in ("DIA", "holdout", "all"):
            rs = [r for r in rows if r["model"] == m and (s == "all" or r["set"] == s) and r["note"] != "turn-taking"]
            ms = [x for r in rs for x in r["ms"]]
            with_checks = [r for r in rs if r["checks"]]
            p = (lambda q: f"{np.percentile(ms, q):.0f}") if ms else (lambda q: "-")
            out.append(f"| {m} | {s} | {len(rs)} | {sum(r['ok'] for r in rs)}/{len(rs)} | "
                       f"{sum(r['checks_ok'] for r in with_checks)}/{len(with_checks)} | {len(ms)} | {p(50)} | {p(95)} | "
                       f"{sum(len(r['errors']) for r in rs)} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--fresh", action="store_true", help="ignore the call cache (re-measures latency)")
    ap.add_argument("--holdout", type=int, default=50)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    items = dia_items() + holdout_items(n=args.holdout)
    rows = [r for m in args.models for r in asyncio.run(evaluate(m, items, args.fresh))]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = summary(rows, args.models)
    diffs = ["| item | " + " | ".join(args.models) + " | gold |", "|---" * (len(args.models) + 2) + "|"]
    for item in items:
        acts = [next((r for r in rows if r["model"] == m and r["id"] == item.id), None) for m in args.models]
        if any(a and (not a["ok"] or not a["checks_ok"]) for a in acts):
            cells = [f"{a['action']}{'' if a['checks_ok'] else ' ✗' + ','.join(k for k, v in a['checks'].items() if not v)}"
                     if a else "-" for a in acts]
            diffs.append(f"| {item.id} {item.segments[-1][:45]} | " + " | ".join(cells) + f" | {'|'.join(sorted(item.gold))} |")
    turn = [r for r in rows if r["note"] == "turn-taking" and r["model"] == args.models[0]]
    md = "\n".join(["# Interpreter LLM comparison", "", table, "",
                    "Turn-taking items (tail classifier, no LLM): "
                    + ", ".join(f"{r['id']} {r['action']}" for r in turn), "",
                    "Items where a model missed the action or a field check (✗ = failed checks):", ""] + diffs)
    (OUT_DIR / "llm_compare.md").write_text(md + "\n", encoding="utf-8")
    with (OUT_DIR / "llm_compare.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n" + md)


if __name__ == "__main__":
    main()
