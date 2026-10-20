"""Condense the 16 base documents into data/proposals_main.json.

Per document: parsed source -> Claude compacter -> deterministic gates -> GPT rubric judge + GPT recognition
attack (Qwen3-14B gates; GPT advisory) -> on failure, feedback back to the compacter (up to MAX_REVISIONS revisions). Every attempt is logged
under runs/compact/<doc_id>/; the final draft per document is cached there, so reruns skip finished docs.

    uv run --group data python -m data.compact.parse               # once: PDFs -> runs/compact/parsed/
    uv run --group data python -m data.compact.run --docs D05,D14  # pilot
    uv run --group data python -m data.compact.run                 # all, then assemble the JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

from . import prompts as P
from .gates import distinctive, recognition_hits, run_gates, split_sections, surnames
from .parse import load_rows
from .tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "compact"
PARSED = RUNS / "parsed"
OUT = ROOT / "data" / "proposals_main.json"
META = ROOT / "data" / "proposals_main.meta.json"

COMPACTER = "anthropic/claude-sonnet-4.6"
JUDGE = "openai/gpt-5.4"
RECOGNIZER = "qwen/qwen3-14b"  # the panel's base model: its recognition is the gate
MAX_REVISIONS = 4
MAX_TRIMS = 3  # length-only shortening calls per attempt, before the judge
PASS_SCORE = 7
RUBRIC_KEYS = ["faithfulness", "science_kept", "welfare_kept", "no_review_data", "anonymization"]
SHUFFLE_SEED = 0
# Claude's provider content filter refuses these virology applications; the animal arm runs with 6.
EXCLUDED = {"D02", "D03"}
# Invented PI and institution per document, fixed so names are unique across the set.
FICTIONAL = {
    "D01": ("Dr. Tobias Engstrand", "Carrow Valley University School of Medicine"),
    "D04": ("Dr. Priya Lindqvist", "Ashgrove University Cardiovascular Institute"),
    "D05": ("Dr. Samuel Okafor-Reyes", "Northmere Medical University"),
    "D06": ("Dr. Ingrid Bellamy", "Kestrel Bay University Health Sciences"),
    "D07": ("Dr. Rafael Dunmore", "Lakemont University School of Medicine"),
    "D08": ("Dr. Elise Marchetti", "Pinecrest College of Veterinary Medicine"),
    "D09": ("Dr. Kenji Albright", "Corvane Therapeutics, Inc."),
    "D10": ("Dr. Leah Fontaine", "Brightwell Bioprocess, Inc."),
    "D11": ("Dr. Mei Castellano", "Redfern University School of Social Work"),
    "D12": ("Dr. Joanna Whitlock", "Highmoor University Cancer Center"),
    "D13": ("Dr. Daniel Osei", "Easton Ridge University School of Medicine"),
    "D14": ("Dr. Clara Wendt", "Thornbury Center for Childhood Hearing"),
    "D15": ("Dr. Victor Szabo", "Glenhaven University Medical Center"),
    "D16": ("Dr. Naomi Pruitt", "Larkspur University"),
}

load_dotenv(ROOT / ".env")
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                     timeout=900)


async def chat(model: str, messages: list[dict], max_tokens: int, extra: dict | None = None) -> str:
    """One completion with retries on rate limits, server errors and empty replies."""
    delay = 5.0
    for attempt in range(6):
        try:
            r = await client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens,
                                                     extra_body=extra or {})
            if not r.choices:  # OpenRouter can return an error payload with HTTP 200
                err: Exception = RuntimeError(f"no choices: {getattr(r, 'error', None)}")
            elif r.choices[0].finish_reason == "content_filter":
                raise RuntimeError(f"{model}: provider content filter refused this document")
            elif text := (r.choices[0].message.content or "").strip():
                return text
            else:
                err = RuntimeError(f"empty reply (finish_reason={r.choices[0].finish_reason})")
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as e:
            err = e
        except openai.APIStatusError as e:
            if e.status_code not in (408, 409, 429, 502, 503, 504, 529):
                raise
            err = e
        print(f"  [{model}] retry {attempt + 1}: {str(err)[:160]}")
        await asyncio.sleep(delay)
        delay *= 2
    raise RuntimeError(f"{model}: giving up after retries: {err}")


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def species_note(row: dict) -> str:
    return f" (species: {row['animal_species']})" if row["animals_used"] == "True" else ""


def institutions_for(stats: dict, doc_id: str) -> list[str]:
    """Regex-extracted cover-page institutions, minus generic fragments like 'Centre'."""
    insts = stats.get(doc_id, {}).get("institutions", [])
    return [re.sub(r"[\s.,]+$", "", i) for i in insts if distinctive(i)]


def strip_fences(text: str) -> str:
    return re.sub(r"^```\w*\n|\n```$", "", text.strip()).strip()


async def judge(draft: str, source: str, row: dict) -> tuple[dict, dict, dict]:
    """Rubric (GPT, sees the source), plus recognition attacks by GPT (advisory) and Qwen3-14B (gate)."""
    animal = row["animals_used"]
    rubric_msgs = [
        {"role": "system", "content": P.RUBRIC_SYSTEM},
        {"role": "user", "content": P.RUBRIC_USER.format(animal=animal, species=species_note(row),
                                                          source=source, draft=draft)},
    ]
    recog_msgs = [
        {"role": "system", "content": P.RECOGNITION_SYSTEM},
        {"role": "user", "content": P.RECOGNITION_USER.format(draft=draft)},
    ]
    extra = {"reasoning": {"effort": "medium"}}
    rubric, gpt_txt, qwen_txt = await asyncio.gather(
        rubric_call(rubric_msgs, extra), chat(JUDGE, recog_msgs, 32000, extra),
        chat(RECOGNIZER, recog_msgs, 8000, {"reasoning": {"enabled": True}}))
    return rubric, safe_json(gpt_txt), safe_json(qwen_txt)


async def rubric_call(messages: list[dict], extra: dict) -> dict:
    """Rubric judge, re-asked if the reply is not valid JSON with every score."""
    for _ in range(3):
        try:
            out = parse_json(await chat(JUDGE, messages, 48000, extra))
            if all(isinstance(out.get("scores", {}).get(k), int) for k in RUBRIC_KEYS):
                return out
        except (json.JSONDecodeError, ValueError):
            pass
    raise RuntimeError("rubric judge returned no valid JSON after 3 tries")


def safe_json(text: str) -> dict:
    try:
        return parse_json(text)
    except (json.JSONDecodeError, ValueError):
        return {"unparsed": text[:2000]}


async def trim(draft: str, messages: list[dict]) -> str:
    """Shorten an over-length draft before judging, so length is not traded against judge feedback."""
    for _ in range(MAX_TRIMS):
        n = count_tokens(draft)
        if n <= P.MAX_TOKENS:
            break
        words = len(draft.split())
        target = int(words * (P.MAX_TOKENS - 200) / n)
        msgs = messages + [{"role": "assistant", "content": draft},
                           {"role": "user", "content": P.TRIM_USER.format(tokens=n, words=words, target=target,
                                                                          limit=P.MAX_TOKENS)}]
        draft = strip_fences(await chat(COMPACTER, msgs, 8000))
    return draft


def assess(gate_fails: list[str], rubric: dict, recog_hits: list[str], advisory: dict,
           advisory_hits: list[str]) -> tuple[bool, float, str]:
    """Pass flag, a ranking score for picking the best failed attempt, and feedback text."""
    scores = rubric.get("scores", {})
    low = {k: scores.get(k, 0) for k in RUBRIC_KEYS if scores.get(k, 0) < PASS_SCORE}
    passed = not gate_fails and not low and not recog_hits
    rank = sum(scores.get(k, 0) for k in RUBRIC_KEYS) - 5 * len(gate_fails) - 10 * len(recog_hits)
    fb = []
    if gate_fails:
        fb.append("Automatic checks failed:\n" + "\n".join(f"- {f}" for f in gate_fails))
    if low:
        fb.append(f"Judge scores below {PASS_SCORE}/10: {low}.")
    if rubric.get("problems"):
        fb.append("Judge's fixes:\n" + "\n".join(f"- {p}" for p in rubric["problems"]))
    if recog_hits:
        fb.append("A model identified the real application from your draft (" + "; ".join(recog_hits) + "). "
                  "Remove or generalize the clues without dropping science or welfare details.")
    if advisory_hits and not passed:
        clues = "\n".join(f"- {c}" for c in (advisory.get("clues") or [])[:8])
        fb.append(P.ADVISORY_CLUES.format(guess="; ".join(advisory_hits), clues=clues))
    return passed, rank, "\n\n".join(fb)


async def process(row: dict, institutions: list[str], other_surnames: list[str], sem: asyncio.Semaphore,
                  force: bool) -> dict:
    doc_id = row["doc_id"]
    out_dir = RUNS / doc_id
    final_path = out_dir / "final.json"
    if final_path.exists() and not force:
        print(f"{doc_id}: cached")
        return json.loads(final_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    source = (PARSED / f"{doc_id}.md").read_text()

    # The source block is marked cacheable so revisions reuse it.
    pi_name, institution = FICTIONAL[doc_id]
    user = P.COMPACTER_USER.format(pi_name=pi_name, institution=institution, animal=row["animals_used"],
                                   species=species_note(row), source=source)
    system = P.COMPACTER_SYSTEM.replace("{banned}", ", ".join(other_surnames))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": user, "cache_control": {"type": "ephemeral"}}]},
    ]
    attempts = []
    async with sem:
        for k in range(MAX_REVISIONS + 1):
            t0 = time.time()
            draft = strip_fences(await chat(COMPACTER, messages, 12000, {"reasoning": {"max_tokens": 4000}}))
            draft = await trim(draft, messages)
            fails = run_gates(draft, source, row, institutions, pi_name, other_surnames)
            rubric, gpt_guess, qwen_guess = await judge(draft, source, row)
            hits = recognition_hits(qwen_guess, row, institutions)
            gpt_hits = recognition_hits(gpt_guess, row, institutions)
            passed, rank, feedback = assess(fails, rubric, hits, gpt_guess, gpt_hits)
            rec = dict(attempt=k, draft=draft, tokens=count_tokens(draft), words=len(draft.split()),
                       gate_fails=fails, rubric=rubric, recognition=qwen_guess, recognition_hits=hits,
                       gpt_recognition=gpt_guess, gpt_recognition_hits=gpt_hits, passed=passed, rank=rank,
                       feedback=feedback, seconds=round(time.time() - t0, 1))
            attempts.append(rec)
            (out_dir / f"attempt_{k}.json").write_text(json.dumps(rec, indent=2))
            print(f"{doc_id} attempt {k}: passed={passed} tokens={rec['tokens']} gates={len(fails)} "
                  f"scores={rubric.get('scores')} qwen_recog={hits} gpt_recog={gpt_hits}")
            if passed:
                break
            messages += [{"role": "assistant", "content": draft},
                         {"role": "user", "content": P.REVISE_USER.format(feedback=feedback)}]

    best = next((a for a in attempts if a["passed"]), max(attempts, key=lambda a: a["rank"]))
    final = dict(doc_id=doc_id, status="passed" if best["passed"] else "needs_review",
                 attempt=best["attempt"], n_attempts=len(attempts), draft=best["draft"], tokens=best["tokens"],
                 scores=best["rubric"].get("scores"), gate_fails=best["gate_fails"],
                 recognition_hits=best["recognition_hits"], gpt_recognition_hits=best["gpt_recognition_hits"],
                 compacter=COMPACTER, judge=JUDGE, recognizer=RECOGNIZER)
    final_path.write_text(json.dumps(final, indent=2))
    return final


def to_record(final: dict, row: dict, pid: str) -> dict:
    draft = final["draft"]
    title = re.search(r"^TITLE:\s*(.+)$", draft, re.M)
    return dict(pid=pid, title=title.group(1).strip() if title else "",
                abstract=split_sections(draft).get("ABSTRACT", ""), full_text=draft,
                animal=row["animals_used"] == "True")


def assemble(rows: list[dict]) -> None:
    """Write proposals_main.json (shuffled pids, so P-number does not reveal animal use) and a meta file."""
    finals = {r["doc_id"]: RUNS / r["doc_id"] / "final.json" for r in rows}
    missing = [d for d, p in finals.items() if not p.exists()]
    if missing:
        print(f"Not assembling: missing finals for {missing}")
        return
    order = [r["doc_id"] for r in rows]
    random.Random(SHUFFLE_SEED).shuffle(order)
    by_id = {r["doc_id"]: r for r in rows}
    records, meta = [], []
    for i, doc_id in enumerate(order, 1):
        final = json.loads(finals[doc_id].read_text())
        records.append(to_record(final, by_id[doc_id], f"P{i}"))
        meta.append(dict(pid=f"P{i}", doc_id=doc_id, animal=by_id[doc_id]["animals_used"] == "True",
                         **{k: final[k] for k in ("status", "attempt", "n_attempts", "tokens", "scores",
                                                  "gate_fails", "recognition_hits", "gpt_recognition_hits",
                                                  "compacter", "judge", "recognizer")}))
    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(records)} proposals) and {META.relative_to(ROOT)}")
    for m in meta:
        print(f"  {m['pid']:>4} {m['doc_id']} animal={m['animal']!s:<5} {m['status']:<12} "
              f"tokens={m['tokens']} attempts={m['n_attempts']} scores={m['scores']}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", help="comma-separated doc_ids (default: all)")
    ap.add_argument("--force", action="store_true", help="redo docs that already have a final.json")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    rows = [r for r in load_rows() if r["doc_id"] not in EXCLUDED]
    stats = json.loads((PARSED / "stats.json").read_text())
    stats = {s["doc_id"]: s for s in stats} if isinstance(stats, list) else stats
    todo = [r for r in rows if not args.docs or r["doc_id"] in args.docs.split(",")]
    sem = asyncio.Semaphore(args.concurrency)
    real_surnames = sorted({s for r in load_rows() for s in surnames(r["pi"])})
    results = await asyncio.gather(*(process(r, institutions_for(stats, r["doc_id"]), real_surnames, sem,
                                             args.force) for r in todo), return_exceptions=True)
    for r, res in zip(todo, results):
        if isinstance(res, Exception):
            print(f"{r['doc_id']}: ERROR {res!r}")
    assemble(rows)


if __name__ == "__main__":
    asyncio.run(main())
