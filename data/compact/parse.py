"""Deterministic PDF -> markdown for the 16 base documents.

Run from the repo root: uv run --group data python -m data.compact.parse
Writes runs/compact/parsed/{doc_id}.md and runs/compact/parsed/stats.json.
"""
import csv
import json
import re
from pathlib import Path

import pymupdf

from data.compact.tokens import count_tokens

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "base_documents.csv"
OUT_DIR = ROOT / "runs" / "compact" / "parsed"
MIN_PAGE_CHARS = 50
INSTITUTION_PAGES = 5

INST_RE = re.compile(
    r"\b(University|Institute|College|Hospital|Center|Centre|School of Medicine|Inc\.?|LLC)\b")
GENERIC = re.compile(
    r"national institutes? of health|national institute of|department of health|"
    r"nih|public health service|center for scientific review|"
    r"national center for advancing|^\W*(the )?(university|institute|college|hospital|center|inc\.?|llc)\W*$|"
    r"\b(of|for) (the )?(application|award|grant)\b|office of|^\W*$", re.I)


def load_rows() -> list[dict]:
    """Rows of data/base_documents.csv."""
    with open(CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def clean(text: str) -> str:
    """Collapse runs of spaces, drop repeated long lines, and collapse 3+ blank lines.

    Some accessible PDFs (D15) repeat whole paragraphs many times in one page's text layer, both as
    separate lines and inside one very long line, so repeats are dropped per sentence.
    """
    seen, lines = set(), []
    for line in text.splitlines():
        line = re.sub(r"[ \t]{2,}", " ", line).rstrip()
        kept = []
        for sent in re.split(r"(?<=[.;:]) ", line):
            if len(sent) >= 40:
                if sent in seen:
                    continue
                seen.add(sent)
            kept.append(sent)
        if kept or not line:
            lines.append(" ".join(kept))
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_pages(row: dict) -> list[str]:
    """Cleaned text of each page of the row's PDF."""
    with pymupdf.open(ROOT / row["pdf_path"]) as pdf:
        return [clean(page.get_text("text")) for page in pdf]


def to_markdown(pages: list[str]) -> str:
    """Join pages with <!-- page N --> markers."""
    return "\n\n".join(f"<!-- page {i} -->\n{t}" for i, t in enumerate(pages, 1)) + "\n"


def parse_doc(row: dict) -> str:
    """Markdown text of one document."""
    return to_markdown(read_pages(row))


def find_institutions(pages: list[str]) -> list[str]:
    """Best-effort institution-like lines from the first pages, deduplicated."""
    found: dict[str, str] = {}
    for text in pages[:INSTITUTION_PAGES]:
        for line in text.splitlines():
            line = " ".join(line.split()).strip(" ,;:|")
            if not 6 <= len(line) <= 90 or not INST_RE.search(line) or GENERIC.search(line):
                continue
            found.setdefault(line.lower(), line)
    return list(found.values())


def doc_stats(row: dict) -> dict:
    """Parse one doc, write its markdown, and return its stats."""
    pages = read_pages(row)
    md = to_markdown(pages)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{row['doc_id']}.md").write_text(md)
    empty = [i for i, t in enumerate(pages, 1) if len(t) < MIN_PAGE_CHARS]
    return dict(doc_id=row["doc_id"], pages=len(pages), empty_pages=len(empty), empty_page_numbers=empty,
                chars=len(md), qwen3_tokens=count_tokens(md), institutions=find_institutions(pages))


def main() -> None:
    """Parse all rows, write stats.json, and print a summary table."""
    stats = []
    for row in load_rows():
        if not (ROOT / row["pdf_path"]).exists():
            print(f"MISSING {row['doc_id']}: {row['pdf_path']}")
            continue
        stats.append(doc_stats(row))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"{'doc_id':<7}{'pages':>6}{'empty':>7}{'chars':>10}{'qwen3_tokens':>14}")
    for s in stats:
        print(f"{s['doc_id']:<7}{s['pages']:>6}{s['empty_pages']:>7}{s['chars']:>10}{s['qwen3_tokens']:>14}")


if __name__ == "__main__":
    main()
