"""Deterministic checks run on every draft before the judge, plus the recognition-guess matcher."""

from __future__ import annotations

import re

from .prompts import HEADER_FIELDS, MAX_ABSTRACT_TOKENS, MAX_TOKENS, SECTIONS
from .tokens import count_tokens

# Case-insensitive terms that must never appear in a condensed proposal.
BANNED = [
    r"\bNIH\b", r"National Institutes? of Health", r"\bNIAID\b", r"\bNCI\b", r"\bNIA\b", r"\bNCATS\b",
    r"\bNINDS\b", r"\bNIDDK\b", r"\bNHLBI\b", r"\bNIDA\b", r"\bNIAAA\b", r"\bNIDCD\b", r"\bNICHD\b",
    r"\bNIMH\b", r"\bNSF\b", r"White Coat", r"\bWCW\b", r"\bFOIA\b", r"\(b\)\s*\(\d\)", r"redact",
    r"summary statement", r"resubmission", r"impact score", r"percentile", r"notice of award",
    r"study section", r"\bSBIR\b", r"\bSTTR\b", r"Halverson", r"Calder Medical",
    # Meta text about the source document, and hints of prior funding.
    r"\bsource (?:document|material|text|application|file)s?\b", r"\b(?:in|from) the source\b",
    r"cover[- ]page", r"\bnot (?:stated|available|provided|reported|given) in\b", r"\bOCR\b",
    r"\bdiscrepanc", r"competitive renewal", r"\brenewal\b", r"prior (?:funding )?(?:cycle|period|award)",
    r"previous (?:funding|award|grant) (?:cycle|period)",
]
# Case-sensitive NIH activity codes (R01, K08, F99, P01, U54 ...), not followed by more alphanumerics.
ACTIVITY_CODE = re.compile(r"\b[RKFPUT](?:0[1-9]|[1-9]\d)\b(?![A-Za-z0-9])")
GENERIC_INST_WORDS = {
    "university", "of", "the", "institute", "institutes", "college", "school", "medicine", "medical",
    "center", "centre", "hospital", "health", "sciences", "science", "inc", "llc", "and", "for", "research",
    "department", "dept", "national", "state", "foundation", "clinic", "system", "at", "in", "&",
}
VERBATIM_N = 15


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def split_sections(text: str) -> dict[str, str]:
    """Body text keyed by heading; headings must sit on their own line."""
    pattern = re.compile(r"^(%s)\s*$" % "|".join(re.escape(s) for s in SECTIONS), re.M)
    parts = pattern.split(text)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def surnames(pi_field: str) -> list[str]:
    """'Doe, Jane A' -> ['Doe']; multiple PIs separated by ';'."""
    out = []
    for name in re.split(r";|/| and ", pi_field):
        name = name.strip()
        if re.search(r"\b(?:Inc|LLC)\b", name):  # a company in the PI column: ban its distinctive first word
            out.append(name.split()[0].strip(","))
        elif name:
            out.append(name.split(",")[0].split()[-1] if "," in name else name.split()[-1])
    return [s for s in out if len(s) >= 2]


def grant_cores(grant: str) -> list[str]:
    """'R01AB123456' -> ['R01AB123456', 'AB123456', '123456']."""
    cores = {grant}
    m = re.search(r"([A-Z]{2})(\d{6})", grant)
    if m:
        cores |= {m.group(0), m.group(2)}
    return [c for c in cores if c and c != "—"]


def distinctive(inst: str) -> set[str]:
    return {w for w in _words(inst) if w not in GENERIC_INST_WORDS and len(w) > 2}


def verbatim_spans(draft: str, source: str, n: int = VERBATIM_N) -> list[str]:
    """Merged runs of >= n consecutive words copied from the source."""
    src = _words(source)
    grams = {tuple(src[i:i + n]) for i in range(len(src) - n + 1)}
    dw = _words(draft)
    spans, i = [], 0
    while i <= len(dw) - n:
        if tuple(dw[i:i + n]) in grams:
            j = i + n
            while j < len(dw) and tuple(dw[j - n + 1:j + 1]) in grams:
                j += 1
            spans.append(" ".join(dw[i:j]))
            i = j
        else:
            i += 1
    return spans


def run_gates(draft: str, source: str, row: dict, institutions: list[str], assigned_pi: str = "",
              other_surnames: list[str] = ()) -> list[str]:
    """Return a list of failure messages (empty means all gates pass).

    `assigned_pi` is the invented PI name the draft must use; `other_surnames` are the real PI surnames of
    every base document, banned so no invented name collides with a real PI elsewhere in the set.
    """
    fails = []
    if assigned_pi and assigned_pi.split()[-1] not in draft:
        fails.append(f"The PI must be the assigned invented name '{assigned_pi}'.")
    for s in other_surnames:
        if re.search(rf"\b{re.escape(s)}\b", draft):
            fails.append(f"'{s}' is a real investigator's surname; choose a different invented name.")
    n = count_tokens(draft)
    if n > MAX_TOKENS:
        words = len(draft.split())
        target = int(words * (MAX_TOKENS - 150) / n)
        fails.append(f"Too long: {n} tokens ({words} words); the limit is {MAX_TOKENS} tokens. At this text's "
                     f"density that means at most {target} words: cut {words - target} words. Compress wording "
                     f"and drop secondary detail first; keep aims, design, numbers and welfare details.")

    lines = draft.strip().splitlines()
    for i, field in enumerate(HEADER_FIELDS):
        if len(lines) <= i or not lines[i].startswith(field):
            fails.append(f"Line {i + 1} must start with '{field}'.")
    found = re.findall(r"^(%s)\s*$" % "|".join(re.escape(s) for s in SECTIONS), draft, re.M)
    if found != SECTIONS:
        missing = [s for s in SECTIONS if s not in found]
        fails.append(f"Section headings must be exactly {SECTIONS} once each, in order. "
                     f"Found {found}; missing {missing}.")
    sections = split_sections(draft)
    if (a := sections.get("ABSTRACT")) and (k := count_tokens(a)) > MAX_ABSTRACT_TOKENS:
        fails.append(f"ABSTRACT is {k} tokens; keep it under {MAX_ABSTRACT_TOKENS}.")
    for s in ("VERTEBRATE ANIMALS", "HUMAN SUBJECTS"):
        if s in sections and not sections[s]:
            fails.append(f"{s} is empty; write the details or an explicit 'none' statement.")

    for pat in BANNED:
        if m := re.search(pat, draft, re.I):
            fails.append(f"Banned term present: '{m.group(0)}'. Remove it.")
    if m := ACTIVITY_CODE.search(draft):
        fails.append(f"Grant activity code present: '{m.group(0)}'. Remove it.")
    for s in surnames(row["pi"]):
        if re.search(rf"\b{re.escape(s)}\b", draft):
            fails.append(f"Real PI surname present: '{s}'. Replace with the invented name.")
    for g in grant_cores(row["grant"]):
        if len(g) >= 6 and g in draft:
            fails.append(f"Real grant number present: '{g}'. Remove it.")
    if row["title"].lower() in draft.lower():
        fails.append("The original title is copied. Write a new title.")
    for inst in institutions:
        if inst.lower() in draft.lower():
            fails.append(f"Real institution present: '{inst}'. Replace with an invented one.")

    spans = verbatim_spans(draft, source)
    if spans:
        shown = "; ".join(f'"{s}"' for s in spans[:6])
        fails.append(f"{len(spans)} passage(s) of {VERBATIM_N}+ words are copied verbatim from the source; "
                     f"paraphrase them: {shown}")
    return fails


def recognition_hits(guess: dict, row: dict, institutions: list[str]) -> list[str]:
    """Which of the judge's identity guesses match the real application."""
    hits = []
    people = " ".join([guess.get("pi_name") or "", *(guess.get("other_people") or [])])
    for s in surnames(row["pi"]):
        if re.search(rf"\b{re.escape(s)}\b", people, re.I):
            hits.append(f"PI identified ({s})")
    g = (guess.get("grant_number") or "").replace(" ", "").upper()
    if g and any(c in g for c in grant_cores(row["grant"]) if len(c) >= 6):
        hits.append(f"grant number identified ({g})")
    gi = distinctive(guess.get("institution") or "")
    for inst in institutions:
        d = distinctive(inst)
        if d and d <= gi:
            hits.append(f"institution identified ({guess.get('institution')})")
            break
    return hits
