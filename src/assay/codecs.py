"""Codec probes — search_replace, whole_file, json_object (spec §7).

Appliers ported from robigo's aider-format rules: SEARCH matches a
contiguous run of whole lines exactly, including indentation (the
robigo dry run measured a 7B failing precisely on unindented SEARCH).
Landing = the applier accepts the reply AND the applied result equals
the expected output, tolerating a single trailing-newline difference
and nothing else.

No grammar/JSON forcing anywhere: probes measure unforced behavior.
Unmeasured cells are ``Landing(lands=None, n=0)`` — never a zero that
was not measured.
"""

import json
from dataclasses import dataclass

from assay import fixtures
from assay.errors import BudgetExhausted

_SEARCH_MARKER = "<<<<<<< SEARCH"
_DIVIDER = "======="
_REPLACE_MARKER = ">>>>>>> REPLACE"

CODECS = ("search_replace", "whole_file", "json_object")
GRADES = ("tiny", "small", "medium")

# The fixed extraction prompt (plan Task 9). No format forcing, ever.
JSON_PROMPT = (
    "Return a JSON object with keys `name` (string), `count` (integer), "
    "and `tags` (array of strings) describing: three apples"
)

_CHARS_PER_TOKEN = 5  # sizing proxy for the budget charge (plan Task 9)
_MAX_TOKENS = {"search_replace": 256, "whole_file": 768, "json_object": 128}

_SEARCH_REPLACE_DIRECTIVE = (
    "Respond with exactly one SEARCH/REPLACE block in this exact format:\n"
    f"{_SEARCH_MARKER}\n"
    "(the exact lines to find, copied verbatim from the file)\n"
    f"{_DIVIDER}\n"
    "(the replacement lines)\n"
    f"{_REPLACE_MARKER}\n"
    "The SEARCH lines must match the file exactly, character for "
    "character, including indentation. Output nothing else."
)

_WHOLE_FILE_DIRECTIVE = (
    "Respond with the complete corrected contents of the file and "
    "nothing else. Do not add commentary."
)


@dataclass(frozen=True)
class CodecDirectives:
    """The presentation given to the model, per codec. Landing is a
    property of model x codec x directive x sampler (live validation,
    2026-08-12: qwen landed 0/15 under the built-in minimal directive
    where robigo's full-envelope presentation measured 100% on the same
    daemon) — so a consumer may supply the directive its application
    actually sends, and the profile's lens records which was used."""
    search_replace: str
    whole_file: str
    json_object: str


@dataclass(frozen=True)
class Landing:
    """Both landing lenses per cell, measured over the same replies.

    ``lands`` is byte-equality with the expected file (v1's lens: strict,
    measures compliance-with-incidentals on whole_file). ``lands_applies``
    is applies-and-parses (robigo stage 2's lens: the edit applied and
    the result is syntactically valid Python — semantic rightness is the
    caller's tests' job). For ``json_object`` the two lenses coincide by
    construction (validation IS the landing) and carry the same value.
    Both are None only when n == 0 — unmeasured, never fake zero."""
    lands: float | None
    lands_applies: float | None
    n: int


def landing_equal(applied: str, expected: str) -> bool:
    """Exact equality, tolerating a single trailing-newline difference."""
    if applied == expected:
        return True
    return applied + "\n" == expected or applied == expected + "\n"


def _parse_blocks(reply: str) -> list[tuple[list[str], list[str]]]:
    """All complete SEARCH/REPLACE blocks in reply, as line lists."""
    lines = reply.split("\n")
    blocks: list[tuple[list[str], list[str]]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != _SEARCH_MARKER:
            i += 1
            continue
        search: list[str] = []
        replace: list[str] = []
        j = i + 1
        while j < len(lines) and lines[j].strip() != _DIVIDER:
            search.append(lines[j])
            j += 1
        if j >= len(lines):  # no divider: malformed, not a block
            break
        j += 1
        while j < len(lines) and lines[j].strip() != _REPLACE_MARKER:
            replace.append(lines[j])
            j += 1
        if j >= len(lines):  # no closer: malformed, not a block
            break
        blocks.append((search, replace))
        i = j + 1
    return blocks


def apply_search_replace(original: str, reply: str) -> str | None:
    """Apply exactly one SEARCH/REPLACE block; None when it cannot land.

    SEARCH must match a contiguous run of whole lines in `original`
    exactly, including indentation. Zero matches, two or more matches,
    or any block count other than one -> None.
    """
    blocks = _parse_blocks(reply)
    if len(blocks) != 1:
        return None
    search, replace = blocks[0]
    if not search:
        return None
    original_lines = original.split("\n")
    span = len(search)
    matches = [
        i
        for i in range(len(original_lines) - span + 1)
        if original_lines[i : i + span] == search
    ]
    if len(matches) != 1:
        return None
    at = matches[0]
    new_lines = original_lines[:at] + replace + original_lines[at + span :]
    return "\n".join(new_lines)


def _strip_one_fence(text: str) -> str | None:
    """Interior of a single outermost code fence, or None when unfenced."""
    lines = text.strip().split("\n")
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return None


def apply_whole_file(reply: str) -> str | None:
    """The reply as the complete new file; one outermost fence stripped."""
    if not reply.strip():
        return None
    interior = _strip_one_fence(reply)
    return interior if interior is not None else reply


def validate_json_object(reply: str) -> bool:
    """Unforced JSON landing for the fixed extraction prompt.

    The reply (outermost fence stripped) must json.loads to a dict with
    `name` (string), `count` (integer), `tags` (array of strings).
    Extra keys are allowed. Never raises: a bad reply is data.
    """
    interior = _strip_one_fence(reply)
    text = interior if interior is not None else reply
    try:
        payload = json.loads(text)
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    name = payload.get("name")
    count = payload.get("count")
    tags = payload.get("tags")
    if "name" not in payload or not isinstance(name, str):
        return False
    if "count" not in payload or not isinstance(count, int) or isinstance(count, bool):
        return False
    if "tags" not in payload or not isinstance(tags, list):
        return False
    return all(isinstance(tag, str) for tag in tags)


DEFAULT_DIRECTIVES = CodecDirectives(
    search_replace=_SEARCH_REPLACE_DIRECTIVE,
    whole_file=_WHOLE_FILE_DIRECTIVE,
    json_object=JSON_PROMPT,
)
DEFAULT_PRESENTATION = "default-v1"


def _build_prompt(codec: str, filename: str, instruction: str,
                  original: str, directives: CodecDirectives) -> str:
    if codec == "json_object":
        return directives.json_object
    directive = getattr(directives, codec)
    return f"{instruction}\n\n{directive}\n\nHere is `{filename}`:\n{original}"


def _parses_as_python(text: str) -> bool:
    try:
        compile(text, "<fixture>", "exec")
    except SyntaxError:
        return False
    return True


def _score(codec: str, reply_text: str, original: str,
           expected: str) -> tuple[bool, bool]:
    """(byte_equality_landed, applies_and_parses_landed) for one reply."""
    if codec == "json_object":
        ok = validate_json_object(reply_text)
        return ok, ok
    if codec == "search_replace":
        applied = apply_search_replace(original, reply_text)
    else:
        applied = apply_whole_file(reply_text)
    if applied is None:
        return False, False
    return (landing_equal(applied, expected), _parses_as_python(applied))


def _cell(landed: int, landed_applies: int, attempted: int) -> Landing:
    if attempted == 0:
        # unmeasured: None, never fake zero
        return Landing(lands=None, lands_applies=None, n=0)
    return Landing(lands=landed / attempted,
                   lands_applies=landed_applies / attempted, n=attempted)


def probe_codecs(
    backend, meter, *, n_per_cell: int, seed_base: int = 500,
    directives: CodecDirectives | None = None,
) -> dict[str, dict[str, Landing]]:
    """Landing rates per codec x size grade (spec §7), both lenses.

    Every cell is measured with `n_per_cell` seeded probes; each reply
    is scored under byte-equality AND applies-and-parses (see Landing).
    ``directives`` substitutes the consumer's own presentation for the
    built-in one — the caller records which was used (run.py stamps
    provenance and the verdict lens). Budget exhaustion mid-run stops
    the matrix: completed and partial cells report what was measured;
    unattempted cells stay ``Landing(None, None, 0)``. Infrastructure
    errors propagate; a bad reply is data (a non-landing), never an
    exception.
    """
    directives = directives or DEFAULT_DIRECTIVES
    by_grade = {entry[0]: entry for entry in fixtures.EXPECTED}
    results: dict[str, dict[str, Landing]] = {
        codec: {grade: Landing(lands=None, lands_applies=None, n=0)
                for grade in GRADES}
        for codec in CODECS
    }
    seed = seed_base
    exhausted = False
    for codec in CODECS:
        if exhausted:
            break
        for grade in GRADES:
            if exhausted:
                break
            _, filename, instruction, original, expected = by_grade[grade]
            prompt = _build_prompt(codec, filename, instruction, original,
                                   directives)
            landed = 0
            landed_applies = 0
            attempted = 0
            for _ in range(n_per_cell):
                try:
                    meter.charge(max(1, len(prompt) // _CHARS_PER_TOKEN))
                except BudgetExhausted:
                    exhausted = True
                    break
                reply = backend.generate(
                    prompt, seed=seed, max_tokens=_MAX_TOKENS[codec]
                )
                seed += 1
                attempted += 1
                exact, applies = _score(codec, reply.text, original, expected)
                landed += int(exact)
                landed_applies += int(applies)
            results[codec][grade] = _cell(landed, landed_applies, attempted)
    return results
