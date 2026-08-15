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
from assay.stats import decided

_SEARCH_MARKER = "<<<<<<< SEARCH"
_DIVIDER = "======="
_REPLACE_MARKER = ">>>>>>> REPLACE"

CODECS = ("search_replace", "whole_file", "json_object")
GRADES = ("tiny", "small", "medium")

# The extraction directive + five task variants (v1.3: one prompt was
# sampler variance, not capability). No format forcing, ever.
JSON_DIRECTIVE = (
    "Return a JSON object with keys `name` (string), `count` (integer), "
    "and `tags` (array of strings) describing:"
)
JSON_TASKS = (
    "three apples",
    "two rusty bicycles leaning on a fence",
    "five copper coins from an old purse",
    "one chess board mid-game",
    "four rain boots by the door",
)
JSON_PROMPT = f"{JSON_DIRECTIVE} {JSON_TASKS[0]}"  # kept for fakes/back-compat

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
    json_object=JSON_DIRECTIVE,
)
DEFAULT_PRESENTATION = "default-v1"


def _build_prompt(codec: str, filename: str, instruction: str,
                  original: str, directives: CodecDirectives) -> str:
    if codec == "json_object":
        # instruction carries the task description for the json codec
        return f"{directives.json_object} {instruction}"
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


def _attempt_order(n_tasks: int, n_per_cell: int,
                   look_schedule: tuple[int, ...] | None) -> list[int]:
    """A cell's attempts as task indices, in the order they are sent.

    Fixed-n (no schedule): the v1.4 order, ``n_per_cell // n_tasks``
    reps of each task in turn (at least one each) — unchanged so quick
    and family runs replay exactly as before.

    Sequential (schedule given): round-robin, one rep across ALL the
    cell's heterogeneous tasks per round. Attempts still spread across
    tasks rather than redrawing one prompt (v1.3), the looks fall on
    round boundaries for a five-task cell (5/10/20/35 = rounds
    1/2/4/7), and a cell that stops early has sampled every defect
    class it could reach instead of exhausting the first one.
    """
    if look_schedule is None:
        reps = max(1, n_per_cell // n_tasks)
        return [task for task in range(n_tasks) for _ in range(reps)]
    cap = look_schedule[-1]  # the last entry is the cap; n_per_cell is moot
    return [attempt % n_tasks for attempt in range(cap)]


def _stop_count(codec: str, landed: int, landed_applies: int) -> int:
    """The count the stop test reads: the codec's VERDICT lens.

    ``json_object``'s two lenses coincide by construction (validation IS
    the landing). The patch codecs are graded applies-and-parses, so
    stopping on byte-equality would decide a cell on a lens no verdict
    uses — a reply that applies and parses but editorializes a comment
    is a landing for the verdict and must not end the cell early."""
    return landed if codec == "json_object" else landed_applies


def probe_codecs(
    backend, meter, *, n_per_cell: int, seed_base: int = 500,
    directives: CodecDirectives | None = None,
    look_schedule: tuple[int, ...] | None = None,
) -> dict[str, dict[str, Landing]]:
    """Landing rates per codec x size grade (spec §7), both lenses.

    Without ``look_schedule`` every cell is measured with `n_per_cell`
    seeded probes at fixed n. With one (``assay.stats.LOOK_SCHEDULE`` is
    the registered schedule), the cell samples round-robin and is
    examined ONLY at the schedule's look points, stopping at the first
    look where the Wilson-95 interval decides a rung on its verdict lens
    (``assay.stats.decided``); the last entry is the cap and
    ``n_per_cell`` is ignored. A cell that stops at n=5 differs from a
    fixed-n=5 cell only in the lens the caller stamps, so run.py records
    the stopping rule and n_used.

    Each reply is scored under byte-equality AND applies-and-parses (see
    Landing). ``directives`` substitutes the consumer's own presentation
    for the built-in one — the caller records which was used (run.py
    stamps provenance and the verdict lens). Budget exhaustion mid-run
    stops the matrix: completed and partial cells report what was
    measured (a cell stopped mid-schedule keeps its honest partial n);
    unattempted cells stay ``Landing(None, None, 0)``. Infrastructure
    errors propagate; a bad reply is data (a non-landing), never an
    exception.
    """
    if look_schedule is not None and not look_schedule:
        # An empty schedule is neither fixed-n nor sequential; coercing
        # it to either would hand the caller a lens that lies about how
        # the number was reached.
        raise ValueError("look_schedule must name at least one look point "
                         "(pass None for fixed-n sampling)")
    directives = directives or DEFAULT_DIRECTIVES
    by_grade: dict[str, list] = {g: [] for g in GRADES}
    for entry in fixtures.EXPECTED:
        by_grade[entry[0]].append(entry)
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
            # v1.3: a cell's attempts spread across HETEROGENEOUS tasks
            # (five defect classes on the grade's base module; five task
            # variants for json), not repeated draws of one prompt — the
            # v1/v2 sets measured sampler variance on a single fixture.
            if codec == "json_object":
                cell_tasks = [(None, task, None, None) for task in JSON_TASKS]
            else:
                cell_tasks = [(entry[2], entry[3], entry[4], entry[5])
                              for entry in by_grade[grade]]
            prompts = [_build_prompt(codec, filename, instruction,
                                     original, directives)
                       for filename, instruction, original, _ in cell_tasks]
            order = _attempt_order(len(cell_tasks), n_per_cell, look_schedule)
            looks = frozenset(look_schedule or ())
            landed = 0
            landed_applies = 0
            attempted = 0
            for task_index in order:
                _, _, original, expected = cell_tasks[task_index]
                prompt = prompts[task_index]
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
                # Looked at ONLY at the pre-registered points: peeking
                # between them inflates the false-decision rate.
                if attempted in looks and decided(
                    _stop_count(codec, landed, landed_applies), attempted
                ):
                    break
            results[codec][grade] = _cell(landed, landed_applies, attempted)
    return results
