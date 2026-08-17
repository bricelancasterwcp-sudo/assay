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

Grades are per codec (``GRADES_FOR``): the patch codecs are graded by
fixture size, and ``json_object`` by size AND by shape — v1.7 added
``nested``/``tabular``/``constrained`` beside the flat tiny/small/medium
cells, because "emits valid JSON" and "emits the JSON you described"
turned out to be different capabilities. The verdict cell did not move
with them (``profile._GRADE_FOR_VERDICTS``): the deep grades are new
columns, not a re-scaled ``structured_extraction``.
"""

import json
from dataclasses import dataclass

from assay import fixtures
from assay.errors import BudgetExhausted
from assay.stats import VERDICT_LENS, decided

_SEARCH_MARKER = "<<<<<<< SEARCH"
_DIVIDER = "======="
_REPLACE_MARKER = ">>>>>>> REPLACE"

CODECS = ("search_replace", "whole_file", "json_object")
GRADES = ("tiny", "small", "medium")

#: The two halves the matrix is bought in (v1.7). A consumer working
#: under a call ceiling preflights structured extraction and patch
#: editing SEPARATELY — they carry different verdicts and cost different
#: numbers of calls — so the split is registered here, once, and both the
#: ``only=`` filter and the cost table name these tuples instead of
#: re-typing the membership.
JSON_CODECS = ("json_object",)
PATCH_CODECS = tuple(codec for codec in CODECS if codec not in JSON_CODECS)

#: The json codec's DEEP grades (v1.7). ``tiny``/``small``/``medium``
#: grade the same flat object against three prompt sizes; these three
#: grade three different SHAPES — a second level of nesting, a
#: fixed-length homogeneous array, and an enum-plus-range object that
#: forbids stray keys. A model that emits `{name, count, tags}` and
#: nothing deeper is exactly what the flat grades cannot tell apart.
JSON_DEEP_GRADES = ("nested", "tabular", "constrained")

#: Which grades each codec is measured at. The patch codecs are graded
#: by fixture SIZE (the fixture set supplies tiny/small/medium bases);
#: json is graded by size and then by shape.
GRADES_FOR: dict[str, tuple[str, ...]] = {
    "search_replace": GRADES,
    "whole_file": GRADES,
    "json_object": GRADES + JSON_DEEP_GRADES,
}

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

# Each deep grade states its shape IN WORDS and validates exactly that
# shape — no grammar, no response_format, no forcing (spec §7). The
# tasks stay in JSON_TASKS' register: short concrete noun phrases with
# nothing to reason about, so the cell measures format-following and
# not world knowledge.
NESTED_DIRECTIVE = (
    "Return a JSON object with keys `name` (string), `location` (an "
    "object with `city` (string) and `coordinates` (an object with "
    "`lat` and `lon` numbers)), and `tags` (array of strings) "
    "describing:"
)
NESTED_TASKS = (
    "a corner bakery in Lisbon",
    "a lighthouse north of Reykjavik",
    "a night market stall in Taipei",
    "a public library branch in Cork",
    "a rooftop observatory above Santiago",
)

#: The row count the ``tabular`` directive asks for in words. Exact, not
#: a floor: "exactly three" is the instruction being measured.
_TABULAR_ROWS = 3
TABULAR_DIRECTIVE = (
    "Return a JSON array of exactly three objects, each with keys `id` "
    "(integer) and `label` (string), listing:"
)
TABULAR_TASKS = (
    "three tools from a bicycle repair kit",
    "three ferry stops along a river",
    "three seats in a small theatre",
    "three crates in a warehouse aisle",
    "three sensors on a weather mast",
)

#: The ``constrained`` contract's three literals: the enum, the
#: inclusive priority range, and the CLOSED key set. The directive
#: states all three in words; a test pins that the words and these
#: values still agree.
_CONSTRAINED_STATUSES = ("open", "closed", "pending")
_PRIORITY_RANGE = (1, 5)
_CONSTRAINED_KEYS = frozenset({"status", "priority", "note"})
CONSTRAINED_DIRECTIVE = (
    'Return a JSON object with keys `status` (exactly one of "open", '
    '"closed" or "pending"), `priority` (an integer from 1 to 5) and, '
    "optionally, `note` (string). Use no other keys. Describe:"
)
CONSTRAINED_TASKS = (
    "a maintenance ticket for a jammed loading door",
    "a support request about a misprinted receipt",
    "a work order for a leaking radiator valve",
    "a request to replace a burnt-out stairwell light",
    "a report of a wobbling cafe table",
)

_CHARS_PER_TOKEN = 5  # sizing proxy for the budget charge (plan Task 9)
_MAX_TOKENS = {"search_replace": 256, "whole_file": 768, "json_object": 128}
#: Per-(codec, grade) overrides of the codec's own ceiling. The deep
#: json grades need more room than the flat 128 — a nested object or a
#: three-row array truncated mid-reply is a measurement of the ceiling,
#: not of the model — and the flat grades keep 128 because every
#: committed profile's json numbers were measured under it.
_DEEP_MAX_TOKENS = 256
_MAX_TOKENS_BY_GRADE = {("json_object", grade): _DEEP_MAX_TOKENS
                        for grade in JSON_DEEP_GRADES}

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


_UNPARSED = object()  # distinct from every JSON value, `null` included


def _payload(reply: str):
    """The reply's JSON value, one outermost fence stripped, or
    ``_UNPARSED`` when it is not JSON. Never raises: a bad reply is
    data (a non-landing), never an exception."""
    interior = _strip_one_fence(reply)
    text = interior if interior is not None else reply
    try:
        return json.loads(text)
    except ValueError:
        return _UNPARSED


def _is_int(value) -> bool:
    """A JSON integer. ``True`` is an int in Python and is not one here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    """A JSON number (integer or float), booleans excluded as above."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_str_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(item, str)
                                           for item in value)


def validate_json_object(reply: str) -> bool:
    """Unforced JSON landing for the fixed extraction prompt.

    The reply (outermost fence stripped) must json.loads to a dict with
    `name` (string), `count` (integer), `tags` (array of strings).
    Extra keys are allowed. Never raises: a bad reply is data.
    """
    payload = _payload(reply)
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("name"), str):
        return False
    if not _is_int(payload.get("count")):
        return False
    return _is_str_list(payload.get("tags"))


def validate_json_nested(reply: str) -> bool:
    """The ``nested`` grade's contract: TWO levels below the root.

    A dict with `name` (string), `location` (a dict with `city` (string)
    and `coordinates`, itself a dict with `lat`/`lon` numbers) and
    `tags` (array of strings). Extra keys allowed, as in the flat grade
    — what this grade measures is whether the nesting survives, not
    whether the model was terse.
    """
    payload = _payload(reply)
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("name"), str):
        return False
    location = payload.get("location")
    if not isinstance(location, dict) or not isinstance(location.get("city"),
                                                        str):
        return False
    coordinates = location.get("coordinates")
    if not isinstance(coordinates, dict):
        return False
    if not _is_number(coordinates.get("lat")):
        return False
    if not _is_number(coordinates.get("lon")):
        return False
    return _is_str_list(payload.get("tags"))


def validate_json_tabular(reply: str) -> bool:
    """The ``tabular`` grade's contract: a homogeneous, fixed-length array.

    A JSON ARRAY of exactly ``_TABULAR_ROWS`` objects, each with `id`
    (integer) and `label` (string). Extra keys per object are allowed;
    the row COUNT is exact, because "exactly three" is the instruction
    being measured.
    """
    payload = _payload(reply)
    if not isinstance(payload, list) or len(payload) != _TABULAR_ROWS:
        return False
    for row in payload:
        if not isinstance(row, dict):
            return False
        if not _is_int(row.get("id")):
            return False
        if not isinstance(row.get("label"), str):
            return False
    return True


def validate_json_constrained(reply: str) -> bool:
    """The ``constrained`` grade's contract: an enum, a range, and a CLOSED
    key set.

    A dict with `status` (one of ``_CONSTRAINED_STATUSES``), `priority`
    (an integer in ``_PRIORITY_RANGE``, inclusive) and at most an
    optional `note` (string). This is the one validator where an EXTRA
    key fails: "use no other keys" is half of what the grade asks, so a
    reply that volunteers an `owner` did not follow the instruction.
    """
    payload = _payload(reply)
    if not isinstance(payload, dict):
        return False
    if set(payload) - _CONSTRAINED_KEYS:
        return False
    if payload.get("status") not in _CONSTRAINED_STATUSES:
        return False
    priority = payload.get("priority")
    low, high = _PRIORITY_RANGE
    if not _is_int(priority) or not low <= priority <= high:
        return False
    return "note" not in payload or isinstance(payload["note"], str)


DEFAULT_DIRECTIVES = CodecDirectives(
    search_replace=_SEARCH_REPLACE_DIRECTIVE,
    whole_file=_WHOLE_FILE_DIRECTIVE,
    json_object=JSON_DIRECTIVE,
)
DEFAULT_PRESENTATION = "default-v1"

#: Each deep json grade's own (directive, tasks, validator). A deep
#: grade's directive is NOT presentation the way ``CodecDirectives`` is
#: — it states the shape its validator enforces, so the two cannot be
#: substituted apart. A consumer's ``json_object`` directive therefore
#: replaces the FLAT grades' presentation only; the deep grades always
#: ask in the built-in words (and the profile's lens still records the
#: presentation, which is what the consumer changed).
_DEEP_GRADES = {
    "nested": (NESTED_DIRECTIVE, NESTED_TASKS, validate_json_nested),
    "tabular": (TABULAR_DIRECTIVE, TABULAR_TASKS, validate_json_tabular),
    "constrained": (CONSTRAINED_DIRECTIVE, CONSTRAINED_TASKS,
                    validate_json_constrained),
}


def _build_prompt(codec: str, grade: str, filename: str, instruction: str,
                  original: str, directives: CodecDirectives) -> str:
    if codec == "json_object":
        # instruction carries the task description for the json codec
        deep = _DEEP_GRADES.get(grade)
        directive = deep[0] if deep is not None else directives.json_object
        return f"{directive} {instruction}"
    directive = getattr(directives, codec)
    return f"{instruction}\n\n{directive}\n\nHere is `{filename}`:\n{original}"


def _max_tokens(codec: str, grade: str) -> int:
    """The reply ceiling for one cell: the codec's, unless the grade
    overrides it (the deep json grades do — see ``_MAX_TOKENS_BY_GRADE``)."""
    return _MAX_TOKENS_BY_GRADE.get((codec, grade), _MAX_TOKENS[codec])


def _parses_as_python(text: str) -> bool:
    try:
        compile(text, "<fixture>", "exec")
    except SyntaxError:
        return False
    return True


def _score(codec: str, grade: str, reply_text: str, original: str,
           expected: str) -> tuple[bool, bool]:
    """(byte_equality_landed, applies_and_parses_landed) for one reply."""
    if codec == "json_object":
        # Every json grade's two lenses coincide: validation IS the
        # application there, deep grades included.
        deep = _DEEP_GRADES.get(grade)
        validate = deep[2] if deep is not None else validate_json_object
        ok = validate(reply_text)
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


def _cell_tasks(codec: str, grade: str) -> list[tuple]:
    """One cell's HETEROGENEOUS tasks, as ``(filename, instruction,
    original, expected)`` — five defect classes on the grade's base
    module for the patch codecs, five task variants for json (v1.3: a
    cell that redraws one prompt measures sampler variance, not
    capability).

    The single source for both what ``probe_codecs`` sends and what
    ``cell_attempts`` prices: a cell whose task list changed size would
    otherwise cost one number and declare another.
    """
    if codec == "json_object":
        deep = _DEEP_GRADES.get(grade)
        tasks = deep[1] if deep is not None else JSON_TASKS
        return [(None, task, None, None) for task in tasks]
    return [(entry[2], entry[3], entry[4], entry[5])
            for entry in fixtures.EXPECTED if entry[0] == grade]


def cell_attempts(codec: str, grade: str, *, n_per_cell: int,
                  look_schedule: tuple[int, ...] | None) -> int:
    """The most calls one cell can cost, from the probe's OWN enumeration.

    Fixed-n: the attempt order is whole reps of the cell's task list, so
    a cell can cost slightly more or less than ``n_per_cell`` and this
    reports what it really costs. Sequential: the schedule's last entry
    is the cap, and a cell that decides early costs less — this is the
    ceiling, which is what a budget has to reserve.
    """
    return len(_attempt_order(len(_cell_tasks(codec, grade)), n_per_cell,
                              look_schedule))


def _stop_count(codec: str, landed: int, landed_applies: int) -> int:
    """The count the stop test reads: the codec's VERDICT lens.

    Which lens that is per codec is registered ONCE, in
    ``stats.VERDICT_LENS``, and profile's verdict layer reads the same
    entry — the rule used to be spelled in both modules, so a change to
    one was a silent disagreement with the other (v1.5 review debt).

    An unregistered codec counts applies-and-parses, the stricter stop
    (byte-equality can decide a cell the verdict would still call open)
    and the behaviour every non-``json_object`` codec has always had.
    """
    counts = {"lands": landed, "lands_applies": landed_applies}
    return counts[VERDICT_LENS.get(codec, "lands_applies")]


def stopped_on_rule(codec: str, cell: Landing,
                    look_schedule: tuple[int, ...]) -> bool:
    """True if a sequential cell ended on the STOPPING RULE, not a dead meter.

    A cell stops when a look decides its rung or when it reaches the
    cap; either way its n is the honest cost of a decision. The caller
    needs the distinction because under a schedule an n below the cap
    is the rule WORKING — reading it as budget death would skip every
    family that follows codecs (and say so in ``dropped``) on a run
    that never ran out of anything.
    """
    if cell.n == 0 or cell.lands is None or cell.lands_applies is None:
        return False
    if cell.n == look_schedule[-1]:
        return True  # the cap: every scheduled attempt was made
    if cell.n not in look_schedule:
        return False  # stopped between looks: nothing but the meter does that
    return decided(
        _stop_count(codec, round(cell.lands * cell.n),
                    round(cell.lands_applies * cell.n)),
        cell.n,
    )


def probe_codecs(
    backend, meter, *, n_per_cell: int, seed_base: int = 500,
    directives: CodecDirectives | None = None,
    look_schedule: tuple[int, ...] | None = None,
    only: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Landing]]:
    """Landing rates per codec x grade (spec §7), both lenses.

    Which grades a codec is measured at comes from ``GRADES_FOR``, so
    ``json_object`` carries its three deep shape grades alongside the
    three size ones and the patch codecs carry three each.

    Without ``look_schedule`` every cell is measured with `n_per_cell`
    seeded probes at fixed n. With one (``assay.stats.LOOK_SCHEDULE`` is
    the registered schedule), the cell samples round-robin and is
    examined ONLY at the schedule's look points, stopping at the first
    look where the Wilson-95 interval decides a rung on its verdict lens
    (``assay.stats.decided``); the last entry is the cap and
    ``n_per_cell`` is ignored. A cell that stops at n=5 differs from a
    fixed-n=5 cell only in the lens the caller stamps, so run.py records
    the stopping rule and n_used.

    ``only`` measures a SUBSET of the codecs (v1.7): the named ones are
    sampled exactly as they would be in a whole-matrix run, and every
    other codec's cells come back ``Landing(None, None, 0)`` — unmeasured,
    which is the value run.py's dropped loop names cell by cell. ``None``
    is the whole matrix, exactly the behaviour every committed profile
    was measured under. A name that is not a codec raises rather than
    quietly measuring nothing: a matrix of unmeasured cells is
    indistinguishable from a budget death, and a typo must not be able to
    produce one. The measured codecs are walked in ``CODECS`` order and
    seed from ``seed_base``, so a subset run is a subset OF THE MATRIX
    and not a full run with holes punched in it: its cells carry their
    own seeds rather than the ones they would have drawn behind the
    codecs nobody bought.

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
    if only is not None:
        unknown = [codec for codec in only if codec not in CODECS]
        if unknown:
            raise ValueError(
                f"unknown codec(s) in only={only!r}: {', '.join(unknown)} "
                f"(known: {', '.join(CODECS)})")
        if not only:
            raise ValueError("only must name at least one codec "
                             "(pass None for the whole matrix)")
    # Canonical order, whatever order `only` spelled: the same subset
    # asked for two ways must sample the same tasks under the same seeds.
    measured = tuple(codec for codec in CODECS
                     if only is None or codec in only)
    directives = directives or DEFAULT_DIRECTIVES
    results: dict[str, dict[str, Landing]] = {
        codec: {grade: Landing(lands=None, lands_applies=None, n=0)
                for grade in GRADES_FOR[codec]}
        for codec in CODECS
    }
    seed = seed_base
    exhausted = False
    for codec in measured:
        if exhausted:
            break
        for grade in GRADES_FOR[codec]:
            if exhausted:
                break
            # v1.3: a cell's attempts spread across HETEROGENEOUS tasks
            # (five defect classes on the grade's base module; five task
            # variants for json), not repeated draws of one prompt — the
            # v1/v2 sets measured sampler variance on a single fixture.
            cell_tasks = _cell_tasks(codec, grade)
            prompts = [_build_prompt(codec, grade, filename, instruction,
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
                    prompt, seed=seed, max_tokens=_max_tokens(codec, grade)
                )
                seed += 1
                attempted += 1
                exact, applies = _score(codec, grade, reply.text, original,
                                        expected)
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
