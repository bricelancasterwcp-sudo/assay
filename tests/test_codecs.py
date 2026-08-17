"""Tests for codec fixtures, appliers, and probes (plan Task 9, spec §7)."""

import hashlib
from collections import Counter

import pytest

from assay import fixtures
from assay.codecs import (
    CodecDirectives,
    apply_search_replace,
    apply_whole_file,
    landing_equal,
    validate_json_constrained,
    validate_json_nested,
    validate_json_object,
    validate_json_tabular,
)

ORIGINAL = (
    "def total(prices):\n"
    "    subtotal = sum(prices)\n"
    "    subtotal * 1.08          # BUG: result dropped\n"
    "def label(n):\n"
    '    return f"{n} items"\n'
)

EXPECTED_FIX = (
    "def total(prices):\n"
    "    subtotal = sum(prices)\n"
    "    return subtotal * 1.08\n"
    "def label(n):\n"
    '    return f"{n} items"\n'
)


def block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


# --- apply_search_replace -------------------------------------------------


def test_unindented_search_does_not_match():
    """The measured 7B failure mode: an unindented SEARCH matches nothing."""
    reply = block(
        "subtotal * 1.08          # BUG: result dropped",
        "return subtotal * 1.08",
    )
    assert apply_search_replace(ORIGINAL, reply) is None


def test_exact_search_replaces_whole_lines():
    reply = block(
        "    subtotal * 1.08          # BUG: result dropped",
        "    return subtotal * 1.08",
    )
    assert apply_search_replace(ORIGINAL, reply) == EXPECTED_FIX


def test_fenced_block_is_accepted():
    reply = "```\n" + block(
        "    subtotal * 1.08          # BUG: result dropped",
        "    return subtotal * 1.08",
    ) + "\n```"
    assert apply_search_replace(ORIGINAL, reply) == EXPECTED_FIX


def test_ambiguous_search_two_matches_is_rejected():
    original = "x = 1\ny = 2\nx = 1\n"
    reply = block("x = 1", "x = 3")
    assert apply_search_replace(original, reply) is None


def test_reply_must_contain_exactly_one_block():
    one = block(
        "    subtotal * 1.08          # BUG: result dropped",
        "    return subtotal * 1.08",
    )
    two = one + "\n" + block("def label(n):", "def label(count):")
    assert apply_search_replace(ORIGINAL, two) is None
    assert apply_search_replace(ORIGINAL, "no block here") is None


# --- landing equality -----------------------------------------------------


@pytest.mark.parametrize(
    ("applied", "expected", "lands"),
    [
        ("a\nb\n", "a\nb\n", True),  # identical
        ("a\nb", "a\nb\n", True),  # applied missing the trailing newline
        ("a\nb\n", "a\nb", True),  # applied has one extra trailing newline
        ("a\nb\n\n", "a\nb", False),  # two extra newlines: not tolerated
        ("a\nb", "a\nb\n\n", False),
        (" a\nb\n", "a\nb\n", False),  # leading whitespace difference
        ("a\nc\n", "a\nb\n", False),  # content difference
    ],
)
def test_trailing_newline_tolerated_nothing_else(applied, expected, lands):
    assert landing_equal(applied, expected) is lands


# --- apply_whole_file -----------------------------------------------------


def test_whole_file_strips_one_outer_fence_only():
    inner = "```python\nx = 1\n```"
    reply = "```\n" + inner + "\n```"
    assert apply_whole_file(reply) == inner


def test_whole_file_strips_language_tagged_fence():
    assert apply_whole_file("```python\nx = 1\n```") == "x = 1"


def test_whole_file_bare_reply_passes_through():
    assert apply_whole_file("x = 1\n") == "x = 1\n"


def test_whole_file_empty_reply_is_none():
    assert apply_whole_file("   \n") is None


# --- validate_json_object -------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "valid"),
    [
        ('{"name": "apples", "count": 3, "tags": ["fruit"]}', True),
        ('```json\n{"name": "apples", "count": 3, "tags": ["fruit"]}\n```', True),
        (
            '{"name": "apples", "count": 3, "tags": ["fruit"], "color": "red"}',
            True,  # extra keys allowed
        ),
        ('{"count": 3, "tags": ["fruit"]}', False),  # missing key
        ('{"name": "apples", "count": "three", "tags": ["fruit"]}', False),
        ('{"name": "apples", "count": true, "tags": ["fruit"]}', False),
        ('{"name": "apples", "count": 3, "tags": "fruit"}', False),
        ('{"name": "apples", "count": 3, "tags": [1, 2]}', False),
        ('["name", "count", "tags"]', False),  # not an object
        ("not json at all", False),
    ],
)
def test_json_required_keys_and_types(reply, valid):
    assert validate_json_object(reply) is valid


# --- the deep json grades (v1.7): nested / tabular / constrained ----------
#
# The validator IS the grade's contract. Each truth table below carries
# one passing reply, the same reply fenced, and one failing reply per
# rule — a grade whose validator accepts a shape the directive never
# asked for measures the validator's slack, not the model.

_NESTED_OK = ('{"name": "corner bakery", "location": {"city": "Lisbon", '
              '"coordinates": {"lat": 38.72, "lon": -9.14}}, '
              '"tags": ["bread", "cafe"]}')


@pytest.mark.parametrize(
    ("reply", "valid"),
    [
        (_NESTED_OK, True),
        (f"```json\n{_NESTED_OK}\n```", True),  # fenced and bare both land
        # extra top-level keys are allowed (only `constrained` forbids them)
        ('{"name": "b", "location": {"city": "L", "coordinates": '
         '{"lat": 1, "lon": 2}}, "tags": [], "rating": 4}', True),
        # ...and integer coordinates are numbers, as are negatives
        ('{"name": "b", "location": {"city": "L", "coordinates": '
         '{"lat": -38, "lon": 9}}, "tags": ["x"]}', True),
        # missing / mistyped name
        ('{"location": {"city": "L", "coordinates": {"lat": 1, "lon": 2}}, '
         '"tags": []}', False),
        ('{"name": 7, "location": {"city": "L", "coordinates": '
         '{"lat": 1, "lon": 2}}, "tags": []}', False),
        # missing location, and location flattened to a string: the
        # nesting IS the grade
        ('{"name": "b", "tags": []}', False),
        ('{"name": "b", "location": "Lisbon", "tags": []}', False),
        # location without city, city mistyped
        ('{"name": "b", "location": {"coordinates": {"lat": 1, "lon": 2}}, '
         '"tags": []}', False),
        ('{"name": "b", "location": {"city": 9, "coordinates": '
         '{"lat": 1, "lon": 2}}, "tags": []}', False),
        # the second nesting level dropped, or flattened to an array
        ('{"name": "b", "location": {"city": "L"}, "tags": []}', False),
        ('{"name": "b", "location": {"city": "L", "coordinates": [1, 2]}, '
         '"tags": []}', False),
        # coordinates missing a component, or carrying a non-number
        ('{"name": "b", "location": {"city": "L", "coordinates": '
         '{"lat": 1}}, "tags": []}', False),
        ('{"name": "b", "location": {"city": "L", "coordinates": '
         '{"lat": "1", "lon": 2}}, "tags": []}', False),
        ('{"name": "b", "location": {"city": "L", "coordinates": '
         '{"lat": true, "lon": 2}}, "tags": []}', False),
        # missing / mistyped tags, and a non-string inside them
        ('{"name": "b", "location": {"city": "L", "coordinates": '
         '{"lat": 1, "lon": 2}}}', False),
        ('{"name": "b", "location": {"city": "L", "coordinates": '
         '{"lat": 1, "lon": 2}}, "tags": "bread"}', False),
        ('{"name": "b", "location": {"city": "L", "coordinates": '
         '{"lat": 1, "lon": 2}}, "tags": ["bread", 7]}', False),
        ("[1, 2, 3]", False),  # not an object at all
        ("I would rather describe the bakery in prose.", False),
    ],
)
def test_json_nested_truth_table(reply, valid):
    assert validate_json_nested(reply) is valid


_TABULAR_OK = ('[{"id": 1, "label": "tyre lever"}, '
               '{"id": 2, "label": "patch kit"}, {"id": 3, "label": "pump"}]')


@pytest.mark.parametrize(
    ("reply", "valid"),
    [
        (_TABULAR_OK, True),
        (f"```json\n{_TABULAR_OK}\n```", True),
        # extra keys per object are allowed; the row shape is a floor
        ('[{"id": 1, "label": "a", "note": "x"}, {"id": 2, "label": "b"}, '
         '{"id": 3, "label": "c"}]', True),
        # wrong length, either way — three is exact, not a minimum
        ('[{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]', False),
        ('[{"id": 1, "label": "a"}, {"id": 2, "label": "b"}, '
         '{"id": 3, "label": "c"}, {"id": 4, "label": "d"}]', False),
        ("[]", False),
        # an object wrapping the rows is not the array that was asked for
        ('{"rows": [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}, '
         '{"id": 3, "label": "c"}]}', False),
        # a row missing a key, or carrying the wrong type for one
        ('[{"id": 1}, {"id": 2, "label": "b"}, {"id": 3, "label": "c"}]',
         False),
        ('[{"label": "a"}, {"id": 2, "label": "b"}, {"id": 3, "label": "c"}]',
         False),
        ('[{"id": "1", "label": "a"}, {"id": 2, "label": "b"}, '
         '{"id": 3, "label": "c"}]', False),
        ('[{"id": 1.5, "label": "a"}, {"id": 2, "label": "b"}, '
         '{"id": 3, "label": "c"}]', False),
        ('[{"id": true, "label": "a"}, {"id": 2, "label": "b"}, '
         '{"id": 3, "label": "c"}]', False),
        ('[{"id": 1, "label": 7}, {"id": 2, "label": "b"}, '
         '{"id": 3, "label": "c"}]', False),
        # heterogeneous rows: a bare string is not a row
        ('["a", {"id": 2, "label": "b"}, {"id": 3, "label": "c"}]', False),
        ("Here are three tools: a tyre lever, a patch kit and a pump.", False),
    ],
)
def test_json_tabular_truth_table(reply, valid):
    assert validate_json_tabular(reply) is valid


_CONSTRAINED_OK = '{"status": "open", "priority": 3}'


@pytest.mark.parametrize(
    ("reply", "valid"),
    [
        (_CONSTRAINED_OK, True),
        (f"```json\n{_CONSTRAINED_OK}\n```", True),
        # `note` is optional, and the whole enum is legal
        ('{"status": "pending", "priority": 5, "note": "waiting on parts"}',
         True),
        ('{"status": "closed", "priority": 1}', True),
        # off the enum, missing, or the right word in the wrong type
        ('{"status": "urgent", "priority": 3}', False),
        ('{"status": "Open", "priority": 3}', False),
        ('{"priority": 3}', False),
        ('{"status": true, "priority": 3}', False),
        # priority outside 1..5 (inclusive), or not an integer
        ('{"status": "open", "priority": 0}', False),
        ('{"status": "open", "priority": 6}', False),
        ('{"status": "open", "priority": "3"}', False),
        ('{"status": "open", "priority": 2.5}', False),
        ('{"status": "open", "priority": true}', False),
        ('{"status": "open"}', False),
        # note present but mistyped
        ('{"status": "open", "priority": 3, "note": 7}', False),
        # THE grade's point: any key beyond the three FAILS here
        ('{"status": "open", "priority": 3, "owner": "sam"}', False),
        ('{"status": "open", "priority": 3, "note": "x", "id": 1}', False),
        ('["open", 3]', False),
        ("The door ticket is open at priority three.", False),
    ],
)
def test_json_constrained_truth_table(reply, valid):
    assert validate_json_constrained(reply) is valid


def test_the_deep_validators_never_raise_on_hostile_replies():
    """A bad reply is DATA — every validator answers False, never throws."""
    hostile = ["", "   ", "null", "true", "3", '"a string"', "{", "[]",
               "```\n```", "```json\nnot json\n```", "\x00"]
    for reply in hostile:
        for validate in (validate_json_nested, validate_json_tabular,
                         validate_json_constrained):
            assert validate(reply) is False, (validate.__name__, reply)


def test_grades_for_gives_only_json_the_deep_grades():
    from assay.codecs import CODECS, GRADES, GRADES_FOR, JSON_DEEP_GRADES

    assert GRADES == ("tiny", "small", "medium")  # unchanged: v2's grades
    assert JSON_DEEP_GRADES == ("nested", "tabular", "constrained")
    assert set(GRADES_FOR) == set(CODECS)
    assert GRADES_FOR["search_replace"] == GRADES
    assert GRADES_FOR["whole_file"] == GRADES
    assert GRADES_FOR["json_object"] == GRADES + JSON_DEEP_GRADES


def test_each_deep_grade_has_five_tasks_and_its_own_directive():
    from assay.codecs import (CONSTRAINED_DIRECTIVE, CONSTRAINED_TASKS,
                              JSON_DIRECTIVE, JSON_TASKS, NESTED_DIRECTIVE,
                              NESTED_TASKS, TABULAR_DIRECTIVE, TABULAR_TASKS)

    directives = [JSON_DIRECTIVE, NESTED_DIRECTIVE, TABULAR_DIRECTIVE,
                  CONSTRAINED_DIRECTIVE]
    assert len(set(directives)) == 4
    # No directive may PREFIX another: prompts are routed by their
    # directive (here, in the fakes, and in any consumer reading a
    # transcript), and a prefix would silently route one grade's replies
    # into another grade's cell.
    for one in directives:
        for other in directives:
            assert one is other or not other.startswith(one)
    for tasks in (NESTED_TASKS, TABULAR_TASKS, CONSTRAINED_TASKS):
        assert len(tasks) == len(JSON_TASKS) == 5
        assert len(set(tasks)) == 5
        assert all(task and task == task.strip() for task in tasks)


def test_each_deep_directive_states_the_contract_its_validator_enforces():
    """The words the model is shown and the rule its reply is judged by
    must not drift apart: a directive that asks for something the
    validator does not enforce (or the reverse) measures the gap between
    them rather than the model."""
    from assay.codecs import (_CONSTRAINED_KEYS, _CONSTRAINED_STATUSES,
                              _PRIORITY_RANGE, _TABULAR_ROWS,
                              CONSTRAINED_DIRECTIVE, NESTED_DIRECTIVE,
                              TABULAR_DIRECTIVE)

    # nested: both levels below the root are named in the words
    for key in ("`name`", "`location`", "`city`", "`coordinates`", "`lat`",
                "`lon`", "`tags`"):
        assert key in NESTED_DIRECTIVE, key
    # tabular: the exact row count, spelled
    assert _TABULAR_ROWS == 3
    assert "exactly three objects" in TABULAR_DIRECTIVE
    # constrained: the enum, the range and the closed key set
    for status in _CONSTRAINED_STATUSES:
        assert f'"{status}"' in CONSTRAINED_DIRECTIVE, status
    low, high = _PRIORITY_RANGE
    assert f"an integer from {low} to {high}" in CONSTRAINED_DIRECTIVE
    for key in _CONSTRAINED_KEYS:
        assert f"`{key}`" in CONSTRAINED_DIRECTIVE, key
    assert "no other keys" in CONSTRAINED_DIRECTIVE


# --- v2 surfaces are frozen ----------------------------------------------


def test_v2_surfaces_are_byte_frozen():
    """The v2 measuring surfaces, pinned as DATA rather than imported.

    Every committed profile's codec numbers were produced by these exact
    bytes. v3 ADDS grades; it does not edit the ones the archive was
    measured under, and a pin that imported the constant it guards would
    move with the edit it exists to catch.
    """
    from assay.codecs import JSON_DIRECTIVE, JSON_TASKS
    from assay.fixtures import _DEFECTS, _DIR

    assert JSON_DIRECTIVE == (
        "Return a JSON object with keys `name` (string), `count` "
        "(integer), and `tags` (array of strings) describing:"
    )
    assert JSON_TASKS == (
        "three apples",
        "two rusty bicycles leaning on a fence",
        "five copper coins from an old purse",
        "one chess board mid-game",
        "four rain boots by the door",
    )
    assert len(_DEFECTS) == 15
    assert _DEFECTS[0] == (
        "tiny", "dropped_return",
        "    return subtotal * 1.08",
        "    subtotal * 1.08",
        "In tiny.py, total([10]) returns None; it should return 10.8. "
        "Fix the single defective line.",
    )
    # The three base modules, byte for byte.
    assert {
        name: hashlib.sha256(
            (_DIR / f"{name}.py.txt").read_bytes()).hexdigest()
        for name in ("tiny", "small", "medium")
    } == {
        "tiny": "0290a4a3361a3ff906182568051e31bb"
                "0a422692dd15c181aee6bb407caf9d27",
        "small": "2491a3b0e7564d9b59af9a8fb0869f05"
                 "4c73f02899315bd56871a13c4fe2864c",
        "medium": "c07513c190975db6eb86e1606557e49c"
                  "27b295721a43e6b8be125b23b2a5ae6b",
    }


# --- fixtures -------------------------------------------------------------


def test_fixture_set_integrity():
    from assay import fixtures

    # v3 (v1.7): the patch fixtures are v2's, byte for byte (pinned
    # above); the set NAME moves because the json side gained three
    # grades, and the name is what travels in every profile's lens.
    assert fixtures.FIXTURE_SET == "codec-fixtures-v3"
    per_grade = {}
    bounds = {"tiny": (1, 250), "small": (400, 700), "medium": (1100, 2200)}
    for grade, dclass, filename, instruction, original, expected in fixtures.EXPECTED:
        per_grade.setdefault(grade, []).append((dclass, expected))
        lo, hi = bounds[grade]
        assert lo <= len(original) <= hi, f"{grade} is {len(original)} chars"
        assert filename in instruction          # the instruction names the file
        assert expected != original
        # both sides compile: a fixture that is not valid Python measures
        # the authoring, not the model
        compile(original, filename, "exec")
        compile(expected, filename, "exec")
        # exactly one line differs
        diff = [i for i, (a, b) in enumerate(
            zip(original.splitlines(), expected.splitlines())) if a != b]
        assert len(diff) == 1, (grade, dclass, diff)
    for grade, entries in per_grade.items():
        # five heterogeneous defect classes per grade, one task each
        assert sorted(d for d, _ in entries) == sorted(fixtures.DEFECT_CLASSES)
        # all entries share the grade's clean base
        assert len({e for _, e in entries}) == 1


def test_historic_tiny_defect_is_preserved():
    # Continuity with the v1 set: the dropped-return on total() survives
    # as tiny/dropped_return, so the oldest regression shape stays covered.
    from assay import fixtures

    entry = [e for e in fixtures.EXPECTED
             if e[0] == "tiny" and e[1] == "dropped_return"][0]
    assert "    subtotal * 1.08" in entry[4]        # broken: value dropped
    assert "    return subtotal * 1.08" in entry[5]  # fixed


# --- probe_codecs ---------------------------------------------------------


class ScriptedBackend:
    """Local fake: `script(prompt) -> reply text`; records every call."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.model = "fake-model"

    def generate(self, prompt, **kwargs):
        from assay.backends.base import Reply

        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return Reply(
            text=self.script(prompt),
            tokens_in=None,
            tokens_out=None,
            stop_reason="stop",
            raw={},
        )


def make_meter(max_calls=10_000, max_prompt_tokens=10_000_000):
    from assay.budget import Budget, BudgetMeter

    return BudgetMeter(Budget(max_calls=max_calls, max_prompt_tokens=max_prompt_tokens))


_FLAT_JSON_REPLY = '{"name": "x", "count": 3, "tags": ["t"]}'
_NESTED_REPLY = ('{"name": "x", "location": {"city": "L", "coordinates": '
                 '{"lat": 1.5, "lon": -2.5}}, "tags": ["t"]}')
_TABULAR_REPLY = ('[{"id": 1, "label": "a"}, {"id": 2, "label": "b"}, '
                  '{"id": 3, "label": "c"}]')
_CONSTRAINED_REPLY = '{"status": "open", "priority": 2}'


def deep_json_reply(prompt):
    """The valid reply for a deep json prompt, or None when it is not one."""
    from assay.codecs import (CONSTRAINED_DIRECTIVE, NESTED_DIRECTIVE,
                              TABULAR_DIRECTIVE)

    for directive, reply in ((NESTED_DIRECTIVE, _NESTED_REPLY),
                             (TABULAR_DIRECTIVE, _TABULAR_REPLY),
                             (CONSTRAINED_DIRECTIVE, _CONSTRAINED_REPLY)):
        if prompt.startswith(directive):
            return reply
    return None


def grade_matrix_script(prompt):
    """Lands json everywhere, whole_file everywhere, search_replace on
    tiny only — flubs small and medium. Routes every v2 fixture task and
    every deep json grade."""
    from assay import fixtures
    from assay.codecs import JSON_DIRECTIVE

    deep = deep_json_reply(prompt)
    if deep is not None:
        return deep
    if prompt.startswith(JSON_DIRECTIVE):
        return _FLAT_JSON_REPLY
    for grade, _, _, _, original, expected in fixtures.EXPECTED:
        if original in prompt:
            if "SEARCH" in prompt:
                if grade != "tiny":
                    return "no patch from me today."
                o_lines = original.split("\n")
                e_lines = expected.split("\n")
                at = next(i for i, (a, b) in enumerate(zip(o_lines, e_lines))
                          if a != b)
                return block(o_lines[at], e_lines[at])
            return expected
    raise AssertionError(f"unrouted codec prompt: {prompt[:80]!r}")


def test_landing_by_grade_matrix():
    from assay.codecs import Landing, probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=3)

    # v2 cells spread over the grade's five tasks (n_per_cell below the
    # task count still measures each task once: n=5).
    assert result["search_replace"]["tiny"] == Landing(lands=1.0, lands_applies=1.0, n=5)
    assert result["search_replace"]["small"] == Landing(lands=0.0, lands_applies=0.0, n=5)
    assert result["search_replace"]["medium"] == Landing(lands=0.0, lands_applies=0.0, n=5)
    for grade in ("tiny", "small", "medium"):
        assert result["whole_file"][grade] == Landing(lands=1.0, lands_applies=1.0, n=5)
        assert result["json_object"][grade] == Landing(lands=1.0, lands_applies=1.0, n=5)


def test_probe_codecs_measures_the_deep_grades():
    """The json codec carries six cells; the patch codecs still carry three."""
    from assay.codecs import GRADES, GRADES_FOR, JSON_DEEP_GRADES, Landing, probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=5)

    assert tuple(result["json_object"]) == GRADES_FOR["json_object"]
    for grade in JSON_DEEP_GRADES:
        assert result["json_object"][grade] == Landing(lands=1.0,
                                                       lands_applies=1.0, n=5)
    for codec in ("search_replace", "whole_file"):
        assert tuple(result[codec]) == GRADES
    # ...and each deep cell spread its five attempts over five DIFFERENT
    # tasks, the v1.3 rule, not five draws of one prompt.
    for grade in JSON_DEEP_GRADES:
        prompts = [call["prompt"] for call in backend.calls
                   if call["prompt"].startswith(_directive_for(grade))]
        assert len(prompts) == 5 and len(set(prompts)) == 5, grade


def _directive_for(grade):
    from assay.codecs import (CONSTRAINED_DIRECTIVE, NESTED_DIRECTIVE,
                              TABULAR_DIRECTIVE)

    return {"nested": NESTED_DIRECTIVE, "tabular": TABULAR_DIRECTIVE,
            "constrained": CONSTRAINED_DIRECTIVE}[grade]


def _is_deep_prompt(prompt):
    from assay.codecs import JSON_DEEP_GRADES

    return any(prompt.startswith(_directive_for(grade))
               for grade in JSON_DEEP_GRADES)


def test_each_deep_grade_is_scored_by_its_own_validator():
    """A reply that satisfies the FLAT contract lands the flat grades and
    nothing else: each deep cell is judged by its own validator, so a
    model that only ever emits `{name, count, tags}` is measured as
    failing every deeper shape."""
    from assay.codecs import GRADES, JSON_DEEP_GRADES, probe_codecs

    backend = ScriptedBackend(lambda prompt: _FLAT_JSON_REPLY)
    result = probe_codecs(backend, make_meter(), n_per_cell=5)

    for grade in GRADES:
        assert result["json_object"][grade].lands == 1.0, grade
    for grade in JSON_DEEP_GRADES:
        cell = result["json_object"][grade]
        assert cell.lands == 0.0, grade
        assert cell.lands_applies == 0.0, grade  # json's lenses coincide
        assert cell.n == 5


def test_deep_cells_are_given_more_room_than_the_flat_json_cells():
    """nested and tabular replies do not fit in the flat grade's 128
    tokens; the flat cells keep 128 because every committed profile's
    json numbers were measured under it."""
    from assay.codecs import JSON_DEEP_GRADES, JSON_DIRECTIVE, probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    probe_codecs(backend, make_meter(), n_per_cell=1)

    flat = [c for c in backend.calls if c["prompt"].startswith(JSON_DIRECTIVE)]
    assert flat and {c["kwargs"]["max_tokens"] for c in flat} == {128}
    for grade in JSON_DEEP_GRADES:
        deep = [c for c in backend.calls
                if c["prompt"].startswith(_directive_for(grade))]
        assert deep and {c["kwargs"]["max_tokens"] for c in deep} == {256}, grade


def test_deep_grades_keep_their_built_in_directive_under_consumer_ones():
    """CodecDirectives substitutes the consumer's presentation for the
    three codecs it names. A deep grade's directive is not presentation
    — it STATES the contract its validator enforces — so it is always
    the built-in one, and the flat json grade is what a consumer's
    `json_object` directive replaces."""
    from assay.codecs import (JSON_DEEP_GRADES, JSON_DIRECTIVE, probe_codecs)

    marker = "REPLY-IN-THE-STYLE-OF-THE-CONSUMER-APP"
    custom = CodecDirectives(search_replace=f"{marker} sr",
                             whole_file=f"{marker} wf",
                             json_object=f"{marker} jo")
    backend = ScriptedBackend(lambda prompt: "not a valid reply")
    probe_codecs(backend, make_meter(), n_per_cell=1, directives=custom)

    prompts = [c["prompt"] for c in backend.calls]
    assert not any(p.startswith(JSON_DIRECTIVE) for p in prompts)
    for grade in JSON_DEEP_GRADES:
        deep = [p for p in prompts if p.startswith(_directive_for(grade))]
        assert len(deep) == 5, grade
        assert not any(marker in p for p in deep), grade


def test_the_verdict_cell_is_pinned():
    """structured_extraction still reads `json_object.small`.

    The deep grades are new COLUMNS, not a new verdict: moving the
    graded cell would silently re-scale every published verdict against
    the archive. Pinned through the lens a consumer actually reads.
    """
    from assay.codecs import GRADES_FOR, Landing
    from assay.profile import compute_verdicts
    from assay.stats import wilson95

    def cell(rate, n):
        return Landing(lands=rate, lands_applies=rate, n=n)

    # Distinctive small cell: 2/5 lands where every other grade is 35/35.
    codecs = {
        "json_object": {grade: (cell(0.4, 5) if grade == "small"
                                else cell(1.0, 35))
                        for grade in GRADES_FOR["json_object"]},
        "search_replace": {grade: cell(1.0, 35) for grade in ("tiny", "small",
                                                              "medium")},
        "whole_file": {grade: cell(1.0, 35) for grade in ("tiny", "small",
                                                          "medium")},
    }
    verdicts = compute_verdicts(None, None, None, codecs)
    lo, hi = wilson95(2, 5)
    assert verdicts["structured_extraction"]["interval95"] == [round(lo, 3),
                                                               round(hi, 3)]
    assert verdicts["structured_extraction"]["verdict"] == "unusable"
    # ...and that is NOT the answer any deep cell would have given.
    other_lo, other_hi = wilson95(35, 35)
    assert [round(other_lo, 3), round(other_hi, 3)] != [round(lo, 3),
                                                        round(hi, 3)]


def test_zero_cell_is_none_not_zero():
    """Budget dies after the first cell: that cell is a measured 0.0,
    every unattempted cell is None — never a zero nobody measured."""
    from assay.codecs import Landing, probe_codecs

    backend = ScriptedBackend(lambda prompt: "no patch for you")
    meter = make_meter(max_calls=2)
    result = probe_codecs(backend, meter, n_per_cell=2)

    assert result["search_replace"]["tiny"] == Landing(lands=0.0, lands_applies=0.0, n=2)  # measured
    assert result["search_replace"]["small"].lands is None
    assert result["search_replace"]["small"].n == 0
    for codec in ("whole_file", "json_object"):
        for grade in ("tiny", "small", "medium"):
            assert result[codec][grade] == Landing(lands=None, lands_applies=None, n=0)


def test_partial_cell_reports_what_was_measured():
    from assay.codecs import Landing, probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    meter = make_meter(max_calls=1)  # dies mid-first-cell
    result = probe_codecs(backend, meter, n_per_cell=3)

    assert result["search_replace"]["tiny"] == Landing(lands=1.0, lands_applies=1.0, n=1)
    assert result["search_replace"]["small"] == Landing(lands=None, lands_applies=None, n=0)


def test_probe_never_sends_format_forcing():
    """Pins the no-forcing constraint at the wire: no request carries a
    format/response_format key, in kwargs or anywhere in the payload."""
    from assay.codecs import probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    probe_codecs(backend, make_meter(), n_per_cell=1)

    assert backend.calls, "probe made no calls"
    for call in backend.calls:
        assert "format" not in call["kwargs"]
        assert "response_format" not in call["kwargs"]
        assert set(call["kwargs"]) <= {"seed", "max_tokens", "num_ctx"}


def test_probe_seeds_are_distinct_and_derived_from_seed_base():
    """An n_per_cell cell must be n DIFFERENT samples: on a
    deterministic endpoint, repeated (prompt, seed) pairs would measure
    once and report n=5 — a value that looks like a measurement but is
    not (spec §10)."""
    from assay.codecs import probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    probe_codecs(backend, make_meter(), n_per_cell=2, seed_base=500)

    seeds = [call["kwargs"]["seed"] for call in backend.calls]
    # 12 cells (3 + 3 + 6 grades) x 5 tasks — v1.7 gave json three more.
    assert seeds == list(range(500, 500 + 60))


def test_probe_charges_the_meter_per_call():
    from assay.codecs import probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    meter = make_meter()
    probe_codecs(backend, meter, n_per_cell=1)

    assert meter.spent.calls == len(backend.calls) == 60  # 12 cells x 5 tasks
    assert meter.spent.prompt_tokens > 0


def test_lenses_diverge_on_semantically_valid_but_not_byte_equal_reply():
    # The 2026-08-12 qwen finding as a regression test: a whole_file
    # reply that fixes the defect but editorializes a comment is valid
    # Python (applies-and-parses lands) while failing byte-equality.
    tiny = [f for f in fixtures.EXPECTED
            if f[0] == "tiny" and f[1] == "dropped_return"][0]
    _, _, _, _, original, expected = tiny
    editorialized = expected.replace("return subtotal * 1.08",
                                     "return subtotal * 1.08  # fixed")
    assert editorialized != expected
    from assay.budget import Budget, BudgetMeter
    from assay.codecs import probe_codecs
    backend = ScriptedBackend(lambda p: editorialized)
    meter = BudgetMeter(Budget(max_calls=999, max_prompt_tokens=10**9))
    result = probe_codecs(backend, meter, n_per_cell=2)
    cell = result["whole_file"]["tiny"]
    assert cell.lands == 0.0
    assert cell.lands_applies == 1.0


def test_custom_directives_reach_the_wire_verbatim():
    marker = "REPLY-IN-THE-STYLE-OF-THE-CONSUMER-APP"
    custom = CodecDirectives(
        search_replace=f"{marker} sr", whole_file=f"{marker} wf",
        json_object=f"{marker} jo",
    )
    from assay.budget import Budget, BudgetMeter
    from assay.codecs import probe_codecs
    backend = ScriptedBackend(lambda p: "not a valid reply")
    meter = BudgetMeter(Budget(max_calls=999, max_prompt_tokens=10**9))
    probe_codecs(backend, meter, n_per_cell=1, directives=custom)
    prompts = [c["prompt"] for c in backend.calls]
    # The three SUBSTITUTABLE codec presentations — the deep json grades
    # ask in their own words by design, and have their own test.
    substitutable = [p for p in prompts if not _is_deep_prompt(p)]
    assert len(substitutable) == 45, "a substitutable cell went missing"
    assert all(marker in p for p in substitutable), "custom directive missing"
    # And the built-in texts must NOT appear anywhere.
    assert not any("character for character" in p for p in prompts)


# --- sequential look-schedule stopping (v1.5, spec §1) --------------------


def never_lands_script(prompt):
    """Nothing lands under EITHER lens: the reply is not a block, not a
    valid file, and not JSON. A cell of these is 0/5 at the first look —
    both Wilson endpoints ladder `unusable`, so it is decided."""
    return "no patch for you"


def json_four_of_five_script(prompt):
    """Lands four of the json cell's five tasks: 0.8 straddles risky and
    ready at every look (4/5, 8/10, 16/20 all span the 0.9 threshold),
    so a json cell must run to the schedule's cap. Patch codecs never
    land, so their cells decide at the first look — and neither do the
    deep json grades, whose shapes this flat reply does not satisfy."""
    from assay.codecs import JSON_DIRECTIVE, JSON_TASKS

    if prompt.startswith(JSON_DIRECTIVE):
        if prompt.endswith(JSON_TASKS[-1]):
            return "sorry, no json today"
        return '{"name": "x", "count": 3, "tags": ["t"]}'
    return "no patch for you"


def json_one_of_five_script(prompt):
    """Lands exactly ONE of the json cell's five tasks, so a round adds
    one landing: 1/5 at the first look is undecided (unusable/risky),
    2/10 at the second is decided unusable. The cell must stop at an
    INTERMEDIATE look — not the schedule's first entry, not its cap."""
    from assay.codecs import JSON_DIRECTIVE, JSON_TASKS

    if prompt.startswith(JSON_DIRECTIVE):
        if prompt.endswith(JSON_TASKS[0]):
            return '{"name": "x", "count": 3, "tags": ["t"]}'
        return "sorry, no json today"
    return "no patch for you"


def applies_but_not_equal_script(prompt):
    """Patch replies that APPLY and parse but are never byte-equal (a
    trailing comment). lands_applies == 1.0 while lands == 0.0, so the
    two lenses disagree about whether the cell is decided."""
    from assay.codecs import JSON_DIRECTIVE

    if prompt.startswith(JSON_DIRECTIVE) or _is_deep_prompt(prompt):
        return "sorry, no json today"
    for _, _, _, _, original, expected in fixtures.EXPECTED:
        if original in prompt:
            if "SEARCH" in prompt:
                o_lines = original.split("\n")
                e_lines = expected.split("\n")
                at = next(i for i, (a, b) in enumerate(zip(o_lines, e_lines))
                          if a != b)
                return block(o_lines[at], e_lines[at] + "  # fixed")
            return expected + "# fixed\n"
    raise AssertionError(f"unrouted codec prompt: {prompt[:80]!r}")


def test_sequential_stops_decided_cell_at_first_look():
    """0/5 decides `unusable` — both Wilson endpoints ladder to the same
    rung — so every cell stops at the first look: 5 calls, never 35."""
    from assay.codecs import CODECS, GRADES_FOR, Landing, probe_codecs
    from assay.stats import LOOK_SCHEDULE

    backend = ScriptedBackend(never_lands_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=35,
                          look_schedule=LOOK_SCHEDULE)

    assert result["json_object"]["small"].n == 5
    for codec in CODECS:
        for grade in GRADES_FOR[codec]:
            assert result[codec][grade] == Landing(lands=0.0,
                                                   lands_applies=0.0, n=5)
    assert len(backend.calls) == 12 * 5  # 12 cells, one look each


def test_sequential_runs_undecided_cell_to_cap():
    """A cell landing 4 of every 5 straddles risky/ready at 5, 10 and 20:
    it must spend the whole schedule and report n == 35."""
    from assay.codecs import GRADES, JSON_DEEP_GRADES, probe_codecs
    from assay.stats import LOOK_SCHEDULE

    backend = ScriptedBackend(json_four_of_five_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=35,
                          look_schedule=LOOK_SCHEDULE)

    for grade in GRADES:
        cell = result["json_object"][grade]
        assert cell.n == 35
        assert cell.lands == 28 / 35  # 7 rounds x 4 landing tasks
    # ...while the never-landing patch cells stop at the first look, as
    # do the three deep json cells this script cannot satisfy.
    for codec in ("search_replace", "whole_file"):
        for grade in GRADES:
            assert result[codec][grade].n == 5
    for grade in JSON_DEEP_GRADES:
        assert result["json_object"][grade].n == 5
    assert len(backend.calls) == 3 * 35 + 9 * 5


def test_sequential_stops_at_an_intermediate_look():
    """EVERY look is a stopping point, not just the first and the cap.
    A cell landing one task in five is undecided at look 5
    (wilson95(1, 5) = [0.036, 0.624], unusable/risky) and decided at
    look 10 (wilson95(2, 10) = [0.057, 0.510], both unusable), so it
    must report n == 10 — the middle of the schedule."""
    from assay.codecs import GRADES, probe_codecs
    from assay.stats import LOOK_SCHEDULE

    backend = ScriptedBackend(json_one_of_five_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=35,
                          look_schedule=LOOK_SCHEDULE)

    for grade in GRADES:
        cell = result["json_object"][grade]
        assert cell.n == 10, "a look after the first one never fired"
        assert cell.n not in (LOOK_SCHEDULE[0], LOOK_SCHEDULE[-1])
        assert cell.lands == 2 / 10  # one landing task x two rounds
    # 3 flat json cells to look 10; the 6 patch and 3 deep cells decide
    # at the first look.
    assert len(backend.calls) == 3 * 10 + 9 * 5


def test_no_schedule_is_exactly_the_old_behavior():
    """look_schedule=None (and its default) is fixed-rep sampling: the
    v1.4 matrix, cell for cell."""
    from assay.codecs import Landing, probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    fixed = probe_codecs(backend, make_meter(), n_per_cell=5)
    explicit = probe_codecs(ScriptedBackend(grade_matrix_script),
                            make_meter(), n_per_cell=5, look_schedule=None)

    assert explicit == fixed
    assert fixed["search_replace"]["tiny"] == Landing(lands=1.0,
                                                      lands_applies=1.0, n=5)
    for grade in ("small", "medium"):
        assert fixed["search_replace"][grade] == Landing(lands=0.0,
                                                         lands_applies=0.0, n=5)
    for grade in ("tiny", "small", "medium"):
        assert fixed["whole_file"][grade] == Landing(lands=1.0,
                                                     lands_applies=1.0, n=5)
        assert fixed["json_object"][grade] == Landing(lands=1.0,
                                                      lands_applies=1.0, n=5)
    for grade in ("nested", "tabular", "constrained"):
        assert fixed["json_object"][grade] == Landing(lands=1.0,
                                                      lands_applies=1.0, n=5)
    assert len(backend.calls) == 60


def test_no_schedule_never_stops_early():
    """Without a schedule there are no looks: an all-failing cell that a
    schedule would decide at 5 still spends every fixed rep. The v1.4
    attempt order (reps of one task consecutively) is preserved too, so
    replay transcripts of quick/family runs do not shift."""
    from assay.codecs import CODECS, GRADES_FOR, probe_codecs

    backend = ScriptedBackend(never_lands_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=35)

    for codec in CODECS:
        for grade in GRADES_FOR[codec]:
            assert result[codec][grade].n == 35
    assert len(backend.calls) == 12 * 35
    first_cell = [call["prompt"] for call in backend.calls[:35]]
    assert len(set(first_cell)) == 5              # five heterogeneous tasks
    assert len(set(first_cell[:7])) == 1          # 35 // 5 reps, task-major


def test_stop_test_uses_applies_lens_for_patch_codecs():
    """The stop test reads the cell's VERDICT lens. search_replace and
    whole_file are judged applies-and-parses: 5/5 applies is undecided,
    so the cell continues — while the byte-equality count (0/5) would
    have decided it unusable at the first look."""
    from assay.codecs import GRADES_FOR, probe_codecs
    from assay.stats import LOOK_SCHEDULE

    backend = ScriptedBackend(applies_but_not_equal_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=35,
                          look_schedule=LOOK_SCHEDULE)

    for codec in ("search_replace", "whole_file"):
        for grade in GRADES_FOR[codec]:
            cell = result[codec][grade]
            assert cell.lands == 0.0
            assert cell.lands_applies == 1.0
            assert cell.n == 35, "the byte-equality lens stopped this cell"
    # json's lens is byte-equality; 0/5 decides — every grade of it.
    for grade in GRADES_FOR["json_object"]:
        assert result["json_object"][grade].n == 5


def test_look_points_come_from_the_given_schedule():
    """The looks are the schedule's members, not a hardcoded 5: under
    (10, 20) a decided-at-5 cell keeps sampling until n == 10."""
    from assay.codecs import probe_codecs

    backend = ScriptedBackend(never_lands_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=35,
                          look_schedule=(10, 20))

    assert result["json_object"]["small"].n == 10
    assert len(backend.calls) == 12 * 10


def test_last_schedule_entry_is_the_cap_and_n_per_cell_is_ignored():
    from assay.codecs import probe_codecs

    backend = ScriptedBackend(json_four_of_five_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=35,
                          look_schedule=(5, 10))
    assert result["json_object"]["small"].n == 10  # the cap, not 35

    other = ScriptedBackend(json_four_of_five_script)
    result = probe_codecs(other, make_meter(), n_per_cell=1,
                          look_schedule=(5, 10))
    assert result["json_object"]["small"].n == 10  # n_per_cell ignored


def test_schedule_spreads_attempts_round_robin_across_tasks():
    """One rep across ALL of the cell's heterogeneous tasks per round —
    never repeated draws of one prompt (the v1.3 rule)."""
    from assay.codecs import JSON_DIRECTIVE, probe_codecs
    from assay.stats import LOOK_SCHEDULE

    backend = ScriptedBackend(json_four_of_five_script)
    probe_codecs(backend, make_meter(), n_per_cell=35,
                 look_schedule=LOOK_SCHEDULE)

    json_prompts = [call["prompt"] for call in backend.calls
                    if call["prompt"].startswith(JSON_DIRECTIVE)]
    cell = json_prompts[:35]  # the first json cell, run to the cap
    assert len(set(cell)) == 5
    assert all(cell[i] == cell[i % 5] for i in range(35))  # round-robin
    assert set(Counter(cell).values()) == {7}  # seven rounds, evenly spread


def test_budget_exhaustion_midschedule_keeps_partial_n():
    """A budget stop between looks records the honest partial n — never
    the look it was heading for, never a zero nobody measured."""
    from assay.codecs import Landing, probe_codecs
    from assay.stats import LOOK_SCHEDULE

    backend = ScriptedBackend(never_lands_script)
    result = probe_codecs(backend, make_meter(max_calls=3), n_per_cell=35,
                          look_schedule=LOOK_SCHEDULE)

    assert result["search_replace"]["tiny"] == Landing(lands=0.0,
                                                       lands_applies=0.0, n=3)
    assert result["search_replace"]["small"] == Landing(lands=None,
                                                        lands_applies=None, n=0)
    assert result["json_object"]["medium"] == Landing(lands=None,
                                                      lands_applies=None, n=0)


def test_empty_schedule_is_rejected_not_coerced():
    """An empty schedule is neither fixed-n nor sequential — refuse it
    rather than silently sampling under a rule nobody registered."""
    from assay.codecs import probe_codecs

    backend = ScriptedBackend(never_lands_script)
    with pytest.raises(ValueError, match="look_schedule"):
        probe_codecs(backend, make_meter(), n_per_cell=5, look_schedule=())
    assert backend.calls == []


def test_schedule_seeds_increment_once_per_attempt():
    """One seed per attempt, contiguous from seed_base — a stopped cell
    must not leave a gap or replay a seed."""
    from assay.codecs import probe_codecs
    from assay.stats import LOOK_SCHEDULE

    backend = ScriptedBackend(json_four_of_five_script)
    probe_codecs(backend, make_meter(), n_per_cell=35, seed_base=500,
                 look_schedule=LOOK_SCHEDULE)

    seeds = [call["kwargs"]["seed"] for call in backend.calls]
    assert len(seeds) == 3 * 35 + 9 * 5  # 3 flat json cells to the cap
    assert seeds == list(range(500, 500 + len(seeds)))


def test_stopped_on_rule_separates_a_decision_from_a_dead_meter():
    """The orchestrator asks each cell whether it ended ITSELF. A cell
    at the cap did; a cell decided at a look did; a cell cut off between
    looks (or never attempted) did not — and only the last kind means
    the budget died."""
    from assay.codecs import Landing, stopped_on_rule
    from assay.stats import LOOK_SCHEDULE

    def cell(rate, n):
        return Landing(lands=rate, lands_applies=rate, n=n)

    # The cap: every scheduled attempt was made, decided or not.
    assert stopped_on_rule("json_object", cell(1.0, 35), LOOK_SCHEDULE)
    # A look that decided the rung (0/5 is entirely below the risky floor).
    assert stopped_on_rule("json_object", cell(0.0, 5), LOOK_SCHEDULE)
    # A look that did NOT decide cannot be where sampling ended.
    assert not stopped_on_rule("json_object", cell(1.0, 5), LOOK_SCHEDULE)
    # Between looks, and never attempted: the meter, not the rule.
    assert not stopped_on_rule("json_object", cell(0.0, 12), LOOK_SCHEDULE)
    assert not stopped_on_rule(
        "json_object", Landing(lands=None, lands_applies=None, n=0),
        LOOK_SCHEDULE)


def test_stopped_on_rule_reads_each_codec_s_verdict_lens():
    """Same rates, different lens: a patch cell that applies every time
    but never matches byte-for-byte is undecided at n=5 (its verdict
    reads applies-and-parses), while the json cell IS decided."""
    from assay.codecs import Landing, stopped_on_rule
    from assay.stats import LOOK_SCHEDULE

    applies_only = Landing(lands=0.0, lands_applies=1.0, n=5)
    assert not stopped_on_rule("search_replace", applies_only, LOOK_SCHEDULE)
    assert not stopped_on_rule("whole_file", applies_only, LOOK_SCHEDULE)
    assert stopped_on_rule("json_object", Landing(lands=0.0, lands_applies=0.0,
                                                  n=5), LOOK_SCHEDULE)


def test_the_stop_test_reads_the_shared_lens_registry(monkeypatch):
    """The stop lens is NOT a rule this module keeps for itself.

    v1.5 spelled ``codec == "json_object"`` here and spelled the same
    rule again in profile's verdict layer; a change to one was a silent
    disagreement with the other. Both now read ``stats.VERDICT_LENS``,
    and this test proves it by moving the registry underneath them.
    """
    from assay import stats
    from assay.codecs import VERDICT_LENS, Landing, stopped_on_rule

    assert VERDICT_LENS is stats.VERDICT_LENS  # the object, not a copy

    # 0/5 byte-equality decides unusable; 5/5 applies-and-parses decides
    # nothing (three rungs still live). Same cell, opposite answers.
    cell = Landing(lands=0.0, lands_applies=1.0, n=5)
    assert stopped_on_rule("json_object", cell, stats.LOOK_SCHEDULE) is True

    monkeypatch.setitem(stats.VERDICT_LENS, "json_object", "lands_applies")
    assert stopped_on_rule("json_object", cell, stats.LOOK_SCHEDULE) is False


# --- the subset filter (v1.7): measure some codecs, name the rest ---------


def test_only_measures_the_named_codec_and_leaves_the_others_unmeasured():
    """`only` is a SUBSET of the matrix, not a matrix with quiet holes.

    The budget-mode consumer preflights the json half and the patch half
    separately, so it must be able to buy one without buying the other —
    and the half it did not buy has to come back `n == 0`, which is the
    value run.py's dropped loop names cell by cell.
    """
    from assay.codecs import GRADES_FOR, Landing, probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=5,
                          only=("json_object",))

    for grade in GRADES_FOR["json_object"]:
        assert result["json_object"][grade].n == 5, grade
    for codec in ("search_replace", "whole_file"):
        for grade in GRADES_FOR[codec]:
            assert result[codec][grade] == Landing(lands=None,
                                                   lands_applies=None, n=0)
    # Not one call was spent on the codecs nobody asked for.
    assert len(backend.calls) == 5 * len(GRADES_FOR["json_object"])


def test_only_the_patch_codecs_leaves_the_json_cells_unmeasured():
    from assay.codecs import GRADES_FOR, Landing, probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    result = probe_codecs(backend, make_meter(), n_per_cell=5,
                          only=("search_replace", "whole_file"))

    for codec in ("search_replace", "whole_file"):
        for grade in GRADES_FOR[codec]:
            assert result[codec][grade].n == 5, (codec, grade)
    for grade in GRADES_FOR["json_object"]:
        assert result["json_object"][grade] == Landing(lands=None,
                                                       lands_applies=None, n=0)
    assert len(backend.calls) == 5 * 3 * 2


def test_only_none_is_exactly_the_whole_matrix():
    """The default is the behavior every committed profile was measured
    under: no filter, every cell."""
    from assay.codecs import probe_codecs

    filtered = probe_codecs(ScriptedBackend(grade_matrix_script), make_meter(),
                            n_per_cell=5, only=None)
    unfiltered = probe_codecs(ScriptedBackend(grade_matrix_script),
                              make_meter(), n_per_cell=5)
    assert filtered == unfiltered
    assert all(cell.n == 5 for grades in filtered.values()
               for cell in grades.values())


def test_only_rejects_a_codec_it_does_not_know():
    """A misspelt codec name is a caller bug, and a silent no-op would
    hand back a matrix of unmeasured cells that reads exactly like a
    budget death. It raises before a single call is charged."""
    from assay.codecs import probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    meter = make_meter()
    with pytest.raises(ValueError) as excinfo:
        probe_codecs(backend, meter, n_per_cell=5, only=("json-object",))

    assert "json-object" in str(excinfo.value)
    assert backend.calls == []
    assert meter.spent.calls == 0


def test_an_empty_only_is_rejected_too():
    """Zero codecs is not a subset anyone can act on: it would spend
    nothing and report twelve unmeasured cells."""
    from assay.codecs import probe_codecs

    with pytest.raises(ValueError):
        probe_codecs(ScriptedBackend(grade_matrix_script), make_meter(),
                     n_per_cell=5, only=())


# --- cell_attempts: what a cell costs, from the probe's own enumeration ---


def test_cell_attempts_is_what_the_fixed_n_cell_actually_sends():
    from assay.codecs import CODECS, GRADES_FOR, cell_attempts, probe_codecs

    for codec in CODECS:
        backend = ScriptedBackend(grade_matrix_script)
        meter = make_meter()
        probe_codecs(backend, meter, n_per_cell=5, only=(codec,))
        declared = sum(cell_attempts(codec, grade, n_per_cell=5,
                                     look_schedule=None)
                       for grade in GRADES_FOR[codec])
        assert meter.spent.calls == declared, codec


def test_cell_attempts_is_the_cap_a_never_deciding_cell_runs_to():
    """Under a schedule the cap is the cost, and a cell that lands every
    time reaches it: 35/35 is the first look that decides `ready`."""
    from assay.codecs import CODECS, GRADES_FOR, cell_attempts, probe_codecs
    from assay.stats import LOOK_SCHEDULE

    for codec in CODECS:
        backend = ScriptedBackend(grade_matrix_script)
        meter = make_meter()
        probe_codecs(backend, meter, n_per_cell=35, only=(codec,),
                     look_schedule=LOOK_SCHEDULE)
        declared = sum(cell_attempts(codec, grade, n_per_cell=35,
                                     look_schedule=LOOK_SCHEDULE)
                       for grade in GRADES_FOR[codec])
        assert declared == len(GRADES_FOR[codec]) * LOOK_SCHEDULE[-1]
        # search_replace flubs small/medium under this script, and a
        # decided cell stops early — the declaration is a ceiling.
        assert meter.spent.calls <= declared
        if codec != "search_replace":
            assert meter.spent.calls == declared
