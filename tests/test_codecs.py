"""Tests for codec fixtures, appliers, and probes (plan Task 9, spec §7)."""

import pytest

from assay import fixtures
from assay.codecs import (
    CodecDirectives,
    apply_search_replace,
    apply_whole_file,
    landing_equal,
    validate_json_object,
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


# --- fixtures -------------------------------------------------------------


def test_fixture_set_v2_integrity():
    from assay import fixtures

    assert fixtures.FIXTURE_SET == "codec-fixtures-v2"
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


def grade_matrix_script(prompt):
    """Lands json everywhere, whole_file everywhere, search_replace on
    tiny only — flubs small and medium. Routes every v2 fixture task."""
    from assay import fixtures
    from assay.codecs import JSON_DIRECTIVE

    if prompt.startswith(JSON_DIRECTIVE):
        return '{"name": "x", "count": 3, "tags": ["t"]}'
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
    assert seeds == list(range(500, 500 + 45))  # 3 codecs x 3 grades x 5 tasks


def test_probe_charges_the_meter_per_call():
    from assay.codecs import probe_codecs

    backend = ScriptedBackend(grade_matrix_script)
    meter = make_meter()
    probe_codecs(backend, meter, n_per_cell=1)

    assert meter.spent.calls == len(backend.calls) == 45  # 9 cells x 5 tasks
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
    assert all(marker in p for p in prompts), "custom directive missing"
    # And the built-in texts must NOT appear anywhere.
    assert not any("character for character" in p for p in prompts)
