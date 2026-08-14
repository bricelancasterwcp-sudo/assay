"""Graded codec fixtures, set ``codec-fixtures-v2`` (v1.3).

The v1 set had ONE task per grade — a single dropped-return defect —
so a cell's "landing rate" measured sampler variance on one prompt,
not codec capability (external review, 2026-08-13: "assay measures its
fixtures with great honesty"; also robigo CARRIED-DEBT lesson 4,
"hand-written fixtures have accidental properties real data lacks",
which this project recorded and then failed to apply to itself).

v2: three clean base modules (one per grade, sized tiny ≤ 250 chars,
small 400–700, medium 1100–2200) each carrying FIVE distinct defect
sites; one corruption per fixture across five defect classes —
``dropped_return``, ``off_by_one``, ``wrong_operator``,
``inverted_guard``, ``wrong_variable`` — for 15 tasks total, five
heterogeneous tasks per grade. Tasks within a grade share the base
module (context correlation is declared, not hidden; module diversity
is a future set). The set NAME travels in the codec lens and in
provenance: the fixture set is part of the instrument.

Each ``EXPECTED`` entry is ``(grade, defect_class, filename,
instruction, original, expected)``: ``expected`` is the clean base,
``original`` is the base with exactly one line swapped. The instruction
names the file and states the failing behavior; it never quotes the
fix. An authoring-integrity test asserts every clean line occurs
exactly once in its base and every original compiles and differs from
its base by exactly one line.
"""

from pathlib import Path

_DIR = Path(__file__).resolve().parent

FIXTURE_SET = "codec-fixtures-v2"

GRADES = ("tiny", "small", "medium")
DEFECT_CLASSES = (
    "dropped_return", "off_by_one", "wrong_operator",
    "inverted_guard", "wrong_variable",
)

# (grade, defect_class, clean_line, broken_line, instruction)
_DEFECTS = [
    # ---- tiny ----------------------------------------------------------
    ("tiny", "dropped_return",
     "    return subtotal * 1.08",
     "    subtotal * 1.08",
     "In tiny.py, total([10]) returns None; it should return 10.8. "
     "Fix the single defective line."),
    ("tiny", "off_by_one",
     "    return items[len(items) - 1]",
     "    return items[len(items) - 2]",
     "In tiny.py, last_of([4, 7, 9]) returns 7; it should return 9. "
     "Fix the single defective line."),
    ("tiny", "wrong_operator",
     "    if x > hi:",
     "    if x < hi:",
     "In tiny.py, clamp(9, 5) returns 9; it should return 5. "
     "Fix the single defective line."),
    ("tiny", "inverted_guard",
     "    if x > hi:",
     "    if not x > hi:",
     "In tiny.py, clamp(2, 5) returns 5; it should return 2. "
     "Fix the single defective line."),
    ("tiny", "wrong_variable",
     "        return hi",
     "        return x",
     "In tiny.py, clamp(9, 5) returns 9 instead of the cap; it should "
     "return 5. Fix the single defective line."),
    # ---- small ---------------------------------------------------------
    ("small", "dropped_return",
     "        return amount * BULK_DISCOUNT",
     "        amount * BULK_DISCOUNT",
     "In small.py, line_cost(2, 10) returns None; it should return 18.0 "
     "(the bulk discount applies). Fix the single defective line."),
    ("small", "off_by_one",
     "    for i in range(len(values)):",
     "    for i in range(len(values) - 1):",
     "In small.py, positions_of(5, [1, 5, 2, 5]) returns [1] and misses "
     "the final position; it should return [1, 3]. Fix the single "
     "defective line."),
    ("small", "wrong_operator",
     "        if best is None or unit < best:",
     "        if best is None or unit > best:",
     "In small.py, cheapest([(3, 1), (9, 1)]) returns 9; it should "
     "return 3. Fix the single defective line."),
    ("small", "inverted_guard",
     "    if quantity >= BULK_THRESHOLD:",
     "    if not quantity >= BULK_THRESHOLD:",
     "In small.py, line_cost(2, 10) returns 20 with no discount while "
     "line_cost(2, 3) gets one; the discount should apply at or above "
     "ten units. Fix the single defective line."),
    ("small", "wrong_variable",
     "        total = total + line_cost(unit, quantity)",
     "        total = total + line_cost(unit, unit)",
     "In small.py, order_total([(2, 5)]) returns 4; it should return "
     "10. Fix the single defective line."),
    # ---- medium --------------------------------------------------------
    ("medium", "dropped_return",
     "    return DEFAULT_RESTOCK_LEVEL - current",
     "    DEFAULT_RESTOCK_LEVEL - current",
     "In medium.py, restock_amount(20) returns None; it should return "
     "30. Fix the single defective line."),
    ("medium", "off_by_one",
     "    return ordered[n - 1]",
     "    return ordered[n]",
     "In medium.py, nth_cheapest(items, 1) returns the second-cheapest "
     "item; asking for position 1 should return the cheapest. Fix the "
     "single defective line."),
    ("medium", "wrong_operator",
     "        if best is None or item.price > best.price:",
     "        if best is None or item.price < best.price:",
     "In medium.py, priciest() returns the cheapest item; it should "
     "return the most expensive one. Fix the single defective line."),
    ("medium", "inverted_guard",
     "    if current >= DEFAULT_RESTOCK_LEVEL:",
     "    if current <= DEFAULT_RESTOCK_LEVEL:",
     "In medium.py, restock_amount(20) returns 0 and restock_amount(60) "
     "returns -10; amounts below the restock level should get a "
     "positive order and full ones zero. Fix the single defective "
     "line."),
    ("medium", "wrong_variable",
     "        return self.count * self.price",
     "        return self.count * self.count",
     "In medium.py, an Item with count 2 and price 7 reports value 4; "
     "it should report 14. Fix the single defective line."),
]


def _base(grade: str) -> str:
    return (_DIR / f"{grade}.py.txt").read_text(encoding="utf-8")


def _build_expected() -> list[tuple[str, str, str, str, str, str]]:
    entries = []
    for grade, defect_class, clean_line, broken_line, instruction in _DEFECTS:
        base = _base(grade)
        original = base.replace(clean_line + "\n", broken_line + "\n")
        entries.append((grade, defect_class, f"{grade}.py", instruction,
                        original, base))
    return entries


EXPECTED = _build_expected()
