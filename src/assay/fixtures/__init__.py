"""Graded codec fixtures (spec §7).

Three committed text files, each a small self-contained Python module
with one deliberate dropped-result defect on a stated line. Grades are
sized by characters (proxy for tokens at ~5 chars/token): tiny <= 200,
small 400-700, medium 1400-2200.

`EXPECTED` holds one ``(grade, filename, instruction, original,
expected)`` tuple per grade. The instruction names the file and states
the failing behavior; it never quotes the fixed line itself.
"""

from pathlib import Path

_DIR = Path(__file__).resolve().parent

# grade -> (defective line, fixed line); the defect is always a
# dropped-return on exactly one line.
_DEFECTS = {
    "tiny": (
        "    subtotal * 1.08          # BUG: result dropped",
        "    return subtotal * 1.08",
    ),
    "small": (
        "        amount * BULK_DISCOUNT          # BUG: discounted amount dropped",
        "        return amount * BULK_DISCOUNT",
    ),
    "medium": (
        "    DEFAULT_RESTOCK_LEVEL - current          # BUG: computed amount dropped",
        "    return DEFAULT_RESTOCK_LEVEL - current",
    ),
}

# States the failing behavior; never quotes the fixed line.
_INSTRUCTIONS = {
    "tiny": (
        "In tiny.py, total([10]) returns None; it should return 10.8. "
        "Fix the single defective line."
    ),
    "small": (
        "In small.py, apply_discount(100.0, 10) returns 100.0; it should "
        "return 95.0. Fix the single defective line."
    ),
    "medium": (
        "In medium.py, restock_amount({'a': 3}, 'a') returns None; it "
        "should return 17. Fix the single defective line."
    ),
}


def _entry(grade: str) -> tuple[str, str, str, str, str]:
    original = (_DIR / f"{grade}.py.txt").read_text(encoding="utf-8")
    defective, fixed = _DEFECTS[grade]
    if original.count(defective) != 1:
        raise ValueError(f"fixture {grade!r} lost its stated defect line")
    expected = original.replace(defective, fixed)
    return (grade, f"{grade}.py", _INSTRUCTIONS[grade], original, expected)


EXPECTED: tuple[tuple[str, str, str, str, str], ...] = tuple(
    _entry(grade) for grade in ("tiny", "small", "medium")
)
