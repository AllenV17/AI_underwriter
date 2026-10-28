"""Currency-unit detection and number parsing for Indian financial statements.

Statements state their unit in a caption such as "(₹ in million)" or "Rs. in lakhs".
Values are always kept as printed (`value_original` + `unit`) and converted to absolute
rupees (`value_inr`) for calculation; nothing is converted without a detected unit.
"""

import re

MULTIPLIERS: dict[str, float] = {
    "rupees": 1.0, "thousand": 1e3, "lakh": 1e5, "million": 1e6, "crore": 1e7, "billion": 1e9,
}

_CUR = r"(?:₹|rs\.?|inr|rupees|`)"  # older PDFs render the rupee sign as a backtick
_UNIT_WORDS = r"(thousands?|'000|lakhs?|lacs?|millions?|mn|crores?|cr\.?|billions?|bn)"
_UNIT_PATTERNS = [
    re.compile(rf"\(\s*(?:amount\s*|all\s+amounts?\s*)?(?:in\s*)?{_CUR}\s*(?:in\s*)?{_UNIT_WORDS}", re.I),
    re.compile(rf"{_CUR}\s*in\s*{_UNIT_WORDS}", re.I),
    re.compile(rf"in\s*(?:indian\s*)?{_CUR}\s*{_UNIT_WORDS}", re.I),
    re.compile(rf"amounts?\s+(?:are\s+)?(?:in|expressed in)\s+(?:{_CUR}\s*)?{_UNIT_WORDS}", re.I),
]


def canonical_unit(word: str) -> str:
    w = word.lower().strip(". ")
    if w.startswith(("thousand", "'000")):
        return "thousand"
    if w.startswith(("lakh", "lac")):
        return "lakh"
    if w in ("mn",) or w.startswith("million"):
        return "million"
    if w.startswith(("crore", "cr")):
        return "crore"
    if w in ("bn",) or w.startswith("billion"):
        return "billion"
    raise ValueError(f"Unknown unit word: {word}")


def detect_unit(text: str) -> tuple[str, str] | None:
    """Return (canonical unit, matched caption) from the first unit caption in the text."""
    for pat in _UNIT_PATTERNS:
        m = pat.search(text)
        if m:
            return canonical_unit(m.group(1)), m.group(0)
    return None


def to_inr(value: float, unit: str) -> float:
    """Convert a value in the given unit to absolute rupees."""
    return value * MULTIPLIERS[unit]


_NUM = re.compile(r"^\(?-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?\)?$|^\(?-?\d+(?:\.\d+)?\)?$")
NIL_TOKENS = {"-", "–", "—", "nil", "NIL", "Nil", "--"}


def is_number_token(tok: str) -> bool:
    return bool(_NUM.match(tok.strip())) or tok.strip() in NIL_TOKENS


def parse_number(tok: str) -> float | None:
    """'(1,234.50)' -> -1234.5; '1,23,456' -> 123456; '-' / 'Nil' -> None (reported as nil, not zero)."""
    t = tok.strip()
    if t in NIL_TOKENS:
        return None
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()").replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v
