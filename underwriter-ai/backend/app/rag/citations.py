"""Evidence formatting for prompts, and citation validation for answers.

The LLM sees evidence blocks labelled [S1], [S2]... and may cite only those labels. After
generation, labels are validated against the evidence actually supplied and rewritten to
"(Source: DRHP, <section>, page <n>)". A label that was not supplied is removed and
reported, so page numbers in an answer always come from retrieved metadata, never from
the model.
"""

import re
from dataclasses import dataclass, field

from app.services.retrieval_service import Evidence

_TAG = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def format_evidence(evidence: list[Evidence], max_chars: int = 2200) -> str:
    blocks = []
    for e in evidence:
        c = e.chunk
        text = c.text if len(c.text) <= max_chars else c.text[:max_chars] + " ..."
        where = f"{c.section_label}, page {c.page_number}" + (f", {c.subsection}" if c.subsection else "")
        blocks.append(f"[{e.ref}] ({where}; {c.chunk_type})\n{text}")
    return "\n\n".join(blocks)


@dataclass
class CitedAnswer:
    text: str
    citations: list[dict] = field(default_factory=list)
    invalid_refs: list[str] = field(default_factory=list)


def resolve_citations(answer: str, evidence: list[Evidence]) -> CitedAnswer:
    """Replace [S#] tags with real source references; drop tags that do not exist."""
    known = {e.ref: e for e in evidence}
    used: dict[str, Evidence] = {}
    invalid: list[str] = []

    def repl(m: re.Match) -> str:
        refs = [r.strip() for r in m.group(1).split(",")]
        parts = []
        for r in refs:
            if r in known:
                used[r] = known[r]
                c = known[r].chunk
                parts.append(f"{c.section_label}, page {c.page_number}")
            else:
                invalid.append(r)
        if not parts:
            return ""
        return "(Source: DRHP, " + "; ".join(dict.fromkeys(parts)) + ")"

    text = _TAG.sub(repl, answer)
    text = re.sub(r"[ \t]+\(Source", " (Source", text)
    cites = [{"ref": r, "page_number": e.chunk.page_number, "section": e.chunk.section_label,
              "subsection": e.chunk.subsection, "excerpt": e.chunk.text[:400]}
             for r, e in sorted(used.items(), key=lambda kv: int(kv[0][1:]))]
    return CitedAnswer(text=text.strip(), citations=cites, invalid_refs=sorted(set(invalid)))
