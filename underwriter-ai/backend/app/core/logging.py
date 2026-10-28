"""Logging setup. Secrets are scrubbed from every log record."""

import logging
import re
import sys

# OpenRouter / OpenAI-style keys and bearer tokens.
_SECRET_PATTERNS = [re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}"), re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{8,}")]


def scrub(text: str) -> str:
    """Replace anything that looks like an API key or bearer token."""
    for pat in _SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


class _ScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = scrub(str(record.msg))
        if record.args and isinstance(record.args, tuple):
            # Only strings can carry secrets; numbers must stay numbers for %d / %f formatting.
            record.args = tuple(scrub(a) if isinstance(a, str) else a for a in record.args)
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, with the secret-scrubbing filter attached."""
    root = logging.getLogger()
    if any(isinstance(f, _ScrubFilter) for h in root.handlers for f in h.filters):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(_ScrubFilter())
    root.addHandler(handler)
    root.setLevel(level)
