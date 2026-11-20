"""Mask personal identifiers before text is logged or sent to the language model."""
import re

_PATTERNS = [
    (re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+'), '[EMAIL]'),
    (re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b', re.I), '[PAN]'),
    (re.compile(r'(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)'), '[AADHAAR]'),
    (re.compile(r'(?<!\d)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)'), '[PHONE]'),
    (re.compile(r'\b(my name is|name\s*[:=-])\s*[A-Za-z][A-Za-z.]*(?:\s+[A-Z][A-Za-z.]*)*', re.I), r'\1 [NAME]'),
]


def mask_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text
