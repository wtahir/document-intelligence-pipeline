# pii_redactor.py
# Redacts Personally Identifiable Information (PII) from text before
# sending it to external LLM APIs.
#
# Insurance claim documents contain sensitive data:
#   - Names, addresses, phone numbers
#   - IBANs, email addresses
#   - Policy numbers
#
# This module provides:
#   1. redact() — replaces PII with placeholder tokens
#   2. unredact() — restores original values from a mapping
#
# The mapping is kept in-memory per call and never sent to the LLM.
# This ensures the model sees "[IBAN_1]" instead of real bank details.

import re
from typing import Tuple

# ─── PII patterns ─────────────────────────────────────────────
# Order matters: more specific patterns first to avoid partial matches.

_PII_PATTERNS = [
    # IBAN (German format: DE## #### #### #### #### ##)
    (
        "IBAN",
        re.compile(r'\bDE\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{2}\b'),
    ),
    # Email addresses
    (
        "EMAIL",
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    ),
    # Phone numbers (German format: +49 1XX XXXXXXX or similar)
    (
        "PHONE",
        re.compile(r'\+49\s?\d{2,3}\s?\d{6,8}\b'),
    ),
    # German ZIP + city (5-digit zip followed by city name)
    (
        "ADDRESS",
        re.compile(r'\b\d{5}\s+[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*\b'),
    ),
]


def redact(text: str, redact_pii: bool = True) -> Tuple[str, dict]:
    """
    Replaces PII in text with numbered placeholders.

    Args:
        text: The input text potentially containing PII.
        redact_pii: If False, returns text unchanged (for easy toggling).

    Returns:
        (redacted_text, mapping) where mapping is {placeholder: original_value}.
        Use mapping with unredact() to restore originals in the LLM response.
    """
    if not redact_pii or not text:
        return text, {}

    mapping = {}
    redacted = text
    counter = {}

    for label, pattern in _PII_PATTERNS:
        matches = list(pattern.finditer(redacted))
        # Process in reverse order to preserve string positions
        for match in reversed(matches):
            original = match.group()
            # Avoid re-redacting already-redacted tokens
            if original.startswith("[") and original.endswith("]"):
                continue

            counter[label] = counter.get(label, 0) + 1
            placeholder = f"[{label}_{counter[label]}]"

            # Only map unique values
            if original not in mapping.values():
                mapping[placeholder] = original

            redacted = redacted[:match.start()] + placeholder + redacted[match.end():]

    return redacted, mapping


def unredact(text: str, mapping: dict) -> str:
    """
    Restores original PII values in text using the mapping from redact().
    Useful for restoring PII in the final answer if needed for display.
    """
    if not mapping:
        return text

    result = text
    for placeholder, original in mapping.items():
        result = result.replace(placeholder, original)

    return result


def redact_chunks(chunks: list[dict]) -> Tuple[list[dict], dict]:
    """
    Redacts PII from a list of retrieved chunks before building LLM context.
    Returns redacted chunks and a combined mapping for unredaction.
    """
    combined_mapping = {}
    redacted_chunks = []

    for chunk in chunks:
        redacted_text, mapping = redact(chunk.get("text", ""))
        combined_mapping.update(mapping)
        # Create a copy with redacted text
        redacted_chunk = {**chunk, "text": redacted_text}
        redacted_chunks.append(redacted_chunk)

    return redacted_chunks, combined_mapping
