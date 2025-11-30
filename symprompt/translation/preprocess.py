from __future__ import annotations


class Preprocessor:
    """Minimal text normalization for SymPrompt translation."""

    def normalize(self, text: str) -> str:
        return " ".join(text.strip().split())

