from __future__ import annotations

from dataclasses import dataclass
from typing import List

from symprompt.config import DEFAULT_FEATURE_CONFIG


LOGIC_KEYWORDS: List[str] = [
    "prove",
    "valid",
    "follows",
    "implies",
    "therefore",
    "if",
    "then",
    "all",
    "some",
    "no",
    "deduce",
    "infer",
    "conclude",
    "must be",
    "cannot be",
]

MATH_KEYWORDS: List[str] = [
    "sum",
    "product",
    "greater than",
    "less than",
    "equals",
    "difference",
]

PLANNING_KEYWORDS: List[str] = ["plan", "goal", "action", "steps", "schedule"]

LEGAL_KEYWORDS: List[str] = ["obligation", "permission", "prohibition", "contract"]


@dataclass
class PromptFeatures:
    length: int
    has_logic_keywords: bool
    has_numbers: bool
    complexity: float
    logical_depth: int
    constraint_count: int
    domain: str


def extract_features(prompt: str) -> PromptFeatures:
    text = prompt.lower()
    tokens = text.split()

    length = len(tokens)
    has_logic = any(keyword in text for keyword in LOGIC_KEYWORDS)
    has_numbers = any(any(ch.isdigit() for ch in token) for token in tokens)

    clause_separators = [";", "therefore", "thus", "hence"]
    clause_count = 1 + sum(text.count(sep) for sep in clause_separators)

    complexity = min(
        1.0,
        length / DEFAULT_FEATURE_CONFIG.complexity_length_divisor
        + DEFAULT_FEATURE_CONFIG.complexity_clause_weight * (clause_count - 1),
    )

    logical_depth = 1 if has_logic else 0
    if "if" in tokens and "then" in tokens:
        logical_depth = max(logical_depth, 2)

    constraint_count = text.count("subject to") + text.count("must") + text.count("cannot")

    domain = "syllogism"
    if any(keyword in text for keyword in MATH_KEYWORDS) or has_numbers:
        domain = "math"
    elif any(keyword in text for keyword in PLANNING_KEYWORDS):
        domain = "planning"
    elif any(keyword in text for keyword in LEGAL_KEYWORDS):
        domain = "legal"

    return PromptFeatures(
        length=length,
        has_logic_keywords=has_logic,
        has_numbers=has_numbers,
        complexity=complexity,
        logical_depth=logical_depth,
        constraint_count=constraint_count,
        domain=domain,
    )

