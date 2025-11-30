from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SymILProfile:
    name: str
    predicate_vocabulary: List[str]
    allowed_constructs: List[str]
    preferred_solver: str
    default_level: int
    translation_hints: List[str] = field(default_factory=list)


_PROFILES: Dict[str, SymILProfile] = {
    "syllogism": SymILProfile(
        name="syllogism",
        predicate_vocabulary=["is_a", "has_property", "subset_of"],
        allowed_constructs=["L1"],
        preferred_solver="scallop",
        default_level=1,
        translation_hints=[
            "Focus on categorical statements like 'all A are B', 'some A are B', 'no A are B'.",
            "Use simple Horn-style implications rather than complex nesting.",
        ],
    ),
    "math": SymILProfile(
        name="math",
        predicate_vocabulary=["equals", "greater_than", "sum", "product"],
        allowed_constructs=["L0", "L1"],
        preferred_solver="z3",
        default_level=0,
        translation_hints=[
            "Treat numbers and arithmetic relations explicitly.",
            "Use equality and inequalities for numeric relationships.",
        ],
    ),
    "planning": SymILProfile(
        name="planning",
        predicate_vocabulary=["action", "precondition", "effect", "goal"],
        allowed_constructs=["L2"],
        preferred_solver="clingo",
        default_level=2,
        translation_hints=[
            "Identify actions, preconditions, effects, and goals.",
            "Allow more complex constraints and default rules for planning.",
        ],
    ),
    "legal": SymILProfile(
        name="legal",
        predicate_vocabulary=["obligation", "permission", "prohibition"],
        allowed_constructs=["L2"],
        preferred_solver="clingo",
        default_level=2,
        translation_hints=[
            "Model obligations, permissions, and prohibitions as predicates.",
            "Capture exceptions and default rules where possible.",
        ],
    ),
    "uncertain": SymILProfile(
        name="uncertain",
        predicate_vocabulary=["probably", "likely", "confidence"],
        allowed_constructs=["L1", "L2"],
        preferred_solver="scallop",
        default_level=1,
        translation_hints=[
            "Allow soft or probabilistic facts with confidence-like predicates.",
            "Represent uncertainty explicitly when it appears in the text.",
        ],
    ),
}


def get_profile(name: str) -> SymILProfile:
    return _PROFILES[name]


def list_profiles() -> List[SymILProfile]:
    return list(_PROFILES.values())
