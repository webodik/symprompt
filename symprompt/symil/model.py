from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class Predicate:
    name: str
    arity: int
    types: List[str]
    description: Optional[str] = None


@dataclass
class Ontology:
    predicates: List[Predicate] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)

    def get_predicate(self, name: str) -> Optional[Predicate]:
        for predicate in self.predicates:
            if predicate.name == name:
                return predicate
        return None


@dataclass
class Atom:
    pred: str
    args: List[str]


@dataclass
class Not:
    formula: "Formula"


@dataclass
class And:
    formulas: List["Formula"]


@dataclass
class Or:
    formulas: List["Formula"]


@dataclass
class Implies:
    premise: "Formula"
    conclusion: "Formula"


@dataclass
class ForAll:
    var: str
    type: str
    body: "Formula"


@dataclass
class Exists:
    var: str
    type: str
    body: "Formula"


Formula = Union[Atom, Not, And, Or, Implies, ForAll, Exists]


@dataclass
class Rule:
    forall: str
    type: str
    body: Formula


@dataclass
class Query:
    prove: Formula


@dataclass
class SymIL:
    ontology: Ontology
    facts: List[Formula] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    query: Optional[Query] = None
    constraints: List[Formula] = field(default_factory=list)
    level: int = 0

