from __future__ import annotations

from symprompt.symil.model import (
    Atom,
    Exists,
    Ontology,
    Predicate,
    Query,
    Rule,
    SymIL,
)
from symprompt.symil.validator import SymILValidationError, SymILValidator


def test_symil_validator_accepts_valid_example() -> None:
    ontology = Ontology(
        predicates=[
            Predicate(name="mammal", arity=1, types=["entity"]),
            Predicate(name="animal", arity=1, types=["entity"]),
        ]
    )
    rule = Rule(
        forall="X",
        type="entity",
        body=Atom(pred="mammal", args=["X"]),
    )
    symil = SymIL(
        ontology=ontology,
        facts=[],
        rules=[rule],
        query=Query(prove=Atom(pred="mammal", args=["X"])),
        level=1,
    )

    validator = SymILValidator()
    validated = validator.validate(symil)
    assert validated is symil


def test_symil_validator_rejects_unknown_predicate() -> None:
    ontology = Ontology(predicates=[])
    symil = SymIL(
        ontology=ontology,
        facts=[Atom(pred="unknown_pred", args=["X"])],
        rules=[],
        query=None,
        level=0,
    )
    validator = SymILValidator()

    try:
        validator.validate(symil)
        assert False, "Expected SymILValidationError for unknown predicate"
    except SymILValidationError:
        pass


def test_level_zero_disallows_rules_and_constraints() -> None:
    ontology = Ontology(predicates=[Predicate(name="p", arity=1, types=["entity"])])
    symil = SymIL(
        ontology=ontology,
        facts=[],
        rules=[Rule(forall="X", type="entity", body=Atom(pred="p", args=["X"]))],
        query=None,
        constraints=[],
        level=0,
    )
    validator = SymILValidator()

    try:
        validator.validate(symil)
        assert False, "Expected SymILValidationError for rules at level 0"
    except SymILValidationError:
        pass


def test_level_one_disallows_exists_in_rules() -> None:
    ontology = Ontology(predicates=[Predicate(name="p", arity=1, types=["entity"])])
    rule = Rule(
        forall="X",
        type="entity",
        body=Exists(var="Y", type="entity", body=Atom(pred="p", args=["Y"])),
    )
    symil = SymIL(
        ontology=ontology,
        facts=[],
        rules=[rule],
        query=None,
        constraints=[],
        level=1,
    )
    validator = SymILValidator()

    try:
        validator.validate(symil)
        assert False, "Expected SymILValidationError for Exists at level 1"
    except SymILValidationError:
        pass
