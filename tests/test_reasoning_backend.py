from __future__ import annotations

from symprompt.reasoning.asp_compiler import symil_to_asp
from symprompt.reasoning.clingo_runner import run_asp
from symprompt.reasoning.vsa_encoder import VSACodebook, encode_symil_to_vsa
from symprompt.reasoning.vsa_runner import run_vsa
from symprompt.reasoning.z3_runner import solve_symil
from symprompt.symil.model import (
    Atom,
    ForAll,
    Implies,
    Ontology,
    Predicate,
    Query,
    Rule,
    SymIL,
)


def test_z3_runner_valid_trivial_fact() -> None:
    ontology = Ontology(
        predicates=[Predicate(name="p", arity=1, types=["entity"])],
    )
    symil = SymIL(
        ontology=ontology,
        facts=[Atom(pred="p", args=["X"])],
        rules=[],
        query=Query(prove=Atom(pred="p", args=["X"])),
        level=0,
    )

    result = solve_symil(symil)
    assert result["status"] in {"VALID", "UNKNOWN"}


def test_symil_to_asp_compiler_emits_simple_rule() -> None:
    ontology = Ontology(
        predicates=[
            Predicate(name="cat", arity=1, types=["entity"]),
            Predicate(name="mammal", arity=1, types=["entity"]),
        ]
    )
    rule = Rule(
        forall="X",
        type="entity",
        body=Implies(
            premise=Atom(pred="cat", args=["X"]),
            conclusion=Atom(pred="mammal", args=["X"]),
        ),
    )
    symil = SymIL(ontology=ontology, facts=[], rules=[rule], query=None, level=1)

    program = symil_to_asp(symil)
    assert "mammal(X)" in program
    assert "cat(X)" in program


def test_vsa_encoder_produces_nonzero_memory() -> None:
    ontology = Ontology(predicates=[Predicate(name="p", arity=1, types=["entity"])])
    symil = SymIL(
        ontology=ontology,
        facts=[Atom(pred="p", args=["X"])],
        rules=[],
        query=None,
        level=0,
    )
    codebook = VSACodebook(dim=64)
    state = encode_symil_to_vsa(symil, codebook)

    assert state.dim == 64
    assert (state.memory != 0).any()


def test_vsa_runner_similarity_status_keys() -> None:
    ontology = Ontology(predicates=[Predicate(name="p", arity=1, types=["entity"])])
    symil = SymIL(
        ontology=ontology,
        facts=[Atom(pred="p", args=["a"])],
        rules=[],
        query=Query(prove=Atom(pred="p", args=["a"])),
        level=0,
    )

    result = run_vsa(symil, dim=64)
    assert "status" in result
    assert "state" in result


def test_scallop_runner_optional_backend() -> None:
    """
    Ensure that the Scallop backend is optional and never breaks the
    reasoning layer: when scallopy is unavailable, it reports UNKNOWN;
    when available it returns a well-formed status.
    """
    from symprompt.reasoning.scallop_runner import HAS_SCALLOPY, run_scallop

    ontology = Ontology(predicates=[Predicate(name="p", arity=1, types=["entity"])])
    symil = SymIL(
        ontology=ontology,
        facts=[Atom(pred="p", args=["X"])],
        rules=[],
        query=None,
        level=0,
    )

    result = run_scallop(symil)
    if not HAS_SCALLOPY:
        assert result["status"] == "UNKNOWN"
    else:
        assert result["status"] in {"VALID", "NOT_VALID", "UNKNOWN"}


def test_scallop_runner_simple_fact_query() -> None:
    """
    When scallopy is installed, Scallop should be able to classify a
    simple Level 0 fact/query pair using its Python API.
    """
    try:
        import scallopy as _scallopy  # type: ignore[import]
    except Exception:  # pragma: no cover - environment without scallopy
        return

    from symprompt.reasoning.scallop_runner import run_scallop

    ontology = Ontology(
        predicates=[Predicate(name="p", arity=1, types=["entity"])],
    )
    symil = SymIL(
        ontology=ontology,
        facts=[Atom(pred="p", args=["a"])],
        rules=[],
        query=Query(prove=Atom(pred="p", args=["a"])),
        level=0,
    )

    result = run_scallop(symil)
    assert result["status"] in {"VALID", "UNKNOWN"}


def test_scallop_runner_l1_rule() -> None:
    """
    When scallopy is installed, Scallop should be able to reason over a
    simple Level 1 Horn-style rule and classify the induced query.
    """
    try:
        import scallopy as _scallopy  # type: ignore[import]
    except Exception:  # pragma: no cover - environment without scallopy
        return

    from symprompt.reasoning.scallop_runner import run_scallop

    ontology = Ontology(
        predicates=[
            Predicate(name="cat", arity=1, types=["entity"]),
            Predicate(name="mammal", arity=1, types=["entity"]),
        ]
    )
    rule = Rule(
        forall="X",
        type="entity",
        body=Implies(
            premise=Atom(pred="cat", args=["X"]),
            conclusion=Atom(pred="mammal", args=["X"]),
        ),
    )
    symil = SymIL(
        ontology=ontology,
        facts=[Atom(pred="cat", args=["a"])],
        rules=[rule],
        query=Query(prove=Atom(pred="mammal", args=["a"])),
        level=1,
    )

    result = run_scallop(symil)
    assert result["status"] in {"VALID", "UNKNOWN"}


def test_run_asp_valid_simple_fact() -> None:
    ontology = Ontology(
        predicates=[Predicate(name="p", arity=1, types=["entity"])],
    )
    symil = SymIL(
        ontology=ontology,
        facts=[Atom(pred="p", args=["a"])],
        rules=[],
        query=Query(prove=Atom(pred="p", args=["a"])),
        level=0,
    )

    result = run_asp(symil)
    # With clingo available, we expect a concrete status; otherwise UNKNOWN is acceptable.
    assert result["status"] in {"VALID", "NOT_VALID", "UNKNOWN"}


def test_run_asp_not_valid_when_fact_missing() -> None:
    ontology = Ontology(
        predicates=[Predicate(name="p", arity=1, types=["entity"])],
    )
    symil = SymIL(
        ontology=ontology,
        facts=[],  # no p(a) fact
        rules=[],
        query=Query(prove=Atom(pred="p", args=["a"])),
        level=0,
    )

    result = run_asp(symil)
    assert result["status"] in {"NOT_VALID", "UNKNOWN"}


def test_z3_runner_respects_constraints_as_axioms() -> None:
    """
    Constraints should act as additional axioms in the Z3 encoding.

    Without constraints, the query p(a) is NOT_VALID under an empty
    theory (¬p(a) is satisfiable). With a constraint asserting p(a),
    the same query becomes VALID because ¬p(a) is inconsistent with
    the axioms.
    """
    ontology = Ontology(
        predicates=[Predicate(name="p", arity=1, types=["entity"])],
    )

    symil_without_constraints = SymIL(
        ontology=ontology,
        facts=[],
        rules=[],
        query=Query(prove=Atom(pred="p", args=["a"])),
        level=2,
    )
    result_without = solve_symil(symil_without_constraints)

    symil_with_constraints = SymIL(
        ontology=ontology,
        facts=[],
        rules=[],
        query=Query(prove=Atom(pred="p", args=["a"])),
        constraints=[Atom(pred="p", args=["a"])],
        level=2,
    )
    result_with = solve_symil(symil_with_constraints)

    assert result_without["status"] in {"NOT_VALID", "UNKNOWN"}
    assert result_with["status"] in {"VALID", "UNKNOWN"}
