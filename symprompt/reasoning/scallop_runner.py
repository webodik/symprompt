from __future__ import annotations

from typing import Dict, List, Tuple

from symprompt.reasoning.scallop_compiler import symil_to_scallop
from symprompt.symil.model import And, Atom, Implies, Rule, SymIL

try:
    import scallopy  # type: ignore[import]
    HAS_SCALLOPY = True
except Exception:  # pragma: no cover
    scallopy = None  # type: ignore[assignment]
    HAS_SCALLOPY = False


def run_scallop(symil: SymIL) -> Dict[str, str]:
    """
    Execute SymIL via a Scallop backend.

    If the scallopy binding is not available in this environment,
    this function returns UNKNOWN. When scallopy is installed, this
    executes a minimal fact-and-rule-based program for simple Level 0
    or Level 1 SymIL with an atomic query and classifies the result
    similarly to the ASP backend. More complex SymIL fragments fall
    back to UNKNOWN.
    """
    if not HAS_SCALLOPY:
        return {"status": "UNKNOWN"}

    # Currently we support a restricted subset:
    # - Level 0 or Level 1 SymIL
    # - Atomic query
    # - Rules limited to Horn-style implications with atomic heads and
    #   bodies that are conjunctions of atoms.
    if symil.level not in {0, 1} or symil.query is None or not isinstance(symil.query.prove, Atom):
        return {"status": "UNKNOWN"}

    # Keep the textual compiler in the loop for potential debugging or
    # future extensions, even though the current runner uses the Python
    # API directly.
    _program = symil_to_scallop(symil)
    _ = _program  # unused for now

    context_type = getattr(scallopy, "ScallopContext", None) or getattr(
        scallopy, "Context", None
    )
    if context_type is None:  # pragma: no cover - unexpected binding layout
        return {"status": "UNKNOWN"}

    ctx = context_type()

    # Declare relations with simple string-typed arguments.
    for predicate in symil.ontology.predicates:
        if predicate.arity <= 0:
            continue
        relation_types: Tuple[type, ...] = tuple([str] * predicate.arity)
        ctx.add_relation(predicate.name, relation_types)

    # Add extensional facts.
    facts_by_predicate: Dict[str, List[Tuple[str, ...]]] = {}
    for fact in symil.facts:
        if not isinstance(fact, Atom):
            continue
        if fact.pred not in facts_by_predicate:
            facts_by_predicate[fact.pred] = []
        facts_by_predicate[fact.pred].append(tuple(str(arg) for arg in fact.args))

    for predicate_name, tuples in facts_by_predicate.items():
        if tuples:
            ctx.add_facts(predicate_name, tuples)

    # Add simple Horn-style rules when present.
    def _rule_to_scallop(rule: Rule) -> str | None:
        body = rule.body
        if isinstance(body, Implies):
            conclusion = body.conclusion
            premise = body.premise
            if not isinstance(conclusion, Atom):
                return None
            head_str = f"{conclusion.pred}({', '.join(conclusion.args)})"
            premise_atoms: List[Atom] = []
            if isinstance(premise, Atom):
                premise_atoms = [premise]
            elif isinstance(premise, And):
                for formula in premise.formulas:
                    if isinstance(formula, Atom):
                        premise_atoms.append(formula)
                    else:
                        return None
            else:
                return None

            body_str = ", ".join(
                f"{atom.pred}({', '.join(atom.args)})" for atom in premise_atoms
            )
            return f"{head_str} :- {body_str}"

        if isinstance(body, Atom):
            return f"{body.pred}({', '.join(body.args)})"

        return None

    for rule in symil.rules:
        if not isinstance(rule, Rule):
            continue
        rule_str = _rule_to_scallop(rule)
        if not rule_str:
            continue
        ctx.add_rule(rule_str)

    ctx.run()

    query_atom = symil.query.prove
    query_tuple = tuple(str(arg) for arg in query_atom.args)
    try:
        relation_tuples = list(ctx.relation(query_atom.pred))
    except Exception:  # pragma: no cover - defensive against runtime issues
        return {"status": "UNKNOWN"}

    if query_tuple in relation_tuples:
        return {"status": "VALID"}
    if relation_tuples:
        return {"status": "NOT_VALID"}
    return {"status": "UNKNOWN"}
