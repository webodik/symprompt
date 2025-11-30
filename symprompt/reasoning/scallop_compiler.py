from __future__ import annotations

from typing import List

from symprompt.symil.model import And, Atom, ForAll, Formula, Implies, Rule, SymIL


def _atom_to_scallop(atom: Atom) -> str:
    args = ", ".join(atom.args)
    return f"{atom.pred}({args})"


def _formula_to_scallop_rule(formula: Formula) -> str | None:
    if isinstance(formula, Atom):
        return f"{_atom_to_scallop(formula)}."
    if isinstance(formula, Implies):
        head = _atom_to_scallop(formula.conclusion) if isinstance(formula.conclusion, Atom) else None
        if head is None:
            return None
        if isinstance(formula.premise, Atom):
            body = _atom_to_scallop(formula.premise)
        elif isinstance(formula.premise, And):
            body_parts = [_atom_to_scallop(a) for a in formula.premise.formulas if isinstance(a, Atom)]
            body = ", ".join(body_parts)
        else:
            return None
        return f"{head} :- {body}."
    if isinstance(formula, ForAll):
        return _formula_to_scallop_rule(formula.body)
    return None


def symil_to_scallop(symil: SymIL) -> str:
    """
    Compile a restricted subset of SymIL into a Scallop-style Datalog program.

    This is a text-only representation suitable for later feeding into
    a Scallop engine or wrapper.
    """
    lines: List[str] = []

    for fact in symil.facts:
        if isinstance(fact, Atom):
            lines.append(f"{_atom_to_scallop(fact)}.")

    for rule in symil.rules:
        if not isinstance(rule, Rule):
            continue
        stmt = _formula_to_scallop_rule(rule.body)
        if stmt:
            lines.append(stmt)

    return "\n".join(lines)

