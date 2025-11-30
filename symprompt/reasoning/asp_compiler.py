from __future__ import annotations

from typing import List

from symprompt.symil.model import And, Atom, ForAll, Formula, Implies, Rule, SymIL


def _atom_to_asp(atom: Atom) -> str:
    args = ", ".join(atom.args)
    return f"{atom.pred}({args})"


def _formula_to_asp_body(formula: Formula) -> List[str]:
    if isinstance(formula, Atom):
        return [_atom_to_asp(formula)]
    if isinstance(formula, And):
        parts: List[str] = []
        for sub in formula.formulas:
            parts.extend(_formula_to_asp_body(sub))
        return parts
    if isinstance(formula, Implies):
        # For body, use premise only; conclusion becomes head.
        return _formula_to_asp_body(formula.premise)
    if isinstance(formula, ForAll):
        return _formula_to_asp_body(formula.body)
    return []


def _formula_to_asp_head(formula: Formula) -> str | None:
    if isinstance(formula, Atom):
        return _atom_to_asp(formula)
    if isinstance(formula, Implies):
        return _formula_to_asp_head(formula.conclusion)
    if isinstance(formula, ForAll):
        return _formula_to_asp_head(formula.body)
    return None


def symil_to_asp(symil: SymIL) -> str:
    """
    Compile a restricted subset of SymIL into an ASP program string.

    Facts become:
        pred(arg1, arg2).

    Simple universally quantified implications become Horn rules:
        q(X) :- p(X).
    """
    lines: List[str] = []

    for fact in symil.facts:
        if isinstance(fact, Atom):
            lines.append(f"{_atom_to_asp(fact)}.")

    for rule in symil.rules:
        if not isinstance(rule, Rule):
            continue
        head = _formula_to_asp_head(rule.body)
        if head is None:
            continue
        body_parts = _formula_to_asp_body(rule.body)
        if not body_parts:
            lines.append(f"{head}.")
        else:
            body = ", ".join(body_parts)
            lines.append(f"{head} :- {body}.")

    return "\n".join(lines)

