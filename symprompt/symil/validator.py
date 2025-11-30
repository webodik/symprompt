from __future__ import annotations

from symprompt.symil.model import And, Atom, Exists, ForAll, Formula, Implies, Not, Or, SymIL


class SymILValidationError(Exception):
    """Raised when SymIL fails structural or semantic validation."""


class SymILValidator:
    """Performs structural checks on SymIL programs, including level rules."""

    def validate(self, symil: SymIL) -> SymIL:
        self._check_predicates(symil)
        self._check_level_constraints(symil)
        return symil

    def _check_predicates(self, symil: SymIL) -> None:
        def check_formula(formula: Formula) -> None:
            if isinstance(formula, Atom):
                predicate = symil.ontology.get_predicate(formula.pred)
                if predicate is None:
                    raise SymILValidationError(f"Unknown predicate: {formula.pred}")
                if len(formula.args) != predicate.arity:
                    raise SymILValidationError(
                        f"Arity mismatch for {formula.pred}: "
                        f"expected {predicate.arity}, got {len(formula.args)}"
                    )
            elif isinstance(formula, Not):
                check_formula(formula.formula)
            elif isinstance(formula, (And,)):
                for sub in formula.formulas:
                    check_formula(sub)
            elif isinstance(formula, (Or,)):
                for sub in formula.formulas:
                    check_formula(sub)
            elif isinstance(formula, Implies):
                check_formula(formula.premise)
                check_formula(formula.conclusion)
            elif isinstance(formula, (ForAll, Exists)):
                check_formula(formula.body)

        for fact in symil.facts:
            check_formula(fact)
        for rule in symil.rules:
            check_formula(rule.body)
        if symil.query is not None:
            check_formula(symil.query.prove)

    def _check_level_constraints(self, symil: SymIL) -> None:
        """
        Enforce basic rules for SymIL levels:
        - L0: facts + simple query only (no rules or constraints).
        - L1: allow Horn-style rules, disallow explicit Exists and
          non-Horn connectives (Not/Or) in rules/query.
        - L2: full language.
        """
        level = symil.level

        if level == 0:
            if symil.rules:
                raise SymILValidationError("Level 0 SymIL must not contain rules")
            if symil.constraints:
                raise SymILValidationError("Level 0 SymIL must not contain constraints")
            for fact in symil.facts:
                if not isinstance(fact, Atom):
                    raise SymILValidationError(
                        "Level 0 SymIL facts must be atomic predicates"
                    )
            if symil.query is not None and not isinstance(symil.query.prove, Atom):
                raise SymILValidationError(
                    "Level 0 SymIL queries must be atomic predicates"
                )

        if level == 1:
            # Disallow existential quantifiers in rules and query for L1.
            def check_no_exists(formula: Formula) -> None:
                if isinstance(formula, Exists):
                    raise SymILValidationError(
                        "Level 1 SymIL must not contain existential quantifiers"
                    )
                if isinstance(formula, Not):
                    check_no_exists(formula.formula)
                elif isinstance(formula, (And,)):
                    for sub in formula.formulas:
                        check_no_exists(sub)
                elif isinstance(formula, Implies):
                    check_no_exists(formula.premise)
                    check_no_exists(formula.conclusion)
                elif isinstance(formula, (ForAll,)):
                    check_no_exists(formula.body)

            for rule in symil.rules:
                check_no_exists(rule.body)
            if symil.query is not None:
                check_no_exists(symil.query.prove)

            # Enforce a Horn-style fragment: only Atom, And, Implies, and
            # top-level ForAll are permitted in rules and query bodies.
            def check_horn(formula: Formula) -> None:
                if isinstance(formula, Atom):
                    return
                if isinstance(formula, And):
                    for sub in formula.formulas:
                        check_horn(sub)
                    return
                if isinstance(formula, Implies):
                    check_horn(formula.premise)
                    check_horn(formula.conclusion)
                    return
                if isinstance(formula, ForAll):
                    check_horn(formula.body)
                    return
                raise SymILValidationError(
                    "Level 1 SymIL must use Horn-style formulas "
                    "(Atoms, And, Implies, and top-level ForAll only)"
                )

            for rule in symil.rules:
                check_horn(rule.body)
            if symil.query is not None:
                check_horn(symil.query.prove)
