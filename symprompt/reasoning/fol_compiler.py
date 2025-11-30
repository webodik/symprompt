from __future__ import annotations

from typing import Dict, List

import z3

from symprompt.symil.model import (
    And,
    Atom,
    Exists,
    ForAll,
    Formula,
    Implies,
    Not,
    Rule,
    SymIL,
)


def symil_to_z3(symil: SymIL) -> Dict[str, object]:
    """
    Compile a SymIL program into Z3 sorts, predicates, axioms, and query.
    """
    entity_sort = z3.DeclareSort("Entity")

    predicate_functions: Dict[str, z3.FuncDeclRef] = {}
    for predicate in symil.ontology.predicates:
        argument_sorts: List[z3.SortRef] = [entity_sort] * predicate.arity
        predicate_functions[predicate.name] = z3.Function(
            predicate.name, *argument_sorts, z3.BoolSort()
        )

    def encode_formula(formula: Formula, environment: Dict[str, z3.ExprRef]) -> z3.BoolRef:
        if isinstance(formula, Atom):
            function = predicate_functions[formula.pred]
            arguments = []
            for name in formula.args:
                if name not in environment:
                    environment[name] = z3.Const(name, entity_sort)
                arguments.append(environment[name])
            return function(*arguments)
        if isinstance(formula, Not):
            return z3.Not(encode_formula(formula.formula, environment))
        if isinstance(formula, And):
            return z3.And(
                *[encode_formula(sub_formula, environment) for sub_formula in formula.formulas]
            )
        if isinstance(formula, Implies):
            return z3.Implies(
                encode_formula(formula.premise, environment),
                encode_formula(formula.conclusion, environment),
            )
        if isinstance(formula, ForAll):
            variable = z3.Const(formula.var, entity_sort)
            new_environment = {**environment, formula.var: variable}
            return z3.ForAll([variable], encode_formula(formula.body, new_environment))
        if isinstance(formula, Exists):
            variable = z3.Const(formula.var, entity_sort)
            new_environment = {**environment, formula.var: variable}
            return z3.Exists([variable], encode_formula(formula.body, new_environment))
        raise TypeError(f"Unsupported formula type: {type(formula)}")

    axioms: List[z3.BoolRef] = []
    for fact in symil.facts:
        axioms.append(encode_formula(fact, {}))

    for rule in symil.rules:
        if isinstance(rule, Rule):
            variable = z3.Const(rule.forall, entity_sort)
            body_ref = encode_formula(rule.body, {rule.forall: variable})
            axioms.append(z3.ForAll([variable], body_ref))

    for constraint in symil.constraints:
        axioms.append(encode_formula(constraint, {}))

    query_ref = None
    if symil.query is not None:
        query_ref = encode_formula(symil.query.prove, {})

    return {
        "Entity": entity_sort,
        "predicates": predicate_functions,
        "axioms": axioms,
        "query": query_ref,
    }
