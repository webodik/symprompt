from __future__ import annotations

import json
from typing import Any, Dict, List

from symprompt.symil.model import (
    And,
    Atom,
    Exists,
    ForAll,
    Formula,
    Implies,
    Not,
    Or,
    Query,
    Rule,
    SymIL,
)


LOGICAL_SYSTEM_PROMPT = """
You are a translator from natural language into a JSON-based logical
intermediate language called SymIL.

Given:
- A short piece of text describing logical relationships.
- An ontology listing predicates.

You must output ONLY a JSON object with the structure:
{
  "facts": [ ... ],
  "rules": [ ... ],
  "query": { ... },
  "constraints": [ ... ]
}

Rules:
- Use only predicates defined in the ontology.
- Use variable names like "X", "Y" (type "entity").
- Use "implies" for conditional statements ("all ... are ...").
- Use "forall" for universal statements.
 - For Level 0, prefer only facts + an atomic query and avoid rules.
 - For Level 1, restrict rules to simple Horn clauses.
 - For Level 2, you may emit full SymIL (nested quantifiers, constraints).
"""


class LogicalTranslator:
    """LLM-based logical translation into SymIL."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    def translate(self, text: str, ontology, target_level: int, hints: list[str]) -> SymIL:
        ontology_json = json.dumps(
            {
                "predicates": [
                    {
                        "name": predicate.name,
                        "arity": predicate.arity,
                        "types": predicate.types,
                        "description": predicate.description or "",
                    }
                    for predicate in ontology.predicates
                ],
                "functions": [],
                "constants": [],
            }
        )
        prompt = LOGICAL_SYSTEM_PROMPT
        if hints:
            prompt += "\nDomain hints:\n"
            for hint in hints:
                prompt += f"- {hint}\n"
        prompt += "\nOntology:\n" + ontology_json + "\nText:\n" + text
        raw = self.llm_client.complete(prompt)
        data = json.loads(raw)

        facts = [self._formula_from_json(fact_json) for fact_json in data.get("facts", [])]
        rules = [self._rule_from_json(rule_json) for rule_json in data.get("rules", [])]
        query = self._query_from_json(data.get("query"))
        constraints = [
            self._formula_from_json(constraint_json)
            for constraint_json in data.get("constraints", [])
        ]

        if target_level == 0:
            rules = []
        elif target_level == 1:
            rules = [rule for rule in rules if not self._contains_exists(rule.body)]

        return SymIL(
            ontology=ontology,
            facts=facts,
            rules=rules,
            query=query,
            constraints=constraints,
            level=target_level,
        )

    def _formula_from_json(self, data: Dict[str, Any]) -> Formula:
        if "pred" in data:
            return Atom(pred=data["pred"], args=list(data.get("args", [])))
        if "implies" in data:
            premise_json, conclusion_json = data["implies"]
            return Implies(
                premise=self._formula_from_json(premise_json),
                conclusion=self._formula_from_json(conclusion_json),
            )
        if "and" in data:
            return And(
                formulas=[self._formula_from_json(item) for item in data["and"]]
            )
        if "or" in data:
            return Or(
                formulas=[self._formula_from_json(item) for item in data["or"]]
            )
        if "not" in data:
            return Not(formula=self._formula_from_json(data["not"]))
        if "forall" in data and "body" in data:
            return ForAll(
                var=data["forall"],
                type=data.get("type", "entity"),
                body=self._formula_from_json(data["body"]),
            )
        if "exists" in data and "body" in data:
            return Exists(
                var=data["exists"],
                type=data.get("type", "entity"),
                body=self._formula_from_json(data["body"]),
            )
        raise ValueError(f"Unsupported formula JSON: {data}")

    def _rule_from_json(self, data: Dict[str, Any]) -> Rule:
        if "forall" in data and "body" in data:
            body_formula = self._formula_from_json(data["body"])
            return Rule(
                forall=data["forall"],
                type=data.get("type", "entity"),
                body=body_formula,
            )
        # Fallback: treat as Horn-style rule encoded directly as formula.
        body_formula = self._formula_from_json(data)
        return Rule(forall="X", type="entity", body=body_formula)

    def _query_from_json(self, data: Dict[str, Any] | None) -> Query | None:
        if data is None:
            return None
        if "prove" in data:
            formula = self._formula_from_json(data["prove"])
            return Query(prove=formula)
        # Allow direct formula as query.
        formula = self._formula_from_json(data)
        return Query(prove=formula)

    def _contains_exists(self, formula: Formula) -> bool:
        if isinstance(formula, Exists):
            return True
        if isinstance(formula, (And, Or)):
            return any(self._contains_exists(sub) for sub in formula.formulas)
        if isinstance(formula, Implies):
            return self._contains_exists(formula.premise) or self._contains_exists(
                formula.conclusion
            )
        if isinstance(formula, ForAll):
            return self._contains_exists(formula.body)
        if isinstance(formula, Not):
            return self._contains_exists(formula.formula)
        return False
