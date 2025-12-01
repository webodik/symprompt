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
You are a translator from natural language into a JSON-based logical intermediate language called SymIL.

Output ONLY valid JSON with this EXACT structure:
{
  "facts": [...],
  "rules": [...],
  "query": {...},
  "constraints": [...]
}

FORMULA SYNTAX (use EXACTLY these keys):
- Atom: {"pred": "name", "args": ["X", "Y"]}
- Implication: {"implies": [<premise>, <conclusion>]}
- Conjunction: {"and": [<formula>, <formula>, ...]}
- Disjunction: {"or": [<formula>, <formula>, ...]}
- Negation: {"not": <formula>}
- Universal: {"forall": "X", "type": "entity", "body": <formula>}
- Existential: {"exists": "X", "type": "entity", "body": <formula>}

CRITICAL - VARIABLES vs CONSTANTS:
- Variables: UPPERCASE like "X", "Y", "Z" - used in forall/exists quantifiers
- Constants: lowercase like "socrates", "earth", "alice" - named entities from ontology

RULE SYNTAX:
{"forall": "X", "type": "entity", "body": {"implies": [<premise>, <conclusion>]}}

QUERY SYNTAX:
{"prove": <formula>}

EXAMPLE 1 - "Socrates is human. All humans are mortal. Is Socrates mortal?":
{
  "facts": [{"pred": "human", "args": ["socrates"]}],
  "rules": [
    {"forall": "X", "type": "entity", "body": {"implies": [{"pred": "human", "args": ["X"]}, {"pred": "mortal", "args": ["X"]}]}}
  ],
  "query": {"prove": {"pred": "mortal", "args": ["socrates"]}},
  "constraints": []
}

EXAMPLE 2 - "All mammals are animals. All cats are mammals. Therefore, all cats are animals.":
{
  "facts": [],
  "rules": [
    {"forall": "X", "type": "entity", "body": {"implies": [{"pred": "mammal", "args": ["X"]}, {"pred": "animal", "args": ["X"]}]}},
    {"forall": "X", "type": "entity", "body": {"implies": [{"pred": "cat", "args": ["X"]}, {"pred": "mammal", "args": ["X"]}]}}
  ],
  "query": {"prove": {"forall": "X", "type": "entity", "body": {"implies": [{"pred": "cat", "args": ["X"]}, {"pred": "animal", "args": ["X"]}]}}},
  "constraints": []
}

Rules:
- Use ONLY predicates from the provided ontology
- Use ONLY constants from the provided ontology (lowercase!)
- Variables (X, Y, Z): UPPERCASE, only inside forall/exists
- Constants (socrates, earth): lowercase, for specific named entities
- For Level 0: prefer facts + atomic query, avoid rules
- For Level 1: use simple Horn clauses (forall with implies)
- For Level 2: full SymIL with nested quantifiers
- Output ONLY the JSON object, no explanation or markdown
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
                "constants": ontology.constants if ontology.constants else [],
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
        if not isinstance(data, dict):
            raise ValueError(f"Formula must be a dict, got {type(data)}: {data}")

        # Atom: {"pred": "name", "args": ["X", "Y"]}
        if "pred" in data:
            return Atom(pred=data["pred"], args=list(data.get("args", [])))

        # Implication: {"implies": [<premise>, <conclusion>]}
        if "implies" in data:
            impl = data["implies"]
            if not isinstance(impl, list) or len(impl) != 2:
                raise ValueError(f"implies must be a list of 2 formulas: {data}")
            return Implies(
                premise=self._formula_from_json(impl[0]),
                conclusion=self._formula_from_json(impl[1]),
            )

        # Conjunction: {"and": [<formula>, ...]}
        if "and" in data:
            return And(formulas=[self._formula_from_json(f) for f in data["and"]])

        # Disjunction: {"or": [<formula>, ...]}
        if "or" in data:
            return Or(formulas=[self._formula_from_json(f) for f in data["or"]])

        # Negation: {"not": <formula>}
        if "not" in data:
            return Not(formula=self._formula_from_json(data["not"]))

        # Universal: {"forall": "X", "type": "entity", "body": <formula>}
        if "forall" in data and "body" in data:
            return ForAll(
                var=data["forall"],
                type=data.get("type", "entity"),
                body=self._formula_from_json(data["body"]),
            )

        # Existential: {"exists": "X", "type": "entity", "body": <formula>}
        if "exists" in data and "body" in data:
            return Exists(
                var=data["exists"],
                type=data.get("type", "entity"),
                body=self._formula_from_json(data["body"]),
            )

        raise ValueError(f"Unsupported formula JSON: {data}")

    def _rule_from_json(self, data: Dict[str, Any]) -> Rule:
        # Rule: {"forall": "X", "type": "entity", "body": <formula>}
        if "forall" not in data or "body" not in data:
            raise ValueError(f"Rule must have 'forall' and 'body' keys: {data}")
        return Rule(
            forall=data["forall"],
            type=data.get("type", "entity"),
            body=self._formula_from_json(data["body"]),
        )

    def _query_from_json(self, data: Dict[str, Any] | None) -> Query | None:
        if data is None:
            return None
        # Query: {"prove": <formula>}
        if "prove" not in data:
            raise ValueError(f"Query must have 'prove' key: {data}")
        return Query(prove=self._formula_from_json(data["prove"]))

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
