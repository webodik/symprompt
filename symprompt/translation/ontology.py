from __future__ import annotations

import json
from typing import List

from symprompt.symil.model import Ontology, Predicate


ONTOLOGY_SYSTEM_PROMPT = """
You are an ontology designer for a logic reasoning system.
Read the text and extract predicates and named constants.

Output ONLY a JSON object:
{
  "predicates": [
    {"name": "mammal", "arity": 1, "types": ["entity"], "description": "X is a mammal"}
  ],
  "constants": ["socrates", "earth", "alice"]
}

EXAMPLE - "Socrates is a human. All humans are mortal.":
{
  "predicates": [
    {"name": "human", "arity": 1, "types": ["entity"], "description": "X is human"},
    {"name": "mortal", "arity": 1, "types": ["entity"], "description": "X is mortal"}
  ],
  "constants": ["socrates"]
}

CRITICAL RULES:
- Predicate names: lowercase, no spaces (e.g., "can_fly", "is_mammal")
- Constants: lowercase named entities from the text (e.g., "socrates", "earth", "alice")
  - "Socrates" -> "socrates"
  - "Earth" -> "earth"
  - "James" -> "james"
- Arity is typically 1; use 2 for relations (e.g., "taller_than(X, Y)")
- Do NOT include "functions" key - only "predicates" and "constants"
"""


class OntologyExtractor:
    """LLM-based ontology extraction for SymIL."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    def extract(self, text: str) -> Ontology:
        prompt = ONTOLOGY_SYSTEM_PROMPT + "\nText:\n" + text
        raw = self.llm_client.complete(prompt)
        data = json.loads(raw)

        predicates_json = data.get("predicates", [])
        predicates: List[Predicate] = [
            Predicate(
                name=predicate_json["name"],
                arity=predicate_json.get("arity", 1),
                types=predicate_json.get("types", ["entity"]),
                description=predicate_json.get("description"),
            )
            for predicate_json in predicates_json
        ]
        # Extract constants (lowercase named entities)
        constants: List[str] = [
            str(c).lower() for c in data.get("constants", [])
        ]
        return Ontology(predicates=predicates, constants=constants)

