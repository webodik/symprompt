from __future__ import annotations

import json
from typing import List

from symprompt.symil.model import Ontology, Predicate


ONTOLOGY_SYSTEM_PROMPT = """
You are an ontology designer for a logic reasoning system.
Read the text and extract a minimal set of logical predicates
and constants that will be used to formalize the text.

Output ONLY a JSON object:
{
  "predicates": [
    {"name": "mammal", "arity": 1, "types": ["entity"], "description": "X is a mammal"}
  ],
  "functions": [],
  "constants": []
}

Rules:
- Use lowercase predicate names without spaces.
- Assume all predicates have arity 1 and type "entity" for now.
- Do NOT include comments or extra keys.
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
        return Ontology(predicates=predicates)

