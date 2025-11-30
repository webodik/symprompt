from __future__ import annotations

from typing import Dict, List


def mammals_symil_l1_json() -> Dict[str, object]:
    """
    Example SymIL JSON for:
    "All mammals are animals. All cats are mammals. Therefore, all cats are animals."
    """
    return {
        "level": 1,
        "ontology": {
            "predicates": [
                {
                    "name": "mammal",
                    "arity": 1,
                    "types": ["entity"],
                    "description": "X is a mammal",
                },
                {
                    "name": "animal",
                    "arity": 1,
                    "types": ["entity"],
                    "description": "X is an animal",
                },
                {
                    "name": "cat",
                    "arity": 1,
                    "types": ["entity"],
                    "description": "X is a cat",
                },
            ],
            "functions": [],
            "constants": [],
        },
        "facts": [],
        "rules": [
            {
                "forall": "X",
                "type": "entity",
                "body": {
                    "implies": [
                        {"pred": "mammal", "args": ["X"]},
                        {"pred": "animal", "args": ["X"]},
                    ]
                },
            },
            {
                "forall": "X",
                "type": "entity",
                "body": {
                    "implies": [
                        {"pred": "cat", "args": ["X"]},
                        {"pred": "mammal", "args": ["X"]},
                    ]
                },
            },
        ],
        "query": {
            "prove": {
                "forall": "X",
                "type": "entity",
                "body": {
                    "implies": [
                        {"pred": "cat", "args": ["X"]},
                        {"pred": "animal", "args": ["X"]},
                    ]
                },
            }
        },
        "constraints": [],
    }

