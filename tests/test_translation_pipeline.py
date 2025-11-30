from __future__ import annotations

import json

from symprompt.translation.pipeline import TranslationPipeline
from symprompt.symil.model import Atom


class DummyLLM:
    """
    Deterministic LLM stub for testing TranslationPipeline without network calls.
    """

    def complete(self, prompt: str) -> str:
        if "ontology designer for a logic reasoning system" in prompt:
            return json.dumps(
                {
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
                }
            )

        if "translator from natural language into a JSON-based logical" in prompt:
            return json.dumps(
                {
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
                }
            )

        raise ValueError("Unexpected prompt in DummyLLM")


def test_translation_pipeline_mammals_example() -> None:
    llm = DummyLLM()
    pipeline = TranslationPipeline.from_llm_client(llm)

    text = "All mammals are animals. All cats are mammals. Therefore, all cats are animals."
    symil = pipeline.translate(text, level=1)

    predicate_names = {p.name for p in symil.ontology.predicates}
    assert {"mammal", "animal", "cat"}.issubset(predicate_names)
    assert symil.level == 1
    assert symil.rules, "Expected rules for mammals/cats example"
    assert symil.query is not None


class DummyLLMWithConstraints:
    """
    Deterministic LLM stub that also emits a constraints list, to verify
    that LogicalTranslator and the pipeline wire constraints into SymIL.
    """

    def complete(self, prompt: str) -> str:
        if "ontology designer for a logic reasoning system" in prompt:
            return json.dumps(
                {
                    "predicates": [
                        {
                            "name": "p",
                            "arity": 1,
                            "types": ["entity"],
                            "description": "X has property p",
                        }
                    ],
                    "functions": [],
                    "constants": [],
                }
            )

        if "translator from natural language into a JSON-based logical" in prompt:
            return json.dumps(
                {
                    "facts": [],
                    "rules": [],
                    "query": {"prove": {"pred": "p", "args": ["a"]}},
                    "constraints": [
                        {"pred": "p", "args": ["a"]},
                    ],
                }
            )

        raise ValueError("Unexpected prompt in DummyLLMWithConstraints")


def test_translation_pipeline_wires_constraints_into_symil() -> None:
    llm = DummyLLMWithConstraints()
    pipeline = TranslationPipeline.from_llm_client(llm)

    text = "Assume that p(a) is required."
    symil = pipeline.translate(text, level=2)

    assert symil.level == 2
    assert symil.query is not None
    assert symil.constraints, "Expected constraints to be populated"
    assert isinstance(symil.constraints[0], Atom)
