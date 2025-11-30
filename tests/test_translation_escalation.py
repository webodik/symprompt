from __future__ import annotations

from symprompt.translation.pipeline import TranslationPipeline, translate_with_escalation


class DummyLLM:
    def complete(self, prompt: str) -> str:
        import json

        if "ontology designer for a logic reasoning system" in prompt:
            return json.dumps(
                {
                    "predicates": [
                        {"name": "p", "arity": 1, "types": ["entity"]},
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
                    "query": {"prove": {"pred": "p", "args": ["X"]}},
                }
            )
        raise ValueError("Unexpected prompt for DummyLLM in escalation test")


def test_translate_with_escalation_uses_lowest_level() -> None:
    llm = DummyLLM()
    pipeline = TranslationPipeline.from_llm_client(llm)

    symil = translate_with_escalation(pipeline, "Test prompt.", min_level=0, max_level=2)

    assert symil.level == 0
    assert symil.query is not None

