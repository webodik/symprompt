from __future__ import annotations

from symprompt.integration.router_adapter import route_and_solve


class DummyLLM:
    def complete(self, prompt: str) -> str:
        # Simple deterministic ontology and SymIL JSON for mammals/cats example.
        if "ontology designer for a logic reasoning system" in prompt:
            return (
                '{"predicates": ['
                '{"name": "mammal", "arity": 1, "types": ["entity"]},'
                '{"name": "animal", "arity": 1, "types": ["entity"]},'
                '{"name": "cat", "arity": 1, "types": ["entity"]}'
                '], "functions": [], "constants": []}'
            )
        if "translator from natural language into a JSON-based logical" in prompt:
            return (
                '{'
                '"facts": [],'
                '"rules": ['
                '{'
                '"forall": "X", "type": "entity",'
                '"body": {"implies": ['
                '{"pred": "mammal", "args": ["X"]},'
                '{"pred": "animal", "args": ["X"]}'
                ']}'
                '},'
                '{'
                '"forall": "X", "type": "entity",'
                '"body": {"implies": ['
                '{"pred": "cat", "args": ["X"]},'
                '{"pred": "mammal", "args": ["X"]}'
                ']}'
                '}'
                '],'
                '"query": {"prove": {'
                '"forall": "X", "type": "entity",'
                '"body": {"implies": ['
                '{"pred": "cat", "args": ["X"]},'
                '{"pred": "animal", "args": ["X"]}'
                ']}}}'
                '}'
            )
        raise ValueError("Unexpected prompt for DummyLLM")


def test_route_and_solve_returns_routing_and_status() -> None:
    llm = DummyLLM()
    text = "All mammals are animals. All cats are mammals. Therefore, all cats are animals."
    result = route_and_solve(text, llm)

    assert result.routing.tier in {1, 2}
    assert result.routing.profile_name == "syllogism"
    assert result.solver_result.get("status") in {"VALID", "UNKNOWN"}

