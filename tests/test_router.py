from __future__ import annotations

from symprompt.router.smart_router import SmartRouter


def test_router_math_prompt_uses_math_profile() -> None:
    router = SmartRouter()
    decision = router.route("Is 2 + 2 equal to 4?", context=None)
    assert decision.profile_name == "math"
    assert decision.tier == 1


def test_router_syllogism_prompt_uses_syllogism_profile() -> None:
    router = SmartRouter()
    text = "All mammals are animals. All cats are mammals. Therefore, all cats are animals."
    decision = router.route(text, context=None)
    assert decision.profile_name == "syllogism"
    assert decision.tier in {1, 2}


def test_router_planning_prompt_prefers_planning_profile() -> None:
    router = SmartRouter()
    text = "You must plan actions to achieve the goal given several constraints."
    decision = router.route(text, context=None)
    assert decision.profile_name == "planning"
    assert decision.tier == 2 or decision.symil_level >= 2

