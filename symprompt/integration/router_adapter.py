from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from symprompt.integration.escalation import EscalationResult, translate_and_solve_with_escalation
from symprompt.reasoning.portfolio import PortfolioDecision, run_portfolio
from symprompt.router.smart_router import RoutingDecision, SmartRouter
from symprompt.symil.profiles import get_profile
from symprompt.symil.validator import SymILValidator
from symprompt.translation.pipeline import TranslationPipeline


@dataclass
class RouteResult:
    routing: RoutingDecision
    symil: Any
    solver_result: Dict[str, Any]


def _route_with_escalation(
    prompt: str,
    llm_client,
    max_level: int = 2,
) -> RouteResult:
    router = SmartRouter()
    pipeline = TranslationPipeline.from_llm_client(llm_client)
    validator = SymILValidator()

    routing_decision: RoutingDecision = router.route(prompt, context=None)
    profile = get_profile(routing_decision.profile_name)

    # Tier 0: BYPASS – pure LLM path without symbolic reasoning.
    if routing_decision.tier == 0:
        answer = llm_client.complete(prompt)
        return RouteResult(
            routing=routing_decision,
            symil=None,
            solver_result={"status": "BYPASS", "answer": answer},
        )

    escalation: EscalationResult = translate_and_solve_with_escalation(
        pipeline=pipeline,
        validator=validator,
        decision=routing_decision,
        text=prompt,
        solve_fn=run_portfolio,
        max_level=max_level,
    )

    return RouteResult(
        routing=routing_decision,
        symil=escalation.symil,
        solver_result=escalation.result,
    )


def route_and_solve(prompt: str, llm_client) -> RouteResult:
    """
    Route a prompt through the SmartRouter, translate to SymIL with
    potential level escalation, and run the reasoning portfolio.
    """
    return _route_with_escalation(prompt, llm_client)
