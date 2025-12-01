from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict

from symprompt.reasoning.portfolio import PortfolioDecision
from symprompt.router.smart_router import RoutingDecision
from symprompt.symil.model import SymIL
from symprompt.symil.profiles import get_profile
from symprompt.symil.validator import SymILValidator
from symprompt.translation.pipeline import TranslationPipeline


SolveFn = Callable[..., Dict[str, Any]]


@dataclass
class EscalationResult:
    result: Dict[str, Any]
    used_level: int
    symil: SymIL
    solver_latency_ms: float


def translate_and_solve_with_escalation(
    pipeline: TranslationPipeline,
    validator: SymILValidator,
    decision: RoutingDecision,
    text: str,
    solve_fn: SolveFn,
    max_level: int = 2,
) -> EscalationResult:
    """
    Shared helper for progressive SymIL level escalation.

    Starting from decision.symil_level, translate and call the solver.
    If the solver returns VALID/NOT_VALID, stop. If it returns UNKNOWN
    and higher levels are available, escalate to the next level.

    Returns an EscalationResult with the final solver result, the level
    used, the last SymIL program, and cumulative solver latency in ms.
    """
    last_result: Dict[str, Any] | None = None
    last_symil: SymIL | None = None
    used_level = decision.symil_level
    total_solver_latency_ms = 0.0

    profile = get_profile(decision.profile_name)
    base_hints = profile.translation_hints

    for level in range(decision.symil_level, max_level + 1):
        hints = list(base_hints)

        # Up to one refinement attempt per level based on solver feedback.
        for attempt in range(2):
            symil = pipeline.translate(text, level=level, hints=hints)
            symil = validator.validate(symil)

            portfolio_decision = PortfolioDecision(
                tier=decision.tier,
                profile_name=decision.profile_name,
                preferred_solver=decision.preferred_solver,
            )
            solve_start = perf_counter()
            result = solve_fn(symil=symil, decision=portfolio_decision)
            total_solver_latency_ms += (perf_counter() - solve_start) * 1000.0

            last_result = result
            last_symil = symil
            used_level = level

            status = str(result.get("status"))
            if status in {"VALID", "NOT_VALID"}:
                break

            # Solver returned UNKNOWN on first attempt at this level; refine hints once.
            if attempt == 0:
                hints = hints + [
                    f"Refinement hint: previous solve status was {status} at SymIL level {level}. "
                    "Try making implicit rules or constraints explicit.",
                ]

        if last_result is not None and str(last_result.get("status")) in {"VALID", "NOT_VALID"}:
            break

    assert last_result is not None and last_symil is not None
    return EscalationResult(
        result=last_result,
        used_level=used_level,
        symil=last_symil,
        solver_latency_ms=total_solver_latency_ms,
    )
