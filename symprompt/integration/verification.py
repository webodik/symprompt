from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from symprompt.config import DEFAULT_REFINEMENT_CONFIG
from symprompt.integration.router_adapter import route_and_solve


@dataclass
class VerificationResult:
    status: str
    route_result: Any


@dataclass
class RefinementResult:
    status: str
    answer: str
    attempts: int
    route_result: Any


def verify_answer(question: str, answer: str, llm_client) -> VerificationResult:
    """
    Post-processing verification (Mode 2, skeleton).

    Combines the original question and LLM answer into a single text
    and routes it through SymPrompt (router + translation + solver).
    The solver status is used as a coarse proxy for whether the answer
    is logically consistent with the question.
    """
    combined = f"Question: {question}\nAnswer: {answer}"
    route_result = route_and_solve(combined, llm_client)
    status = str(route_result.solver_result.get("status", "UNKNOWN"))
    return VerificationResult(status=status, route_result=route_result)


def verify_and_refine_answer(
    question: str,
    initial_answer: str,
    llm_client,
    max_attempts: int | None = None,
) -> RefinementResult:
    """
    Hybrid Mode 3 refinement loop.

    Starts from an initial answer, verifies it symbolically, and if the
    result is not VALID, asks the LLM to refine the answer using a short
    feedback message. Performs up to max_attempts verification/refinement
    cycles and returns the final status and answer.
    """
    if max_attempts is None:
        max_attempts = DEFAULT_REFINEMENT_CONFIG.max_attempts

    answer = initial_answer
    last_route_result: Any | None = None
    attempts = 0

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        verification = verify_answer(question, answer, llm_client)
        last_route_result = verification.route_result
        status = verification.status
        if status == "VALID":
            return RefinementResult(
                status=status,
                answer=answer,
                attempts=attempts,
                route_result=last_route_result,
            )

        feedback_prompt = (
            f"Question: {question}\n"
            f"Previous answer: {answer}\n\n"
            f"A separate logical verification engine classified the previous answer as '{status}'. "
            "Please provide a new answer that is logically consistent with the question."
        )
        answer = llm_client.complete(feedback_prompt)

    final_status = "UNKNOWN"
    if last_route_result is not None:
        final_status = str(last_route_result.solver_result.get("status", "UNKNOWN"))

    return RefinementResult(
        status=final_status,
        answer=answer,
        attempts=attempts,
        route_result=last_route_result,
    )
