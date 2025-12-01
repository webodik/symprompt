from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from symprompt.config import DEFAULT_VSA_CONFIG
from symprompt.reasoning.clingo_runner import run_asp
from symprompt.reasoning.scallop_runner import run_scallop
from symprompt.reasoning.vsa_runner import run_vsa
from symprompt.reasoning.z3_runner import solve_symil
from symprompt.symil.model import SymIL


@dataclass
class PortfolioDecision:
    tier: int
    profile_name: str
    preferred_solver: str


def run_portfolio(symil: SymIL, decision: PortfolioDecision | None = None) -> Dict[str, str]:
    """
    Minimal portfolio runner.

    For Tier 1, uses the preferred solver hint when possible and falls back
    to Z3. For Tier 2, runs a small multi-backend portfolio and aggregates
    the results, falling back to Z3 when consensus is unclear.
    """
    if decision is None:
        return solve_symil(symil)

    solver = decision.preferred_solver.lower()

    # Tier 1: single-backend path based on profile, fallback to Z3 on UNKNOWN.
    if decision.tier == 1:
        result = None
        if solver in {"asp", "clingo"}:
            result = run_asp(symil)
        elif solver in {"scallop", "datalog"}:
            result = run_scallop(symil)
        elif solver in {"vsa", "vector"}:
            vsa_result = run_vsa(symil)
            result = {"status": str(vsa_result.get("status", "UNKNOWN"))}

        if result is not None and result.get("status") != "UNKNOWN":
            return result
        # Fallback to Z3 when preferred solver returns UNKNOWN
        return solve_symil(symil)

    # Tier 2: simple multi-backend portfolio with consensus-style aggregation.
    z3_result = solve_symil(symil)
    asp_result = run_asp(symil)
    scallop_result = run_scallop(symil)
    vsa_result = run_vsa(symil)

    statuses = [
        str(z3_result.get("status")),
        str(asp_result.get("status")),
        str(scallop_result.get("status")),
        str(vsa_result.get("status")),
    ]

    if statuses.count("VALID") >= 2:
        return {"status": "VALID"}
    if statuses.count("NOT_VALID") >= 2:
        return {"status": "NOT_VALID"}

    similarity = vsa_result.get("similarity")
    if isinstance(similarity, (int, float)):
        similarity_value = float(similarity)
        if similarity_value > DEFAULT_VSA_CONFIG.tiebreaker_threshold:
            return {"status": "VALID"}
        if similarity_value < DEFAULT_VSA_CONFIG.invalid_threshold:
            return {"status": "NOT_VALID"}

    return z3_result
