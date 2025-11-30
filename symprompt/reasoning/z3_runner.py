from __future__ import annotations

from typing import Dict

import z3

from symprompt.config import DEFAULT_SOLVER_CONFIG
from symprompt.reasoning.fol_compiler import symil_to_z3
from symprompt.symil.model import SymIL


def solve_symil(symil: SymIL, timeout_ms: int | None = None) -> Dict[str, str]:
    """
    Run Z3 on the compiled SymIL program and classify the query as VALID,
    NOT_VALID, or UNKNOWN.
    """
    if timeout_ms is None:
        timeout_ms = DEFAULT_SOLVER_CONFIG.z3_timeout_ms

    context = symil_to_z3(symil)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    for axiom in context["axioms"]:
        solver.add(axiom)

    query_ref = context["query"]
    if query_ref is not None:
        solver.push()
        solver.add(z3.Not(query_ref))
        sat_result = solver.check()
        solver.pop()
        if sat_result == z3.unsat:
            status = "VALID"
        elif sat_result == z3.sat:
            status = "NOT_VALID"
        else:
            status = "UNKNOWN"
    else:
        sat_result = solver.check()
        status = str(sat_result)

    return {"status": status}

