from __future__ import annotations

from typing import Dict

from symprompt.reasoning.asp_compiler import _atom_to_asp, symil_to_asp
from symprompt.symil.model import Atom, SymIL

try:
    import clingo  # type: ignore[import]
except Exception:  # pragma: no cover
    clingo = None  # type: ignore[assignment]


def run_asp(symil: SymIL) -> Dict[str, str]:
    """
    Execute SymIL via an ASP backend.

    We currently support simple yes/no queries where the SymIL query
    is an Atom. To classify VALID vs NOT_VALID:

    - VALID: base program ∪ {:- not Q.} has a stable model, and
      base program ∪ {:- Q.} has no stable model.
    - NOT_VALID: base program ∪ {:- Q.} has a stable model, and
      base program ∪ {:- not Q.} has no stable model.
    - UNKNOWN: any other combination (including non-atomic queries).
    """
    if clingo is None:
        return {"status": "UNKNOWN"}

    base_program = symil_to_asp(symil)
    if not symil.query or not isinstance(symil.query.prove, Atom):
        return {"status": "UNKNOWN"}

    query_atom = symil.query.prove
    query_str = _atom_to_asp(query_atom)

    def has_model(program: str) -> bool:
        ctl = clingo.Control()
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        result = ctl.solve()
        return result.satisfiable

    program_valid = f"{base_program}\n:- not {query_str}.\n"
    program_notvalid = f"{base_program}\n:- {query_str}.\n"

    sat_valid = has_model(program_valid)
    sat_notvalid = has_model(program_notvalid)

    if sat_valid and not sat_notvalid:
        return {"status": "VALID"}
    if sat_notvalid and not sat_valid:
        return {"status": "NOT_VALID"}
    return {"status": "UNKNOWN"}
