from __future__ import annotations

from typing import Dict

import numpy as np

from symprompt.config import DEFAULT_VSA_CONFIG
from symprompt.reasoning.vsa_encoder import VSACodebook, encode_atom_to_vsa, encode_symil_to_vsa
from symprompt.symil.model import Atom, SymIL


def run_vsa(symil: SymIL, dim: int | None = None) -> Dict[str, object]:
    """
    Execute SymIL in a vector-symbolic manner by encoding it into a high-
    dimensional memory vector. For simple Level 0 programs with an atomic
    query, this acts as a soft similarity-based solver; otherwise it returns
    UNKNOWN plus the encoded state.
    """
    if dim is None:
        dim = DEFAULT_VSA_CONFIG.dimension

    codebook = VSACodebook(dim=dim)
    state = encode_symil_to_vsa(symil, codebook)

    if symil.level == 0 and symil.query is not None and isinstance(symil.query.prove, Atom):
        query_atom = symil.query.prove
        query_vector = encode_atom_to_vsa(query_atom, codebook)
        memory_norm = float(np.linalg.norm(state.memory) + 1e-9)
        query_norm = float(np.linalg.norm(query_vector) + 1e-9)
        similarity = float(np.dot(state.memory, query_vector) / (memory_norm * query_norm))

        if similarity > DEFAULT_VSA_CONFIG.valid_threshold:
            status = "VALID"
        elif similarity < DEFAULT_VSA_CONFIG.invalid_threshold:
            status = "NOT_VALID"
        else:
            status = "UNKNOWN"
        return {"status": status, "state": state, "similarity": similarity}

    return {"status": "UNKNOWN", "state": state}
