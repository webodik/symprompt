from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from symprompt.config import DEFAULT_VSA_CONFIG
from symprompt.symil.model import Atom, SymIL


@dataclass
class VSACodebook:
    dim: int = DEFAULT_VSA_CONFIG.dimension
    _vectors: Dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, key: str) -> np.ndarray:
        if key not in self._vectors:
            rng = np.random.default_rng(abs(hash(key)) % (2**32))
            vec = rng.standard_normal(self.dim)
            vec /= np.linalg.norm(vec) + 1e-9
            self._vectors[key] = vec
        return self._vectors[key]


@dataclass
class VSAState:
    dim: int
    memory: np.ndarray


def _vsa_permute(vector: np.ndarray, steps: int) -> np.ndarray:
    """Simple permutation operator implemented as a cyclic roll."""
    if steps == 0:
        return vector
    return np.roll(vector, steps)


def _vsa_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Binding via element-wise multiplication."""
    return a * b


def _vsa_bundle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bundling via addition."""
    return a + b


def encode_atom_to_vsa(atom: Atom, codebook: VSACodebook) -> np.ndarray:
    """
    Encode a single atomic fact into a VSA vector by binding the
    predicate vector with permuted argument vectors and bundling
    them into a single representation.
    """
    vector = codebook.get(atom.pred)
    for index, argument in enumerate(atom.args):
        argument_vec = codebook.get(argument)
        argument_vec = _vsa_permute(argument_vec, index + 1)
        vector = _vsa_bind(vector, argument_vec)
    return vector


def encode_symil_to_vsa(symil: SymIL, codebook: VSACodebook | None = None) -> VSAState:
    """
    Encode SymIL facts into a VSA memory vector using compositional
    binding and bundling across all atomic facts.
    """
    if codebook is None:
        codebook = VSACodebook()

    memory = np.zeros(codebook.dim)

    for fact in symil.facts:
        if not isinstance(fact, Atom):
            continue
        fact_vector = encode_atom_to_vsa(fact, codebook)
        memory = _vsa_bundle(memory, fact_vector)

    return VSAState(dim=codebook.dim, memory=memory)
