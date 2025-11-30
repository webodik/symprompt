from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LLMConfig:
    """Configuration for the primary LLM used by SymPrompt."""

    model_name: str = "openrouter/x-ai/grok-4.1-fast:free"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout_seconds: int = 60
    retries: int = 2
    retry_delay_seconds: float = 1.0
    random_seed: Optional[int] = None


@dataclass
class VSAConfig:
    """Configuration for the Vector Symbolic Architecture backend."""

    dimension: int = 1024
    valid_threshold: float = 0.9
    invalid_threshold: float = 0.3
    tiebreaker_threshold: float = 0.8


@dataclass
class SolverConfig:
    """Configuration for reasoning solvers."""

    z3_timeout_ms: int = 1000


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline."""

    tier1_latency_target_ms: float = 50.0
    tier2_latency_target_ms: float = 500.0
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "syllogism": 0.2,
        "math": 0.3,
        "planning": 0.3,
        "legal": 0.2,
    })


@dataclass
class RefinementConfig:
    """Configuration for verification and refinement."""

    max_attempts: int = 2


@dataclass
class FeatureConfig:
    """Configuration for feature extraction in routing."""

    complexity_length_divisor: float = 40.0
    complexity_clause_weight: float = 0.2


DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_VSA_CONFIG = VSAConfig()
DEFAULT_SOLVER_CONFIG = SolverConfig()
DEFAULT_EVALUATION_CONFIG = EvaluationConfig()
DEFAULT_REFINEMENT_CONFIG = RefinementConfig()
DEFAULT_FEATURE_CONFIG = FeatureConfig()
