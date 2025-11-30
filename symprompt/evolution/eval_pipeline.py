from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List

import importlib.util
import json
import sys

from symprompt.config import DEFAULT_EVALUATION_CONFIG
from symprompt.integration.escalation import EscalationResult, translate_and_solve_with_escalation
from symprompt.router.smart_router import RoutingDecision, SmartRouter
from symprompt.translation.pipeline import TranslationPipeline
from symprompt.reasoning.portfolio import run_portfolio
from symprompt.symil.validator import SymILValidator
from symprompt.llm.sync_client import build_default_sync_client


@dataclass
class EvalResult:
    tier1_accuracy: float
    tier1_coverage: float
    tier1_p95_latency_ms: float
    tier2_accuracy: float
    tier2_coverage: float
    tier2_p95_latency_ms: float
    syntactic_validity: float
    routing_score: float


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(0.95 * (len(ordered) - 1))
    return ordered[index]


def _translate_and_solve_with_escalation(
    pipeline: TranslationPipeline,
    validator: SymILValidator,
    decision: RoutingDecision,
    text: str,
    solve_fn: Callable[..., Dict[str, str]],
    max_level: int = 2,
) -> tuple[Dict[str, str], int]:
    """
    Wrapper around the shared translate_and_solve_with_escalation helper
    that preserves the original (result, used_level) return shape.
    """
    escalation: EscalationResult = translate_and_solve_with_escalation(
        pipeline=pipeline,
        validator=validator,
        decision=decision,
        text=text,
        solve_fn=solve_fn,
        max_level=max_level,
    )
    return escalation.result, escalation.used_level


def evaluate_system(
    router: SmartRouter,
    pipeline: TranslationPipeline,
    benchmarks: List[Dict[str, object]],
    solve_fn: Callable[..., Dict[str, str]] = run_portfolio,
) -> EvalResult:
    validator = SymILValidator()

    tier1_latencies: List[float] = []
    tier2_latencies: List[float] = []
    tier1_total = tier1_correct = 0
    tier2_total = tier2_correct = 0
    syntactic_ok = syntactic_total = 0
    routing_hits = routing_total = 0

    for item in benchmarks:
        text = str(item["nl"])
        expected = str(item["expected_result"])

        decision: RoutingDecision = router.route(text, context=None)
        routing_total += 1

        start = perf_counter()
        result, used_level = _translate_and_solve_with_escalation(
            pipeline,
            validator,
            decision,
            text,
            solve_fn,
            max_level=2,
        )
        elapsed_ms = (perf_counter() - start) * 1000.0

        syntactic_total += 1
        syntactic_ok += 1

        correct = str(result.get("status")) == expected

        if decision.tier == 1:
            tier1_total += 1
            tier1_latencies.append(elapsed_ms)
            if correct:
                tier1_correct += 1
        else:
            tier2_total += 1
            tier2_latencies.append(elapsed_ms)
            if correct:
                tier2_correct += 1

        ideal_tier = int(item.get("ideal_tier", decision.tier))
        if correct and decision.tier == ideal_tier:
            routing_hits += 1

    benchmarks_count = len(benchmarks) if benchmarks else 1

    return EvalResult(
        tier1_accuracy=tier1_correct / tier1_total if tier1_total else 0.0,
        tier1_coverage=tier1_total / benchmarks_count,
        tier1_p95_latency_ms=_p95(tier1_latencies),
        tier2_accuracy=tier2_correct / tier2_total if tier2_total else 0.0,
        tier2_coverage=tier2_total / benchmarks_count,
        tier2_p95_latency_ms=_p95(tier2_latencies),
        syntactic_validity=syntactic_ok / syntactic_total if syntactic_total else 0.0,
        routing_score=routing_hits / routing_total if routing_total else 0.0,
    )


def evaluate(program_path: str) -> Dict[str, float]:
    """
    OpenEvolve evaluator entrypoint.

    Given a path to a candidate translation pipeline module, this:
      - Loads the candidate TranslationPipeline class.
      - Evaluates it on tiny_folio.json using SmartRouter + portfolio.
      - Returns a metrics dict including combined_score and features.
    """
    root = Path(__file__).resolve().parents[2]

    benchmark_files = [
        "tiny_folio.json",
        "v2_syllogism.json",
        "v2_math.json",
        "v2_planning.json",
        "v2_legal.json",
    ]

    benchmarks: list[Dict[str, object]] = []
    for name in benchmark_files:
        path = root / "benchmarks" / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            benchmarks.extend(data)

    if not benchmarks:
        return {"combined_score": 0.0}

    # Dynamically load candidate pipeline module from program_path
    try:
        spec = importlib.util.spec_from_file_location("candidate_pipeline", program_path)
        if spec is None or spec.loader is None:
            return {"combined_score": 0.0}
        module = importlib.util.module_from_spec(spec)
        sys.modules["candidate_pipeline"] = module
        spec.loader.exec_module(module)
    except Exception:
        return {"combined_score": 0.0}

    pipeline_cls = getattr(module, "TranslationPipeline", None)
    if pipeline_cls is None:
        return {"combined_score": 0.0}

    router = SmartRouter()
    llm_client = build_default_sync_client()
    pipeline: TranslationPipeline = pipeline_cls.from_llm_client(llm_client)

    # Global metrics
    try:
        global_result = evaluate_system(router, pipeline, benchmarks)
    except Exception:
        code_len = float(len(Path(program_path).read_text(encoding="utf-8")))
        return {"combined_score": 0.0, "complexity": code_len}

    # Per-domain metrics
    domains: Dict[str, List[Dict[str, object]]] = {}
    for item in benchmarks:
        domain = str(item.get("domain", "generic"))
        domains.setdefault(domain, []).append(item)

    domain_metrics: Dict[str, float] = {}
    for domain, items in domains.items():
        try:
            domain_result = evaluate_system(router, pipeline, items)
        except Exception:
            continue
        domain_accuracy = max(domain_result.tier1_accuracy, domain_result.tier2_accuracy)
        domain_metrics[f"{domain}_accuracy"] = domain_accuracy
        domain_metrics[f"{domain}_coverage"] = (
            domain_result.tier1_coverage + domain_result.tier2_coverage
        )

    # Global and tier-weighted accuracy.
    global_accuracy = max(global_result.tier1_accuracy, global_result.tier2_accuracy)
    tier1_acc_cov = global_result.tier1_accuracy * global_result.tier1_coverage
    tier2_acc_cov = global_result.tier2_accuracy * global_result.tier2_coverage
    tier_weighted_accuracy = 0.6 * tier1_acc_cov + 0.4 * tier2_acc_cov

    # Domain-weighted accuracy (for analysis and feature dimensions).
    weights = DEFAULT_EVALUATION_CONFIG.domain_weights
    total_weight = sum(weights.values())
    weighted_accuracy = global_accuracy
    if domain_metrics:
        acc_sum = 0.0
        for domain_name, weight in weights.items():
            key = f"{domain_name}_accuracy"
            acc_sum += weight * domain_metrics.get(key, global_accuracy)
        weighted_accuracy = acc_sum / total_weight

    # Latency score, following the v2 architecture: Tier 1 is more
    # heavily weighted and has a stricter target.
    def _score_latency(p95_ms: float, target_ms: float) -> float:
        if p95_ms <= 0.0:
            return 0.0
        if p95_ms <= target_ms:
            return 1.0
        return max(0.0, target_ms / p95_ms)

    tier1_latency_score = _score_latency(
        global_result.tier1_p95_latency_ms,
        DEFAULT_EVALUATION_CONFIG.tier1_latency_target_ms,
    )
    tier2_latency_score = _score_latency(
        global_result.tier2_p95_latency_ms,
        DEFAULT_EVALUATION_CONFIG.tier2_latency_target_ms,
    )
    latency_score = 0.7 * tier1_latency_score + 0.3 * tier2_latency_score

    routing_score = global_result.routing_score
    syntactic = global_result.syntactic_validity

    # Combined scalar fitness, matching the v2 architecture: accuracy,
    # latency, routing, and syntactic validity.
    combined_score = (
        0.50 * tier_weighted_accuracy
        + 0.25 * latency_score
        + 0.15 * routing_score
        + 0.10 * syntactic
    )

    # Complexity penalty for overly large candidate files.
    code_text = Path(program_path).read_text(encoding="utf-8")
    code_len = float(len(code_text))
    max_len = 20000.0
    if code_len > max_len:
        combined_score *= max_len / code_len

    metrics: Dict[str, float] = {
        "combined_score": combined_score,
        "tier1_accuracy": global_result.tier1_accuracy,
        "tier2_accuracy": global_result.tier2_accuracy,
        "syntactic_validity": syntactic,
        "tier1_p95_latency_ms": global_result.tier1_p95_latency_ms,
        "tier2_p95_latency_ms": global_result.tier2_p95_latency_ms,
        "routing_score": routing_score,
        "tier_weighted_accuracy": tier_weighted_accuracy,
        "weighted_accuracy": weighted_accuracy,
        "latency_score": latency_score,
        "complexity": code_len,
    }
    metrics.update(domain_metrics)
    return metrics
