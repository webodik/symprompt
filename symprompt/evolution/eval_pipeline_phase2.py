from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Dict, List

from openevolve.evaluation_result import EvaluationResult

from symprompt.config import DEFAULT_EVALUATION_CONFIG
from symprompt.integration.escalation import (
    EscalationResult,
    translate_and_solve_with_escalation,
)
from symprompt.llm.sync_client import build_default_sync_client
from symprompt.reasoning.portfolio import run_portfolio
from symprompt.router.smart_router import RoutingDecision, SmartRouter
from symprompt.symil.validator import SymILValidator
from symprompt.translation.pipeline import TranslationPipeline

from symprompt.evolution.eval_pipeline import _load_benchmarks, _p95


@dataclass
class TierStats:
    total: int = 0
    correct: int = 0
    latencies_ms: List[float] = field(default_factory=list)


@dataclass
class DomainStats:
    total: int = 0
    tier1: TierStats = field(default_factory=TierStats)
    tier2: TierStats = field(default_factory=TierStats)


def _score_latency(latency_ms: float, target_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return 1.0 if latency_ms < target_ms else target_ms / max(latency_ms, 1.0)


@dataclass
class Phase2BenchmarkResult:
    item_id: str
    domain: str
    tier: int
    ideal_tier: int
    expected: str
    status: str
    correct: bool
    latency_ms: float
    syntactic_ok: bool
    routing_hit: bool
    translation_error: str | None = None


def _eval_single_benchmark_phase2(
    item: Dict[str, object],
    idx: int,
    router: SmartRouter,
    pipeline: TranslationPipeline,
    validator: SymILValidator,
) -> Phase2BenchmarkResult:
    """
    Evaluate a single benchmark item with escalation (used in parallel).
    """
    text = str(item["nl"])
    expected = str(item["expected_result"])
    ideal_tier = int(item.get("ideal_tier", 1))
    domain = str(item.get("domain", "unknown"))
    item_id = str(item.get("id", f"item_{idx + 1}"))

    decision: RoutingDecision = router.route(text, context=None)

    translation_error: str | None = None
    syntactic_ok = False
    try:
        esc: EscalationResult = translate_and_solve_with_escalation(
            pipeline=pipeline,
            validator=validator,
            decision=decision,
            text=text,
            solve_fn=run_portfolio,
            max_level=2,
        )
        result = esc.result
        syntactic_ok = True
    except Exception as e:
        result = {"status": "UNKNOWN"}
        translation_error = str(e)

    status = str(result.get("status", "UNKNOWN"))
    correct = status == expected
    routing_hit = correct and decision.tier == ideal_tier

    return Phase2BenchmarkResult(
        item_id=item_id,
        domain=domain,
        tier=decision.tier,
        ideal_tier=ideal_tier,
        expected=expected,
        status=status,
        correct=correct,
        latency_ms=esc.solver_latency_ms if syntactic_ok else 0.0,
        syntactic_ok=syntactic_ok,
        routing_hit=routing_hit,
        translation_error=translation_error,
    )


def evaluate(program_path: str) -> EvaluationResult:
    """
    OpenEvolve evaluator entrypoint (Phase 2 / full system).

    - Uses router + progressive SymIL level escalation.
    - Aggregates Tier 1 and Tier 2 metrics.
    - Applies domain-weighted, tier-weighted accuracy as in v2 architecture.
    """
    root = Path(__file__).resolve().parents[2]

    include_wild = os.environ.get("EVAL_INCLUDE_WILD", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    benchmarks = _load_benchmarks(root, include_wild=include_wild)

    if not benchmarks:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": "No benchmarks found"},
        )

    # Dynamically load candidate pipeline module from program_path
    load_error = None
    try:
        spec = importlib.util.spec_from_file_location(
            "candidate_pipeline_phase2", program_path
        )
        if spec is None or spec.loader is None:
            load_error = "Failed to create module spec"
        else:
            module = importlib.util.module_from_spec(spec)
            sys.modules["candidate_pipeline_phase2"] = module
            spec.loader.exec_module(module)
    except Exception as e:
        load_error = f"Module load error: {e}\n{traceback.format_exc()}"

    if load_error:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": load_error},
        )

    pipeline_cls = getattr(module, "TranslationPipeline", None)
    if pipeline_cls is None:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": "TranslationPipeline class not found in module"},
        )

    if not hasattr(pipeline_cls, "from_llm_client"):
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={
                "error": "TranslationPipeline missing from_llm_client classmethod"
            },
        )

    router = SmartRouter()
    llm_client = build_default_sync_client()

    try:
        pipeline: TranslationPipeline = pipeline_cls.from_llm_client(llm_client)
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": f"Pipeline initialization failed: {e}"},
        )

    if not hasattr(pipeline, "translate"):
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={
                "error": "TranslationPipeline instance missing translate method"
            },
        )

    validator = SymILValidator()

    # Evaluate benchmarks in parallel (I/O-bound due to LLM/solvers)
    env_workers = int(os.environ.get("EVAL_PARALLEL_BENCHMARKS", "16"))
    max_workers = max(1, min(env_workers, 8))
    show_progress = os.environ.get("EVAL_SHOW_PROGRESS", "1").lower() in (
        "1",
        "true",
        "yes",
    )

    total_benchmarks = len(benchmarks)

    progress_lock = threading.Lock()
    completed_count = [0]
    pass_count = [0]

    def update_progress(res: Phase2BenchmarkResult) -> None:
        with progress_lock:
            completed_count[0] += 1
            if res.correct:
                pass_count[0] += 1
            if show_progress:
                status = "PASS" if res.correct else "FAIL"
                pct = completed_count[0] * 100 // total_benchmarks
                print(
                    f"  [{completed_count[0]}/{total_benchmarks}] {pct:3d}% {status} {res.item_id}",
                    flush=True,
                )

    results: List[Phase2BenchmarkResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _eval_single_benchmark_phase2,
                item,
                idx,
                router,
                pipeline,
                validator,
            ): idx
            for idx, item in enumerate(benchmarks)
        }
        for future in as_completed(futures):
            res = future.result()
            update_progress(res)
            results.append(res)

    if show_progress:
        print(
            f"Evaluation complete: {pass_count[0]}/{total_benchmarks} passed "
            f"({pass_count[0] * 100 // total_benchmarks}%)"
        )

    # Global metrics
    domain_stats: Dict[str, DomainStats] = {}
    syntactic_ok = 0
    routing_hits = 0
    routing_total = 0

    benchmark_results: List[str] = []
    validation_errors: List[str] = []

    for r in results:
        stats = domain_stats.setdefault(r.domain, DomainStats())
        stats.total += 1

        # Tier-wise accounting
        if r.tier == 1:
            stats.tier1.total += 1
            stats.tier1.latencies_ms.append(r.latency_ms)
            if r.correct:
                stats.tier1.correct += 1
        else:
            stats.tier2.total += 1
            stats.tier2.latencies_ms.append(r.latency_ms)
            if r.correct:
                stats.tier2.correct += 1

        if r.syntactic_ok:
            syntactic_ok += 1
        routing_total += 1
        if r.routing_hit:
            routing_hits += 1

        notes: List[str] = []
        if r.translation_error:
            notes.append(f"translation_error={r.translation_error[:100]}")
        notes_str = f" [{'; '.join(notes)}]" if notes else ""
        benchmark_results.append(
            f"{'PASS' if r.correct else 'FAIL'} [{r.item_id}]: expected={r.expected} got={r.status}{notes_str}"
        )
        if r.translation_error:
            validation_errors.append(f"[{r.item_id}] {r.translation_error}")

    # Aggregate domain-weighted, tier-weighted accuracy
    domain_weights_cfg = DEFAULT_EVALUATION_CONFIG.domain_weights
    domain_accuracies: Dict[str, float] = {}

    total_tier1_total = 0
    total_tier1_correct = 0
    total_tier2_total = 0
    total_tier2_correct = 0
    tier1_latencies_all: List[float] = []
    tier2_latencies_all: List[float] = []

    for domain, ds in domain_stats.items():
        t1_total = ds.tier1.total
        t2_total = ds.tier2.total
        total = ds.total or 1

        t1_acc = ds.tier1.correct / t1_total if t1_total else 0.0
        t2_acc = ds.tier2.correct / t2_total if t2_total else 0.0

        t1_cov = t1_total / total if total else 0.0
        t2_cov = t2_total / total if total else 0.0

        tier1_acc_weighted = t1_acc * t1_cov
        tier2_acc_weighted = t2_acc * t2_cov
        domain_accuracies[domain] = 0.6 * tier1_acc_weighted + 0.4 * tier2_acc_weighted

        total_tier1_total += t1_total
        total_tier1_correct += ds.tier1.correct
        total_tier2_total += t2_total
        total_tier2_correct += ds.tier2.correct
        tier1_latencies_all.extend(ds.tier1.latencies_ms)
        tier2_latencies_all.extend(ds.tier2.latencies_ms)

    # Normalize domain weights to present domains
    present_weights: Dict[str, float] = {}
    for d, acc in domain_accuracies.items():
        w = float(domain_weights_cfg.get(d, 0.0))
        if w > 0.0:
            present_weights[d] = w

    if not present_weights:
        # Fallback: equal weighting over domains seen
        equal_w = 1.0 / max(len(domain_accuracies), 1)
        present_weights = {d: equal_w for d in domain_accuracies.keys()}

    weight_sum = sum(present_weights.values()) or 1.0
    accuracy = sum(
        (present_weights[d] / weight_sum) * domain_accuracies[d]
        for d in domain_accuracies.keys()
    )

    # Global tier metrics
    tier1_accuracy = (
        total_tier1_correct / total_tier1_total if total_tier1_total else 0.0
    )
    tier2_accuracy = (
        total_tier2_correct / total_tier2_total if total_tier2_total else 0.0
    )
    tier1_p95_latency_ms = _p95(tier1_latencies_all)
    tier2_p95_latency_ms = _p95(tier2_latencies_all)

    syntactic_validity = (
        syntactic_ok / total_benchmarks if total_benchmarks else 0.0
    )
    routing_score = routing_hits / routing_total if routing_total else 0.0

    # Latency score as in v2 Phase 2 fitness
    latency_score = (
        0.7 * _score_latency(tier1_p95_latency_ms, target_ms=50.0)
        + 0.3 * _score_latency(tier2_p95_latency_ms, target_ms=500.0)
    )

    combined_score = (
        0.50 * accuracy
        + 0.25 * latency_score
        + 0.15 * routing_score
        + 0.10 * syntactic_validity
    )

    # Complexity penalty (same as Phase 1)
    code_text = Path(program_path).read_text(encoding="utf-8")
    code_len = float(len(code_text))
    max_len = 20000.0
    if code_len > max_len:
        combined_score *= max_len / code_len

    metrics = {
        "combined_score": combined_score,
        "accuracy": accuracy,
        "tier1_accuracy": tier1_accuracy,
        "tier2_accuracy": tier2_accuracy,
        "tier1_p95_latency_ms": tier1_p95_latency_ms,
        "tier2_p95_latency_ms": tier2_p95_latency_ms,
        "latency_score": latency_score,
        "routing_score": routing_score,
        "syntactic_validity": syntactic_validity,
        "complexity": code_len,
    }

    artifacts: Dict[str, str] = {}
    if benchmark_results:
        artifacts["benchmark_results"] = "\n".join(benchmark_results)
    if validation_errors:
        artifacts["validation_errors"] = "\n".join(validation_errors)

    return EvaluationResult(metrics=metrics, artifacts=artifacts)
