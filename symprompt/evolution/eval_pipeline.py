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
from typing import Callable, Dict, List, Union

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
) -> tuple[Dict[str, str], int, float]:
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
    return escalation.result, escalation.used_level, escalation.solver_latency_ms


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
        syntactic_total += 1

        try:
            result, used_level, solver_latency_ms = _translate_and_solve_with_escalation(
                pipeline,
                validator,
                decision,
                text,
                solve_fn,
                max_level=2,
            )
            syntactic_ok += 1
        except Exception:
            # Translation or validation failed - count as syntactic failure
            result = {"status": "UNKNOWN"}
            solver_latency_ms = 0.0

        correct = str(result.get("status")) == expected

        if decision.tier == 1:
            tier1_total += 1
            tier1_latencies.append(solver_latency_ms)
            if correct:
                tier1_correct += 1
        else:
            tier2_total += 1
            tier2_latencies.append(solver_latency_ms)
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


@dataclass
class EvalResultWithArtifacts:
    """Extended EvalResult that includes artifacts for LLM feedback."""

    result: EvalResult
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from evaluating a single benchmark."""

    item_id: str
    expected: str
    correct: bool
    syntactic_ok: bool
    routing_hit: bool
    latency_ms: float
    solver_status: str
    translation_error: str | None = None
    solver_error: str | None = None
    symil_debug: Dict | None = None


def _eval_single_benchmark(
    item: Dict[str, object],
    idx: int,
    router: SmartRouter,
    pipeline: TranslationPipeline,
    validator: SymILValidator,
    collect_artifacts: bool,
) -> BenchmarkResult:
    """Evaluate a single benchmark item (can be run in parallel)."""
    from symprompt.reasoning.portfolio import PortfolioDecision
    from symprompt.symil.profiles import get_profile

    text = str(item["nl"])
    expected = str(item["expected_result"])
    item_id = str(item.get("id", f"item_{idx + 1}"))

    decision: RoutingDecision = router.route(text, context=None)

    # Translation phase (includes LLM calls)
    symil = None
    translation_error = None
    symil_debug = None
    syntactic_ok = False
    try:
        profile = get_profile(decision.profile_name)
        hints = profile.translation_hints if profile else []
        symil = pipeline.translate(text, level=decision.symil_level, hints=hints)
        # Capture SymIL before validation for debugging
        if collect_artifacts and symil:
            symil_debug = {
                "facts": [str(f) for f in symil.facts],
                "rules": [str(r) for r in symil.rules],
                "query": str(symil.query) if symil.query else None,
                "constants": symil.ontology.constants if symil.ontology else [],
            }
        symil = validator.validate(symil)
        syntactic_ok = True
    except Exception as e:
        translation_error = str(e)

    # Solver phase (inference latency - measured)
    start = perf_counter()
    solver_error = None
    try:
        if symil is not None:
            portfolio_decision = PortfolioDecision(
                tier=decision.tier,
                profile_name=decision.profile_name,
                preferred_solver=decision.preferred_solver,
            )
            result = run_portfolio(symil=symil, decision=portfolio_decision)
        else:
            result = {"status": "UNKNOWN"}
    except Exception as e:
        result = {"status": "UNKNOWN"}
        solver_error = str(e)
    elapsed_ms = (perf_counter() - start) * 1000.0

    correct = str(result.get("status")) == expected
    ideal_tier = int(item.get("ideal_tier", decision.tier))
    routing_hit = correct and decision.tier == ideal_tier

    return BenchmarkResult(
        item_id=item_id,
        expected=expected,
        correct=correct,
        syntactic_ok=syntactic_ok,
        routing_hit=routing_hit,
        latency_ms=elapsed_ms,
        solver_status=str(result.get("status", "UNKNOWN")),
        translation_error=translation_error,
        solver_error=result.get("error") or solver_error,
        symil_debug=symil_debug if collect_artifacts else None,
    )


def evaluate_fast(
    router: SmartRouter,
    pipeline: TranslationPipeline,
    benchmarks: List[Dict[str, object]],
    collect_artifacts: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> Union[EvalResult, EvalResultWithArtifacts]:
    """
    Fast evaluation for evolution - no escalation, single attempt per benchmark.
    Follows v2 architecture Tier 1 principles: fast path, single solver.

    Uses ThreadPoolExecutor for parallel LLM calls (I/O bound).
    """
    validator = SymILValidator()

    # Default to env var but cap per-process thread count for stability.
    # High concurrency should primarily come from process-level parallelism
    # (OpenEvolve's evaluator.parallel_evaluations and islands), not an
    # unbounded number of threads inside each worker.
    if max_workers is None:
        env_workers = int(os.environ.get("EVAL_PARALLEL_BENCHMARKS", "16"))
        max_workers = max(1, min(env_workers, 8))

    total = len(benchmarks)
    if show_progress:
        print(f"Evaluating {total} benchmarks with {max_workers} workers...")

    # Thread-safe progress counter
    progress_lock = threading.Lock()
    completed_count = [0]  # Use list for mutable reference in closure
    pass_count = [0]

    def update_progress(result: BenchmarkResult) -> None:
        with progress_lock:
            completed_count[0] += 1
            if result.correct:
                pass_count[0] += 1
            if show_progress:
                status = "PASS" if result.correct else "FAIL"
                pct = completed_count[0] * 100 // total
                print(
                    f"  [{completed_count[0]}/{total}] {pct:3d}% {status} {result.item_id}",
                    flush=True,
                )

    # Run benchmarks in parallel
    results: List[BenchmarkResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _eval_single_benchmark,
                item,
                idx,
                router,
                pipeline,
                validator,
                collect_artifacts,
            ): idx
            for idx, item in enumerate(benchmarks)
        }
        for future in as_completed(futures):
            result = future.result()
            update_progress(result)
            results.append(result)

    if show_progress:
        print(
            f"Evaluation complete: {pass_count[0]}/{total} passed ({pass_count[0] * 100 // total}%)"
        )

    # Aggregate results
    tier1_latencies: List[float] = []
    tier1_correct = 0
    syntactic_ok = 0
    routing_hits = 0
    benchmark_results: List[str] = []
    validation_errors: List[str] = []

    for r in results:
        tier1_latencies.append(r.latency_ms)
        if r.correct:
            tier1_correct += 1
        if r.syntactic_ok:
            syntactic_ok += 1
        if r.routing_hit:
            routing_hits += 1

        if collect_artifacts:
            status = "PASS" if r.correct else "FAIL"
            notes = []
            if r.translation_error:
                notes.append(f"translation_error={r.translation_error[:100]}")
            if r.solver_error:
                notes.append(f"solver_error={r.solver_error[:100]}")
            notes_str = f" [{'; '.join(notes)}]" if notes else ""
            benchmark_results.append(
                f"{status} [{r.item_id}]: expected={r.expected} got={r.solver_status}{notes_str}"
            )
            if r.translation_error:
                error_detail = f"[{r.item_id}] {r.translation_error}"
                if r.symil_debug:
                    error_detail += f"\nSymIL: {r.symil_debug}"
                validation_errors.append(error_detail)

    tier1_total = len(benchmarks)

    benchmarks_count = len(benchmarks) if benchmarks else 1

    eval_result = EvalResult(
        tier1_accuracy=tier1_correct / tier1_total if tier1_total else 0.0,
        tier1_coverage=tier1_total / benchmarks_count,
        tier1_p95_latency_ms=_p95(tier1_latencies),
        tier2_accuracy=0.0,
        tier2_coverage=0.0,
        tier2_p95_latency_ms=0.0,
        syntactic_validity=syntactic_ok / tier1_total if tier1_total else 0.0,
        routing_score=routing_hits / tier1_total if tier1_total else 0.0,
    )

    if not collect_artifacts:
        return eval_result

    artifacts: Dict[str, str] = {}
    if benchmark_results:
        artifacts["benchmark_results"] = "\n".join(benchmark_results)
    if validation_errors:
        artifacts["validation_errors"] = "\n".join(validation_errors)

    return EvalResultWithArtifacts(result=eval_result, artifacts=artifacts)


def _load_benchmarks(root: Path, include_wild: bool = False) -> List[Dict[str, object]]:
    """Load all benchmark files for multi-domain evaluation."""
    benchmark_files = [
        "tiny_folio.json",  # 40 syllogisms
        "v2_math.json",  # 15 math
        "v2_planning.json",  # 15 planning
        "v2_legal.json",  # 15 legal
    ]
    if include_wild:
        benchmark_files.append("wild_prompts.json")  # 15 anti-overfitting

    benchmarks: List[Dict[str, object]] = []
    for name in benchmark_files:
        path = root / "benchmarks" / name
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                benchmarks.extend(data)
            except Exception:
                pass  # Skip invalid files
    return benchmarks


def evaluate(program_path: str) -> EvaluationResult:
    """
    OpenEvolve evaluator entrypoint.

    Multi-domain evaluation across syllogism, math, planning, and legal benchmarks.
    Follows v2 architecture with domain-weighted fitness.

    Returns EvaluationResult with metrics and artifacts for LLM feedback.
    """
    root = Path(__file__).resolve().parents[2]

    # Load all domain benchmarks
    # Include wild prompts if EVAL_INCLUDE_WILD=1 (default: disabled for speed)
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
            "candidate_pipeline", program_path
        )
        if spec is None or spec.loader is None:
            load_error = "Failed to create module spec"
        else:
            module = importlib.util.module_from_spec(spec)
            sys.modules["candidate_pipeline"] = module
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

    # Check for required methods
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

    # Check for translate method
    if not hasattr(pipeline, "translate"):
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={
                "error": "TranslationPipeline instance missing translate method"
            },
        )

    # Fast evaluation with artifact collection
    # Show progress if EVAL_SHOW_PROGRESS=1 (default: enabled)
    show_progress = os.environ.get("EVAL_SHOW_PROGRESS", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    try:
        eval_with_artifacts = evaluate_fast(
            router,
            pipeline,
            benchmarks,
            collect_artifacts=True,
            show_progress=show_progress,
        )
        eval_result = eval_with_artifacts.result
        artifacts = eval_with_artifacts.artifacts
    except Exception as e:
        code_len = float(len(Path(program_path).read_text(encoding="utf-8")))
        return EvaluationResult(
            metrics={"combined_score": 0.0, "complexity": code_len},
            artifacts={"error": f"Evaluation failed: {e}\n{traceback.format_exc()}"},
        )

    # Fitness function for Phase 1 (Tier 1 focus)
    # Weights: accuracy 60%, latency 15%, routing 15%, syntactic 10%
    accuracy = eval_result.tier1_accuracy
    syntactic = eval_result.syntactic_validity
    routing = eval_result.routing_score
    latency_score = (
        1.0
        if eval_result.tier1_p95_latency_ms < 50
        else 50.0 / max(eval_result.tier1_p95_latency_ms, 1.0)
    )

    combined_score = (
        0.60 * accuracy + 0.15 * latency_score + 0.15 * routing + 0.10 * syntactic
    )

    # Complexity penalty
    code_text = Path(program_path).read_text(encoding="utf-8")
    code_len = float(len(code_text))
    max_len = 20000.0
    if code_len > max_len:
        combined_score *= max_len / code_len

    metrics = {
        "combined_score": combined_score,
        "accuracy": accuracy,
        "syntactic_validity": syntactic,
        "latency_ms": eval_result.tier1_p95_latency_ms,
        "routing_score": eval_result.routing_score,
        "complexity": code_len,
    }

    return EvaluationResult(metrics=metrics, artifacts=artifacts)
