from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List

import importlib.util
import json
import sys

from openevolve.evaluation_result import EvaluationResult

from symprompt.evolution.eval_pipeline import evaluate_fast
from symprompt.llm.sync_client import build_default_sync_client
from symprompt.router.smart_router import SmartRouter
from symprompt.translation.pipeline import TranslationPipeline


def evaluate(program_path: str) -> EvaluationResult:
    """
    OpenEvolve evaluator for router evolution.

    Loads a candidate SmartRouter implementation from program_path,
    then evaluates it with a fixed TranslationPipeline over the full
    benchmark suite (tiny_folio + v2_* slices).
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
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": "No benchmarks found"},
        )

    # Load candidate router module
    try:
        spec = importlib.util.spec_from_file_location("candidate_router", program_path)
        if spec is None or spec.loader is None:
            return EvaluationResult(
                metrics={"combined_score": 0.0},
                artifacts={"error": "Failed to create module spec"},
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules["candidate_router"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": f"Module load error: {e}"},
        )

    router_cls = getattr(module, "SmartRouter", None)
    if router_cls is None:
        return EvaluationResult(
            metrics={"combined_score": 0.0},
            artifacts={"error": "SmartRouter class not found in module"},
        )

    llm_client = build_default_sync_client()
    pipeline = TranslationPipeline.from_llm_client(llm_client)
    router = router_cls()

    start = perf_counter()
    # Use parallel evaluation with artifacts for LLM feedback
    eval_with_artifacts = evaluate_fast(router, pipeline, benchmarks, collect_artifacts=True, show_progress=True)
    elapsed = (perf_counter() - start) * 1000.0

    eval_result = eval_with_artifacts.result
    artifacts = eval_with_artifacts.artifacts

    # Router evolution fitness function
    # Emphasizes routing quality (tier/profile selection) over raw accuracy
    # Weights: routing 40%, accuracy 30%, latency 20%, syntactic 10%
    accuracy = max(eval_result.tier1_accuracy, eval_result.tier2_accuracy)
    latency_score = 1.0 if eval_result.tier1_p95_latency_ms < 50 else 50.0 / max(eval_result.tier1_p95_latency_ms, 1.0)

    combined_score = (
        0.40 * eval_result.routing_score
        + 0.30 * accuracy
        + 0.20 * latency_score
        + 0.10 * eval_result.syntactic_validity
    )

    metrics = {
        "combined_score": combined_score,
        "tier1_accuracy": eval_result.tier1_accuracy,
        "tier2_accuracy": eval_result.tier2_accuracy,
        "syntactic_validity": eval_result.syntactic_validity,
        "tier1_p95_latency_ms": eval_result.tier1_p95_latency_ms,
        "tier2_p95_latency_ms": eval_result.tier2_p95_latency_ms,
        "routing_score": eval_result.routing_score,
        "latency_score": latency_score,
        "evaluation_time_ms": elapsed,
    }

    return EvaluationResult(metrics=metrics, artifacts=artifacts)

