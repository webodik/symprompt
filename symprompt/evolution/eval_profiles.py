from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Dict, List

import importlib.util
import json
import sys

from symprompt.evolution.eval_pipeline import evaluate_system
from symprompt.llm.sync_client import build_default_sync_client
from symprompt.router.smart_router import SmartRouter
from symprompt.translation.pipeline import TranslationPipeline


def evaluate(program_path: str) -> Dict[str, float]:
    """
    OpenEvolve evaluator for SymIL profile evolution.

    Mutating the profiles module changes the behavior of get_profile()
    and thus affects routing and translation hints. This evaluator
    loads the candidate profiles implementation and then evaluates
    the system as a whole using the fixed SmartRouter and
    TranslationPipeline.
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

    # Load candidate profiles module
    try:
        spec = importlib.util.spec_from_file_location("candidate_profiles", program_path)
        if spec is None or spec.loader is None:
            return {"combined_score": 0.0}
        module = importlib.util.module_from_spec(spec)
        sys.modules["symprompt.symil.profiles"] = module  # override for get_profile
        spec.loader.exec_module(module)
    except Exception:
        return {"combined_score": 0.0}

    llm_client = build_default_sync_client()
    pipeline = TranslationPipeline.from_llm_client(llm_client)
    router = SmartRouter()

    start = perf_counter()
    result = evaluate_system(router, pipeline, benchmarks)
    elapsed = (perf_counter() - start) * 1000.0

    accuracy = max(result.tier1_accuracy, result.tier2_accuracy)
    combined_score = 0.6 * accuracy + 0.4 * result.syntactic_validity

    return {
        "combined_score": combined_score,
        "tier1_accuracy": result.tier1_accuracy,
        "tier2_accuracy": result.tier2_accuracy,
        "syntactic_validity": result.syntactic_validity,
        "tier1_p95_latency_ms": result.tier1_p95_latency_ms,
        "tier2_p95_latency_ms": result.tier2_p95_latency_ms,
        "routing_score": result.routing_score,
        "evaluation_time_ms": elapsed,
    }

