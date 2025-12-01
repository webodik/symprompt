from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from symprompt.evolution.eval_pipeline import EvalResult, evaluate_system
from symprompt.router.smart_router import SmartRouter
from symprompt.translation.pipeline import TranslationPipeline
from symprompt.llm.sync_client import build_default_sync_client


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # Load .env from repo root for LLM configuration
    load_dotenv(root / ".env")
    benchmarks_path = root / "benchmarks" / "tiny_folio.json"
    benchmarks = json.loads(benchmarks_path.read_text(encoding="utf-8"))

    router = SmartRouter()
    llm_client = build_default_sync_client()
    pipeline = TranslationPipeline.from_llm_client(llm_client)

    result: EvalResult = evaluate_system(router, pipeline, benchmarks)

    print("Tier1 accuracy:", result.tier1_accuracy)
    print("Tier1 coverage:", result.tier1_coverage)
    print("Tier1 p95 latency (ms):", result.tier1_p95_latency_ms)
    print("Tier2 accuracy:", result.tier2_accuracy)
    print("Tier2 coverage:", result.tier2_coverage)
    print("Tier2 p95 latency (ms):", result.tier2_p95_latency_ms)
    print("Syntactic validity:", result.syntactic_validity)
    print("Routing score:", result.routing_score)


if __name__ == "__main__":
    main()
