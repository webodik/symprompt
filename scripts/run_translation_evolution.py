#!/usr/bin/env python
"""
Translation pipeline evolution runner.

Evolves the TranslationPipeline component (NL -> SymIL translation).

Usage:
    EVOLVE_LLM_MODEL="openrouter/x-ai/grok-4.1-fast:free" \
    EVOLVE_ITERATIONS=100 \
    .venv/bin/python scripts/run_translation_evolution.py

Environment variables:
    EVOLVE_LLM_MODEL: LLM model to use (default: from config)
    EVOLVE_ITERATIONS: Number of iterations (default: from config)
    EVOLVE_RESUME: Set to "1" to resume from checkpoint
    EVAL_PARALLEL_BENCHMARKS: Number of parallel benchmark workers (default: 8)
    EVAL_SHOW_PROGRESS: Show per-benchmark progress (default: "1", set "0" to disable)
"""
from __future__ import annotations

import logging
import os
import sys

# Suppress verbose LiteLLM logging before any imports
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    # Parse environment variables
    iterations_str = os.environ.get("EVOLVE_ITERATIONS")
    resume = os.environ.get("EVOLVE_RESUME", "").lower() in ("1", "true", "yes")
    mode = os.environ.get("EVOLVE_MODE", "phase1").lower()
    if mode not in ("phase1", "phase2"):
        mode = "phase1"

    # Set model via SYMPROMPT_LLM_MODEL (used by the runner)
    llm_model = os.environ.get("EVOLVE_LLM_MODEL")
    if llm_model:
        os.environ["SYMPROMPT_LLM_MODEL"] = llm_model

    # Build argv for the runner
    sys.argv = ["run_translation_evolution"]
    if iterations_str:
        sys.argv.extend(["-n", iterations_str])
    if resume:
        sys.argv.append("--resume")
    if mode:
        sys.argv.extend(["--mode", mode])

    # Import and run
    from symprompt.evolution.run_translation_evolution import main as run_main
    run_main()


if __name__ == "__main__":
    main()
