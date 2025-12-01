from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from openevolve.config import Config, LLMConfig, LLMModelConfig
from openevolve import run_evolution

from symprompt.evolution.backups import backup_database
from symprompt.evolution.litellm_client import LiteLLMLLM

# Suppress verbose LiteLLM logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run router evolution.")
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=None,
        help="Number of evolution iterations (overrides config)",
    )
    args = parser.parse_args()

    load_dotenv()
    root = Path(__file__).resolve().parents[2]
    config_path = root / "openevolve_config.yaml"

    config = Config.from_yaml(config_path)

    if args.iterations is not None:
        config.max_iterations = args.iterations

    model_name = (
        os.getenv("SYMPROMPT_LLM_MODEL")
        or config.llm.primary_model
        or "openrouter/x-ai/grok-4.1-fast:free"
    )

    model_cfg = LLMModelConfig(
        name=model_name,
        weight=1.0,
        init_client=LiteLLMLLM,
        max_tokens=8192,
    )
    config.llm = LLMConfig(
        primary_model=model_name,
        temperature=config.llm.temperature,
        timeout=config.llm.timeout,
        retries=config.llm.retries,
        retry_delay=config.llm.retry_delay,
    )
    config.llm.models = [model_cfg]
    config.llm.evaluator_models = [model_cfg]

    output_dir = root / "evolution" / "openevolve_output_router"
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "evolution.db"
    config.database.db_path = str(db_path)
    config.database.in_memory = False

    backup_path = backup_database(db_path)
    if backup_path is not None:
        print(f"Created router evolution DB backup at {backup_path}")

    initial_program = root / "symprompt" / "router" / "smart_router.py"
    evaluator = root / "symprompt" / "evolution" / "eval_router.py"

    result = run_evolution(
        initial_program=str(initial_program),
        evaluator=str(evaluator),
        config=config,
        output_dir=str(output_dir),
        cleanup=False,
    )

    print("Router evolution finished.")
    print("Best metrics:", result.metrics)


if __name__ == "__main__":
    main()

