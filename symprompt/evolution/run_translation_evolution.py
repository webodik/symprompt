from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from openevolve.config import Config, LLMConfig, LLMModelConfig
from openevolve.controller import OpenEvolve
from openevolve.database import Program

from symprompt.evolution.backups import backup_database
from symprompt.evolution.litellm_client import LiteLLMLLM
from symprompt.evolution.seeds import extract_top_programs, SeedProgram


async def run_evolution_with_final_save(
    initial_program: str,
    evaluator: str,
    config: Config,
    output_dir: str,
    iterations: int | None = None,
    checkpoint_path: str | None = None,
    seeds: List[SeedProgram] | None = None,
) -> dict:
    """
    Run evolution using the OpenEvolve controller directly.

    This allows us to call database.save() at the end to persist prompts
    and artifacts (workaround for OpenEvolve bug where prompts/artifacts
    are only persisted during checkpoints, not at evolution end).

    Seeds are injected into the database before evolution starts, allowing
    the best programs from previous runs to serve as starting points.
    """
    controller = OpenEvolve(
        initial_program_path=initial_program,
        evaluation_file=evaluator,
        config=config,
        output_dir=output_dir,
    )

    # Inject seeds into database before evolution starts
    if seeds and not checkpoint_path:
        print(f"Injecting {len(seeds)} seed programs from previous evolution...")
        for seed in seeds:
            # Re-evaluate seed to get current metrics
            seed_metrics = await controller.evaluator.evaluate_program(
                seed.code, seed.program_id
            )
            seed_program = Program(
                id=str(uuid.uuid4()),  # New ID for this evolution
                code=seed.code,
                language=config.language,
                parent_id=seed.program_id,  # Track lineage
                generation=0,
                iteration_found=0,
                metrics=seed_metrics,
            )
            controller.database.add(seed_program)
            score = seed_metrics.get("combined_score", 0.0)
            print(f"  Seed {seed.program_id[:8]}... -> score={score:.4f}")

    best_program = await controller.run(iterations=iterations, checkpoint_path=checkpoint_path)

    # Final save to persist prompts and artifacts (OpenEvolve bug workaround)
    final_save_path = os.path.join(output_dir, "evolution.db")
    controller.database.save(final_save_path)

    return best_program.metrics if best_program else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run translation pipeline evolution.")
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=None,
        help="Number of evolution iterations (overrides config)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (evolution.db)",
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

    output_dir = root / "evolution" / "openevolve_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "evolution.db"
    config.database.db_path = str(db_path)
    config.database.in_memory = False

    # Handle backup and seeds
    checkpoint_path = None
    seeds = None

    if args.resume:
        # Resume mode: load from existing checkpoint
        if db_path.exists():
            checkpoint_path = str(db_path)
            print(f"Resuming evolution from checkpoint: {checkpoint_path}")
        else:
            print("Warning: --resume specified but no checkpoint found, starting fresh")
    else:
        # Fresh run: backup previous evolution and extract seeds
        if db_path.exists():
            # Extract seeds BEFORE backup (from current db)
            print("Extracting top programs from previous evolution as seeds...")
            seeds = extract_top_programs(
                config_path=config_path,
                db_path=db_path,
                metric="combined_score",
                limit=5,
            )
            if seeds:
                print(f"  Found {len(seeds)} seed programs")
            else:
                print("  No seeds found (empty database)")

            # Create backup
            backup_path = backup_database(db_path)
            if backup_path is not None:
                print(f"Created evolution DB backup at {backup_path}")

    initial_program = root / "symprompt" / "translation" / "pipeline.py"
    evaluator = root / "symprompt" / "evolution" / "eval_pipeline.py"

    metrics = asyncio.run(
        run_evolution_with_final_save(
            initial_program=str(initial_program),
            evaluator=str(evaluator),
            config=config,
            output_dir=str(output_dir),
            iterations=args.iterations,
            checkpoint_path=checkpoint_path,
            seeds=seeds,
        )
    )

    print("Evolution finished.")
    print("Best metrics:", metrics)


if __name__ == "__main__":
    main()
