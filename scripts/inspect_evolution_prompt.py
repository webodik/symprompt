from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from openevolve.config import Config, DatabaseConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.prompt.sampler import PromptSampler


def load_database(config_path: Path, db_dir: Path) -> ProgramDatabase:
    """
    Load an existing OpenEvolve database from disk.
    """
    if not db_dir.exists():
        raise FileNotFoundError(f"Database directory not found: {db_dir}")

    base_config = Config.from_yaml(str(config_path))
    db_cfg = DatabaseConfig(
        db_path=str(db_dir),
        in_memory=False,
        population_size=base_config.database.population_size,
        archive_size=base_config.database.archive_size,
        num_islands=base_config.database.num_islands,
        feature_dimensions=base_config.database.feature_dimensions,
        feature_bins=base_config.database.feature_bins,
        migration_interval=base_config.database.migration_interval,
        migration_rate=base_config.database.migration_rate,
        random_seed=base_config.database.random_seed,
        log_prompts=base_config.database.log_prompts,
    )
    db = ProgramDatabase(db_cfg)
    db.load(str(db_dir))
    return db


def select_programs(programs: Dict[str, Program], program_id: Optional[str], limit: int) -> List[Program]:
    """
    Select one or more programs from the database.
    """
    if program_id:
        prog = programs.get(program_id)
        if prog is None:
            raise KeyError(f"Program id not found in database: {program_id}")
        return [prog]

    all_programs = list(programs.values())
    if not all_programs:
        return []

    def score(p: Program) -> float:
        metrics = p.metrics or {}
        return float(metrics.get("combined_score", 0.0))

    all_programs.sort(key=score, reverse=True)
    return all_programs[:limit]


def build_prompt_for_parent(
    config: Config,
    parent: Program,
    parent_artifacts: Dict[str, str] | None,
) -> str:
    """
    Rebuild the user prompt for a given parent program, focusing on the artifacts section.
    """
    prompt_config = config.prompt

    # Ensure template_dir is resolved relative to the repo root
    root = Path(__file__).resolve().parents[1]
    prompt_config.template_dir = str(root / "symprompt" / "evolution" / "prompts")

    sampler = PromptSampler(prompt_config)

    # Use a minimal set of fields; artifacts drive the Last Execution Output section.
    feature_dims = config.database.feature_dimensions or []
    artifacts = parent_artifacts or {}

    prompt = sampler.build_prompt(
        current_program=parent.code,
        parent_program=parent.code,
        program_metrics=parent.metrics or {},
        previous_programs=[],
        top_programs=[],
        inspirations=[],
        language=config.language or "python",
        evolution_round=-1,
        diff_based_evolution=config.diff_based_evolution,
        program_artifacts=artifacts,
        feature_dimensions=feature_dims,
    )

    return prompt["user"]


def extract_last_execution_output(user_prompt: str) -> str:
    """
    Extract the 'Last Execution Output' section from a user prompt.
    """
    marker = "## Last Execution Output"
    idx = user_prompt.find(marker)
    if idx == -1:
        return "## Last Execution Output\n\n[No artifacts section present in prompt]"
    return user_prompt[idx:]


def resolve_db_dir(root: Path, kind: str) -> Path:
    """
    Map evolution kind to the corresponding database directory.
    """
    if kind == "translation":
        return root / "evolution" / "openevolve_output" / "evolution.db"
    if kind == "router":
        return root / "evolution" / "openevolve_output_router" / "evolution.db"
    if kind == "profiles":
        return root / "evolution" / "openevolve_output_profiles" / "evolution.db"
    raise ValueError(f"Unknown evolution kind: {kind}")


def apply_prompt_overrides(config: Config, kind: str) -> None:
    """
    Apply the same prompt overrides as the run_*_evolution scripts.
    """
    if kind == "translation":
        config.prompt.system_message = "system_message"
        config.prompt.evaluator_system_message = "system_message"
    elif kind == "router":
        config.prompt.system_message = "router_system_message"
        config.prompt.evaluator_system_message = "router_system_message"
    elif kind == "profiles":
        config.prompt.system_message = "profiles_system_message"
        config.prompt.evaluator_system_message = "profiles_system_message"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect evolution prompts and artifacts for SymPrompt/OpenEvolve runs."
    )
    parser.add_argument(
        "--kind",
        choices=["translation", "router", "profiles"],
        default="translation",
        help="Which evolution database to inspect.",
    )
    parser.add_argument(
        "--program-id",
        help="Specific program id to inspect (defaults to top programs by combined_score).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many top programs to inspect when program-id is not provided.",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / "openevolve_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    db_dir = resolve_db_dir(root, args.kind)
    config = Config.from_yaml(str(config_path))
    apply_prompt_overrides(config, args.kind)

    db = load_database(config_path, db_dir)

    programs = select_programs(db.programs, args.program_id, args.limit)
    if not programs:
        print("No programs found in database.")
        return

    for prog in programs:
        print("=" * 80)
        print(f"Program id:   {prog.id}")
        print(f"Parent id:    {prog.parent_id or '[none]'}")
        print(f"Metrics:      {prog.metrics}")

        parent_id = prog.parent_id
        if not parent_id or parent_id not in db.programs:
            print("\n[No parent program or parent not found in database; no parent artifacts to inspect.]")
            continue

        parent = db.programs[parent_id]
        parent_artifacts = db.get_artifacts(parent_id)
        if not parent_artifacts:
            print("\n[Parent has no stored artifacts; Last Execution Output will be empty.]")
            continue

        user_prompt = build_prompt_for_parent(config, parent, parent_artifacts)
        last_output = extract_last_execution_output(user_prompt)
        print("\n--- Reconstructed 'Last Execution Output' section for parent prompt ---\n")
        print(last_output)
        print()


if __name__ == "__main__":
    main()

