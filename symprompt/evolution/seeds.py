from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from openevolve.config import Config
from openevolve.database import DatabaseConfig, ProgramDatabase


@dataclass
class SeedProgram:
    program_id: str
    code: str


def _load_database(config_path: Path, db_path: Path) -> ProgramDatabase | None:
    """Load an existing OpenEvolve database from disk."""
    if not db_path.exists():
        return None

    base_config = Config.from_yaml(config_path)
    db_cfg = DatabaseConfig(
        db_path=str(db_path),
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
    # Load programs from disk into memory
    db.load(str(db_path))
    return db


def extract_top_programs(
    config_path: Path,
    db_path: Path,
    metric: str = "combined_score",
    limit: int = 5,
) -> List[SeedProgram]:
    """
    Extract top programs from a previous OpenEvolve run as seeds.

    Returns a list of SeedProgram objects with program_id and code.
    If no database is present, returns an empty list.
    """
    db = _load_database(config_path, db_path)
    if db is None:
        return []

    programs = list(db.programs.values())
    if not programs:
        return []

    def score(prog) -> float:
        return float(prog.metrics.get(metric, 0.0))

    programs.sort(key=score, reverse=True)
    top = programs[:limit]

    return [SeedProgram(program_id=p.id, code=p.code) for p in top]

