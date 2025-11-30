from __future__ import annotations

from pathlib import Path

from symprompt.evolution.seeds import extract_top_programs


def test_extract_top_programs_handles_missing_db(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("max_iterations: 1\n", encoding="utf-8")
    db_path = tmp_path / "evolution.db"

    seeds = extract_top_programs(config, db_path, limit=3)
    assert seeds == []

