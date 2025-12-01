from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from symprompt.evolution.seeds import SeedProgram, extract_top_programs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # Load .env from repo root (for any downstream config usage)
    load_dotenv(root / ".env")
    config_path = root / "openevolve_config.yaml"
    output_dir = root / "evolution" / "openevolve_output"
    db_path = output_dir / "evolution.db"

    seeds: list[SeedProgram] = extract_top_programs(config_path, db_path, metric="combined_score", limit=5)

    if not seeds:
        print("No seeds extracted (database missing or empty).")
        return

    seeds_dir = root / "evolution" / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    for idx, seed in enumerate(seeds, start=1):
        filename = f"seed_{idx}_{seed.program_id[:8]}.py"
        path = seeds_dir / filename
        path.write_text(seed.code, encoding="utf-8")
        print(f"Wrote seed program {seed.program_id} to {path}")


if __name__ == "__main__":
    main()
