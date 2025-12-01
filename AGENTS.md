# Repository Guidelines

This repository contains the SymPrompt framework: a Python 3.11+ toolkit for translating natural language into neuro-symbolic representations and verifying them with solvers like Z3.

## Project Structure & Module Organization
- Core code lives under `symprompt/`:
  - `symprompt/symil/`: SymIL datamodel, validation, and examples.
  - `symprompt/translation/`: NL → SymIL pipeline (preprocessor, ontology, logical translation, constraints).
  - `symprompt/reasoning/`: SymIL → solver compilers and runners (e.g., Z3).
  - `symprompt/evolution/`: OpenEvolve integration and fitness evaluation.
  - `symprompt/integration/`: CLI, router, and notebook demos.
- Tests live in `tests/` (e.g., `test_symil.py`, `test_translation.py`, `test_reasoning.py`).
- Benchmarks and configs live in `benchmarks/` and `openevolve_config.yaml`.
- Architecture and planning documents live in `docs/`.

## Build, Test, and Development Commands
- Create and activate a virtual environment, then install dependencies:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -e .[dev]` or `pip install -r requirements.txt`
- Run tests: `pytest` from the repository root (runs all tests under `tests/`).
- Run the CLI once implemented: `python -m symprompt.integration.cli`.
- Optionally run formatting and linting: `black symprompt tests` and `ruff symprompt tests`.

## Coding Style & Naming Conventions
- Language: Python 3.11+, 4-space indentation, UTF-8 source files.
- Follow PEP 8 where reasonable; prefer readability over cleverness.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Keep modules cohesive and small; follow the existing package boundaries (`symil`, `translation`, `reasoning`, `evolution`, `integration`).
- Use `black` and `ruff` (or their equivalents) to keep formatting and imports consistent.

## Testing Guidelines
- Use `pytest` with tests in `tests/` named `test_*.py`.
- Each new feature should have unit tests covering:
  - SymIL construction and validation.
  - Translation pipeline behavior on representative NL examples.
  - Reasoning correctness against small benchmark cases (e.g., `benchmarks/tiny_folio.json`).
- Run `pytest` before pushing or opening a pull request; add regression tests for any bug fixes.

## Commit & Pull Request Guidelines
- Write concise, imperative commit messages (e.g., `add symil validator`, `fix z3 runner edge cases`); group related changes into a single commit where possible.
- For pull requests:
  - Provide a short summary of the change, implementation notes, and any trade-offs.
  - Reference related issues or documents (e.g., sections in `docs/SymPrompt_Architecture.md` or `docs/SymPrompt_dev_plan.md`).
  - Include test output or a brief note on how you verified the changes (commands run, benchmarks used).

## Security & Configuration Tips
- Keep API keys and credentials out of version control; use environment variables or a local config file (e.g., `.env` or `symprompt/config.py` excluded via `.gitignore`).
- Default to local, offline operation where possible; avoid sending benchmark or user data to external services unless explicitly configured to do so.
- When extending solver or LLM integrations, prefer explicit configuration (paths, URLs, timeouts) over hidden defaults.

## Development Progress Notes
- Initial v2 project skeleton created:
  - Core package `symprompt/` with SymIL datamodel (`symprompt/symil/model.py`), validator, profiles, and examples.
  - LiteLLM-backed LLM clients in `symprompt/llm/litellm_client.py` and `symprompt/llm/sync_client.py` wired for OpenEvolve and CLI use.
  - Baseline NL → SymIL translation pipeline (`symprompt/translation/*`) and Z3 reasoning path (`symprompt/reasoning/*`), plus ASP/Scallop/VSA compilers and stub runners under `symprompt/reasoning/`; Clingo-backed ASP solver (`symprompt/reasoning/clingo_runner.py`) classifies simple atomic queries.
  - Heuristic router with feature extraction in `symprompt/router/features.py` and `symprompt/router/smart_router.py`, and a router adapter (`symprompt/integration/router_adapter.py`) that uses profile-aware translation hints, solver-driven SymIL level escalation, and a Tier 0 BYPASS path; covered by `tests/test_router.py` and `tests/test_router_adapter.py`.
  - Evaluation harness in `symprompt/evolution/eval_pipeline.py` for OpenEvolve fitness computation, including solver-driven level escalation, profile-aware refinement, and an `evaluate(program_path)` entrypoint over multi-domain benchmarks (`tiny_folio.json`, `v2_syllogism.json`, `v2_math.json`, `v2_planning.json`, `v2_legal.json`); plus a tiny benchmark + eval script (`benchmarks/tiny_folio.json`, `scripts/run_eval_tiny.py`).
  - Translation pipeline tested end-to-end (ontology + logical translation) using deterministic dummy LLMs in `tests/test_translation_pipeline.py` and `tests/test_translation_escalation.py`.
  - OpenEvolve config (`openevolve_config.yaml`) and evolution runners:
    - Translation evolution (`symprompt/evolution/run_translation_evolution.py`),
    - Router evolution (`symprompt/evolution/run_router_evolution.py`, evaluator `symprompt/evolution/eval_router.py`),
    - SymIL profile evolution (`symprompt/evolution/run_profile_evolution.py`, evaluator `symprompt/evolution/eval_profiles.py`),
    each backing up the evolution database before runs and using component-specific system prompts under `symprompt/evolution/prompts/`. Translation evolution supports:
    - Phase 1 (Tier 1 focus) via `symprompt/evolution/eval_pipeline.py`
    - Phase 2 (full system with escalation and domain-weighted accuracy) via `symprompt/evolution/eval_pipeline_phase2.py` and the `--mode phase2` flag / `EVOLVE_MODE=phase2`.
  - Evolution seeding helpers in `symprompt/evolution/seeds.py` and `scripts/extract_translation_seeds.py` to export top programs from `evolution/openevolve_output/evolution.db` into `evolution/seeds/` for reuse.
  - Evolution prompt inspection helper in `scripts/inspect_evolution_prompt.py` that reconstructs the LLM user prompt (including the `## Last Execution Output` section) for selected programs based on stored artifacts.
  - CLI entrypoint in `symprompt/integration/cli.py` using LiteLLM (`openrouter/x-ai/grok-4.1-fast:free` by default) to show tier/profile, SymIL level, solver status, and optional natural-language explanations (`--show-symil`, `--explain`), plus a verification mode using `symprompt/integration/verification.py`.
  - Scallop backend (`symprompt/reasoning/scallop_runner.py`) now uses the real `scallopy` Python binding when installed to classify simple Level 0 fact/query programs and small Level 1 Horn-style rule sets, while remaining optional and returning `UNKNOWN` when the binding is absent; VSA backend (`symprompt/reasoning/vsa_encoder.py`, `vsa_runner.py`) encodes SymIL facts into a high-dimensional memory vector and provides a similarity-based soft status for Level 0 atomic queries that the Tier 2 portfolio can use as a conservative tie-breaker.
- v2 development is tracked in `docs/SymPrompt_dev_plan_v2.md` under the "Execution Status" section.

### Architecture Review & Fixes (2025-11-30)
- Architecture review completed; findings documented in `docs/architecture_review_2025-11-30.md`.
- **Fixes applied:**
  - `LogicalTranslator` now parses `not` and `or` JSON keys (previously missing, causing parse failures for full FOL).
  - Scallop rule syntax corrected from `=` to `:-` for proper Datalog compatibility.
  - Profile tests added in `tests/test_profiles.py` (13 tests covering all profiles).
  - Hardcoded values consolidated into `symprompt/config.py`:
    - `VSAConfig`: dimension, valid/invalid/tiebreaker thresholds
    - `SolverConfig`: Z3 timeout
    - `EvaluationConfig`: latency targets, domain weights
    - `RefinementConfig`: max attempts
    - `FeatureConfig`: complexity scaling factors
  - VSA thresholds now consistent across `vsa_runner.py` and `portfolio.py`.
- **Test status:** 36 tests passing.
