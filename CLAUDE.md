# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SymPrompt is a two-tier neuro-symbolic framework for translating natural language prompts into formal symbolic representations (SymIL) and verifying them with solvers. The system routes prompts through either a fast path (Tier 1) for simple reasoning or a full pipeline (Tier 2) for complex problems.

## Build & Development Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]

# Run tests
pytest

# Run single test file
pytest tests/test_router.py

# Run single test
pytest tests/test_router.py::test_router_math_prompt_uses_math_profile -v

# Run CLI
python -m symprompt.integration.cli "All mammals are animals. All cats are mammals. Therefore, all cats are animals."
python -m symprompt.integration.cli --show-symil --explain "prompt text"
python -m symprompt.integration.cli --mode verify --answer "candidate answer" "question"

# Formatting/linting
black symprompt tests
ruff symprompt tests
```

## Architecture

### Two-Tier Pipeline
- **Tier 0 (BYPASS)**: Pure LLM for trivial non-logical prompts
- **Tier 1 (Fast Path)**: Single solver (Z3 or Scallop), SymIL L0/L1, <50ms target
- **Tier 2 (Full SymPrompt)**: Multi-solver portfolio, SymIL L0→L2 escalation, <500ms target

### SymIL Levels (Progressive Complexity)
- **L0**: Facts + query only (fact checking, simple Q&A)
- **L1**: Facts + query + Horn clauses (syllogisms, implications)
- **L2**: Full ontology + rules + constraints + nested quantifiers (planning, abduction)

### Package Structure
- `symprompt/symil/`: SymIL datamodel (`model.py`), validator, profiles, examples
- `symprompt/translation/`: NL → SymIL pipeline (preprocessor, ontology extraction, logical translation)
- `symprompt/reasoning/`: Solver backends (Z3, Clingo/ASP, Scallop, VSA) and portfolio runner
- `symprompt/router/`: SmartRouter with feature extraction and tier/profile selection
- `symprompt/evolution/`: OpenEvolve integration, fitness evaluation, evolution runners
- `symprompt/integration/`: CLI, verification mode, router adapter with escalation
- `symprompt/llm/`: LiteLLM client for LLM calls

### Key Data Flow
1. `SmartRouter.route()` → extracts features → selects tier, SymIL level, profile, solver
2. `TranslationPipeline.translate()` → preprocessor → ontology extraction → logical translation → validator
3. `run_portfolio()` or single solver → Z3/Clingo/Scallop/VSA → result status (VALID/NOT_VALID/UNKNOWN)

### Domain Profiles
Profiles in `symprompt/symil/profiles.py` configure translation hints and preferred solvers:
- `syllogism`: Scallop, L1, categorical statements
- `math`: Z3, L0-L1, arithmetic
- `planning`: Clingo, L2, actions/preconditions/effects
- `legal`: Clingo, L2, deontic modalities
- `uncertain`: Scallop, L1-L2, probabilistic facts

## Testing

Tests are in `tests/` using pytest. Key test files:
- `test_router.py`: SmartRouter tier/profile selection
- `test_translation_pipeline.py`: End-to-end NL → SymIL translation
- `test_reasoning_backend.py`: Solver correctness
- `test_symil_levels.py`: SymIL level validation
- `test_profiles.py`: Domain profile configuration and registry

Benchmarks in `benchmarks/`: `tiny_folio.json`, `v2_syllogism.json`, `v2_math.json`, `v2_planning.json`, `v2_legal.json`

## OpenEvolve Integration

Evolution targets three components:
1. **TranslationPipeline**: Prompt templates, parsing strategies (`run_translation_evolution.py`)
2. **SmartRouter**: Feature thresholds, tier selection (`run_router_evolution.py`)
3. **SymILProfiles**: Domain vocabularies, solver preferences (`run_profile_evolution.py`)

### Phase 1 Fitness (current - Tier 1 focus)
Fitness function in `symprompt/evolution/eval_pipeline.py` combines:
- Accuracy (60%) - Tier 1 accuracy on syllogism benchmarks
- Latency score (15%) - Bonus for <50ms P95 latency
- Routing score (15%) - Correct tier selection for each benchmark
- Syntactic validity (10%) - SymIL parses without errors

### Phase 2+ Fitness (future - full system)
- Tier-weighted accuracy (50%)
- Latency score (25%)
- Routing score (15%)
- Syntactic validity (10%)

Config: `openevolve_config.yaml`

## Configuration

All configurable values are centralized in `symprompt/config.py`:

- **LLMConfig**: Model name, temperature, max_tokens, timeout, retries
- **VSAConfig**: Vector dimension, similarity thresholds (valid, invalid, tiebreaker)
- **SolverConfig**: Z3 timeout
- **EvaluationConfig**: Latency targets (Tier 1: 50ms, Tier 2: 500ms), domain weights
- **RefinementConfig**: Max verification/refinement attempts
- **FeatureConfig**: Complexity calculation parameters

Default model: `openrouter/x-ai/grok-4.1-fast:free`

LLM client interface expects a `complete(prompt: str) -> str` method. The `LiteLLMLLM` class in `symprompt/llm/litellm_client.py` implements async LLM calls via LiteLLM; `SyncLLMClient` in `symprompt/llm/sync_client.py` wraps it for synchronous use.
