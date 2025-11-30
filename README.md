# SymPrompt

A two-tier neuro-symbolic framework for translating natural language prompts into formal symbolic representations (SymIL) and verifying them with solvers.

## Overview

SymPrompt routes prompts through either a fast path (Tier 1) for simple reasoning or a full pipeline (Tier 2) for complex problems:

- **Tier 0 (BYPASS)**: Pure LLM for trivial non-logical prompts
- **Tier 1 (Fast Path)**: Single solver (Z3 or Scallop), SymIL L0/L1, <50ms target
- **Tier 2 (Full Pipeline)**: Multi-solver portfolio, SymIL L0→L2 escalation, <500ms target

### SymIL Levels

- **L0**: Facts + query only (fact checking, simple Q&A)
- **L1**: Facts + query + Horn clauses (syllogisms, implications)
- **L2**: Full ontology + rules + constraints + nested quantifiers (planning, abduction)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/symprompt.git
cd symprompt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .[dev]
```

### Requirements

- Python 3.11+
- Z3 solver (`z3-solver`)
- Optional: Clingo (`clingo`), Scallop (`scallopy`)

## Quick Start

### Command Line Interface

```bash
# Basic usage - analyze a logical statement
python -m symprompt.integration.cli "All mammals are animals. All cats are mammals. Therefore, all cats are animals."

# Show the generated SymIL representation
python -m symprompt.integration.cli --show-symil "All mammals are animals. All cats are mammals. Therefore, all cats are animals."

# Get a natural language explanation
python -m symprompt.integration.cli --explain "All mammals are animals. All cats are mammals. Therefore, all cats are animals."

# Verification mode - verify an answer against a question
python -m symprompt.integration.cli --mode verify --answer "Yes, cats are animals" "Are cats animals given that all mammals are animals and all cats are mammals?"
```

### Python API

```python
from symprompt.llm.sync_client import build_default_sync_client
from symprompt.integration.router_adapter import route_and_solve

# Create LLM client
llm_client = build_default_sync_client()

# Route and solve a prompt
result = route_and_solve(
    "All mammals are animals. All cats are mammals. Therefore, all cats are animals.",
    llm_client
)

print(f"Tier: {result.routing.tier}")
print(f"Profile: {result.routing.profile_name}")
print(f"Status: {result.solver_result['status']}")  # VALID, NOT_VALID, or UNKNOWN
```

### Verification Mode

```python
from symprompt.llm.sync_client import build_default_sync_client
from symprompt.integration.verification import verify_answer, verify_and_refine_answer

llm_client = build_default_sync_client()

# Simple verification
result = verify_answer(
    question="Are cats animals?",
    answer="Yes, because all cats are mammals and all mammals are animals.",
    llm_client=llm_client
)
print(f"Status: {result.status}")

# Verification with refinement loop
result = verify_and_refine_answer(
    question="Are cats animals?",
    initial_answer="Maybe",
    llm_client=llm_client,
    max_attempts=3
)
print(f"Final answer: {result.answer}")
print(f"Status: {result.status}")
```

## Configuration

All configurable values are in `symprompt/config.py`:

```python
from symprompt.config import (
    DEFAULT_LLM_CONFIG,      # LLM model, temperature, timeout
    DEFAULT_VSA_CONFIG,      # Vector dimension, similarity thresholds
    DEFAULT_SOLVER_CONFIG,   # Z3 timeout
    DEFAULT_EVALUATION_CONFIG,  # Latency targets, domain weights
)
```

### Environment Variables

Set `SYMPROMPT_LLM_MODEL` to override the default LLM model:

```bash
export SYMPROMPT_LLM_MODEL="openrouter/anthropic/claude-3-haiku"
```

## Domain Profiles

SymPrompt supports multiple reasoning domains with optimized configurations:

| Profile | Preferred Solver | Default Level | Use Case |
|---------|-----------------|---------------|----------|
| `syllogism` | Scallop | L1 | Categorical statements |
| `math` | Z3 | L0 | Arithmetic reasoning |
| `planning` | Clingo | L2 | Actions, goals, effects |
| `legal` | Clingo | L2 | Deontic modalities |
| `uncertain` | Scallop | L1 | Probabilistic facts |

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_router.py

# Run with verbose output
pytest -v

# Run single test
pytest tests/test_router.py::test_router_math_prompt_uses_math_profile -v
```

## Project Structure

```
symprompt/
├── symil/          # SymIL datamodel, validator, profiles
├── translation/    # NL → SymIL pipeline
├── reasoning/      # Solver backends (Z3, Clingo, Scallop, VSA)
├── router/         # Smart router with feature extraction
├── evolution/      # OpenEvolve integration
├── integration/    # CLI, verification, router adapter
└── llm/            # LiteLLM client
```

## OpenEvolve Integration

SymPrompt supports evolutionary optimization of three components:

```bash
# Evolve the translation pipeline
python -m symprompt.evolution.run_translation_evolution

# Evolve the router
python -m symprompt.evolution.run_router_evolution

# Evolve domain profiles
python -m symprompt.evolution.run_profile_evolution
```

## Documentation

- [Architecture Overview](docs/SymPrompt_Architecture_v2.md)
- [Development Plan](docs/SymPrompt_dev_plan_v2.md)
- [Architecture Review](docs/architecture_review_2025-11-30.md)

## License

MIT
