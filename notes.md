Yes, that's the plan. The evolution strategy is:

  1. Current phase: Optimize for 77% → 100% on the 40 benchmarks (syllogisms, basic logic)
  2. When we hit ~95-100%: Add harder benchmarks:
    - Nested quantifiers (∀x∃y...)
    - Modal logic (necessity, possibility)
    - Abductive reasoning (inference to best explanation)
    - Multi-step planning problems
    - Counterfactuals
  3. Multi-objective: Even at 100% accuracy, evolution can still optimize:
    - Latency (15% of score) - faster translations
    - Syntactic validity (25%) - fewer parse errors
    - Code complexity - simpler prompts that generalize better
  4. Generalization: We should also test evolved prompts on held-out benchmarks to ensure they don't just overfit to the training set.

  The benchmark files in benchmarks/ can be extended progressively. The v2_*.json files (planning, legal, math) have more complex patterns waiting to be integrated.


scripts/
  ├── run_translation_evolution.py  # TranslationPipeline (NL -> SymIL)
  ├── run_router_evolution.py       # SmartRouter (tier/profile selection)
  └── run_profile_evolution.py      # SymIL profiles (domain vocabularies)

  Usage:
  # Translation evolution (100 iterations)
  EVOLVE_ITERATIONS=100 .venv/bin/python scripts/run_translation_evolution.py

  # Router evolution (50 iterations, custom model)
  EVOLVE_ITERATIONS=50 EVOLVE_LLM_MODEL="openrouter/..." .venv/bin/python scripts/run_router_evolution.py

  # Profile evolution (resume from checkpoint)
  EVOLVE_ITERATIONS=100 EVOLVE_RESUME=1 .venv/bin/python scripts/run_profile_evolution.py

  All scripts support the same environment variables:
  - EVOLVE_LLM_MODEL - LLM model to use
  - EVOLVE_ITERATIONS - Number of iterations
  - EVOLVE_RESUME - Resume from checkpoint ("1" to enable)
  - EVAL_PARALLEL_BENCHMARKS - Parallel benchmark workers
  - EVAL_SHOW_PROGRESS - Show progress ("0" to disable)
