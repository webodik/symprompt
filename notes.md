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
