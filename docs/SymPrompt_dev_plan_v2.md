Here’s a concrete development plan to build **SymPrompt v2.0** as a two‑tier, evolvable neuro‑symbolic framework, starting from the v1 plan and upgrading toward the v2 architecture.

Assumptions:

* Single developer / small setup on a local machine
* Python 3.11+
* OpenEvolve + LiteLLM available as Python libraries
* You are starting from SymPrompt‑Lite (v1 dev plan) and incrementally extending it

---

### Execution Status (last updated: 2025-11-30)

- Phase 1 – v2 Skeleton & Env: **complete** (core `symprompt/` package, SymIL core, LiteLLM client, baseline translation and Z3 reasoning paths, router scaffold; virtualenv configured with `pytest`, `litellm`, `z3-solver`, `clingo`, `openevolve`, and optional `scallopy`; test suite passing).
- Phase 2 – SymIL Levels & Profiles: **complete** (datamodel, level-aware validator, examples, profiles in `symprompt/symil/profiles.py` with domain-specific translation hints; `tests/test_profiles.py` added).
- Phase 3 – Reasoning Backends & Portfolio: **complete** (Z3 backend implemented; ASP/Scallop/VSA compilers in place; Clingo-backed ASP classification implemented for atomic queries; Scallop backend uses the real `scallopy` Python binding when available, with correct `:-` rule syntax; VSA backend encodes SymIL facts and uses configurable thresholds from `config.py`; the Tier 2 portfolio aggregates decisions with consistent VSA tiebreaker thresholds).
- Phase 4 – Translation v2: **complete** (baseline TranslationPipeline with tests, profile-aware translation hints, validator-guided refinement; `LogicalTranslator` now parses `not` and `or` JSON keys in addition to existing formula types).
- Phase 5 – Smart Router: **complete** (heuristic router with feature extraction using configurable complexity factors from `config.py`; router adapter with solver-driven level escalation and Tier 0 BYPASS path).
- Phase 6 – OpenEvolve v2: **complete** (evaluation harness with configurable domain weights and latency targets from `config.py`; translation/router/profile evolution runners with DB backups and seeding).
- Phase 7 – CLI & Notebook: **partial** (CLI implemented with routing, SymIL display, verification mode with configurable max attempts; notebook demo not started).
- Phase 8 – Hardening & Docs: **in progress** (configuration consolidated into `symprompt/config.py`; architecture review completed in `docs/architecture_review_2025-11-30.md`; 36 tests passing).

---

## 0. Scope & Strategy (v2 vs v1)

**Goal:** Evolve SymPrompt‑Lite into the full v2 system without losing simplicity.

We will:

* Keep: SymIL as JSON IL, TranslationPipeline, Z3 backend, tiny benchmarks, OpenEvolve integration.
* Add (v2 features):
  * Two‑tier architecture (Tier 1 fast path, Tier 2 full pipeline).
  * Progressive SymIL levels (L0, L1, L2).
  * SymIL Profiles (syllogism, math, planning, legal, uncertain).
  * Neuro‑symbolic backends (Scallop, VSA) alongside Z3 and Clingo.
  * Evolvable Router and evolvable SymIL Profiles via OpenEvolve.
* Adopt best practices from `/Users/webik/projects/evolve`:
  * Route all LLM calls through LiteLLM.
  * Make backups of the evolution database / program archive before each evolution run.
  * Use top programs from previous runs as seeds for the next evolution.
  * Keep evaluators robust, fast, and well‑logged.

Phases are cumulative: each phase leaves the system in a working state.

---

## 1. Project Skeleton & Environment (v2‑ready)

**Goal:** Extend the v1 skeleton to accommodate tiers, profiles, and new backends.

**Tasks:**

1. Confirm / create core layout (building on v1 plan):

   ```text
   symprompt/
     symprompt/
       __init__.py
       config.py
       llm/
         __init__.py
         litellm_client.py     # LiteLLM wrapper (similar to evolve/evolution/litellm_client.py)
       symil/
         __init__.py
         model.py              # SymIL dataclasses, levels L0/L1/L2
         validator.py
         profiles.py           # SymILProfile definitions + registry
         examples.py
       translation/
         __init__.py
         pipeline.py           # TranslationPipeline (Tier 1/2 aware)
         preprocess.py
         ontology.py
         logical.py
         constraints.py
       reasoning/
         __init__.py
         fol_compiler.py       # SymIL -> FOL/SMT (Z3)
         asp_compiler.py       # SymIL -> ASP (Clingo)
         scallop_compiler.py   # SymIL -> Scallop
         vsa_encoder.py        # SymIL -> VSA state
         z3_runner.py
         clingo_runner.py
         scallop_runner.py
         vsa_runner.py
         portfolio.py          # multi‑solver strategy + consensus
       router/
         __init__.py
         smart_router.py       # Tier selection + profile choice (evolvable)
         features.py           # prompt features for routing
       evolution/
         __init__.py
         eval_pipeline.py      # fitness eval for translation + routing + profiles
         seeds.py              # seed configurations (top programs from previous runs)
         run_translation_evolution.py
         run_router_evolution.py
         run_profile_evolution.py
         backups.py            # evolution DB backup helpers
       integration/
         __init__.py
         router_adapter.py     # Glue SmartRouter to external LLM calls
         cli.py                # CLI entrypoint (Tier 1/Tier 2 routing, verify + refine modes)
         notebook_demo.ipynb
     tests/
       test_symil_levels.py
       test_profiles.py
       test_translation_tier1.py
       test_translation_tier2.py
       test_router.py
       test_reasoning_backends.py
       test_evolution_eval.py
     benchmarks/
       tiny_folio.json
       v2_syllogism.json
       v2_math.json
       v2_planning.json
       v2_legal.json
     openevolve_config.yaml
     pyproject.toml
     README.md
   ```

2. Environment:

   * Add dependencies:
     * `litellm`, `openevolve`, `z3-solver`, `clingo` (or bindings), `scallop` (or wrapper), VSA implementation (custom or third‑party), `pytest`, `pydantic` or `dataclasses`.
   * Configure LiteLLM via environment variables (API keys) and a small config section in `config.py`, following the pattern in `evolve/evolution/litellm_client.py`.
   * Ensure `.env` / `config.py` are excluded from version control and all LLM access in this repo goes through `symprompt.llm.litellm_client.LiteLLMLLM`.

3. Security & privacy:

   * No telemetry or remote logging by default; logs are local files.
   * All benchmarks and evolution data live under `symprompt/` or `data/` (if added later).

---

## 2. SymIL Core v2: Levels & Profiles

**Goal:** Extend SymIL to support L0/L1/L2 and SymIL Profiles as per v2 architecture.

### 2.1 SymIL Levels (L0, L1, L2)

**Reference implementation sketch (`symprompt/symil/model.py`):**

Start from the v1 SymIL datamodel and extend it with a `level` field and L0/L1/L2 semantics:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class Predicate:
    name: str
    arity: int
    types: List[str]          # e.g. ["entity"]
    description: Optional[str] = None


@dataclass
class Ontology:
    predicates: List[Predicate] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)

    def get_predicate(self, name: str) -> Optional[Predicate]:
        for p in self.predicates:
            if p.name == name:
                return p
        return None


@dataclass
class Atom:
    pred: str
    args: List[str]  # variable or constant names


@dataclass
class Not:
    formula: "Formula"


@dataclass
class And:
    formulas: List["Formula"]


@dataclass
class Or:
    formulas: List["Formula"]


@dataclass
class Implies:
    premise: "Formula"
    conclusion: "Formula"


@dataclass
class ForAll:
    var: str
    type: str
    body: "Formula"


@dataclass
class Exists:
    var: str
    type: str
    body: "Formula"


Formula = Atom | Not | And | Or | Implies | ForAll | Exists


@dataclass
class Rule:
    # For L1 you can restrict body to Horn clauses; for L2 allow full Formula.
    forall: str
    type: str
    body: Formula


@dataclass
class Query:
    # For v0/v1 we only support "prove" queries, but this can be extended.
    prove: Formula


@dataclass
class SymIL:
    ontology: Ontology
    facts: List[Formula] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    query: Optional[Query] = None
    constraints: List[Formula] = field(default_factory=list)
    level: int = 0  # 0=L0, 1=L1, 2=L2
```

**Level semantics:**

- L0: use `SymIL` with `level=0`, only `facts` (Atoms) and an `Atom` or boolean `query`.
- L1: `level=1`, allow Horn‑style `Rule` instances (`forall`, `if`/`then` encoded as `Implies`).
- L2: `level=2`, full FOL (`Not`, `And`, `Or`, nested quantifiers, constraints, ontology).

### 2.1.1 SymIL Validator Example (`symprompt/symil/validator.py`)

```python
class SymILValidationError(Exception):
    pass


class SymILValidator:
    def validate(self, symil: SymIL) -> SymIL:
        self._check_predicates(symil)
        # You can extend this with free-variable and level-specific checks.
        return symil

    def _check_predicates(self, symil: SymIL) -> None:
        def check_formula(f: Formula):
            if isinstance(f, Atom):
                pred = symil.ontology.get_predicate(f.pred)
                if pred is None:
                    raise SymILValidationError(f"Unknown predicate: {f.pred}")
                if len(f.args) != pred.arity:
                    raise SymILValidationError(
                        f"Arity mismatch for {f.pred}: "
                        f"expected {pred.arity}, got {len(f.args)}"
                    )
            elif isinstance(f, Not):
                check_formula(f.formula)
            elif isinstance(f, (And, Or)):
                for sub in f.formulas:
                    check_formula(sub)
            elif isinstance(f, Implies):
                check_formula(f.premise)
                check_formula(f.conclusion)
            elif isinstance(f, (ForAll, Exists)):
                check_formula(f.body)

        for fact in symil.facts:
            check_formula(fact)
        for rule in symil.rules:
            check_formula(rule.body)
        if symil.query is not None:
            check_formula(symil.query.prove)
```

Build on this to add level‑aware rules (e.g., disallow `Exists` at L1).

### 2.1.2 Examples (`symprompt/symil/examples.py`)

Use the mammals/cats/animals example from the original dev plan:

```python
MAMMALS_SYMIL_L1 = {
    "level": 1,
    "ontology": {
        "predicates": [
            {"name": "mammal", "arity": 1, "types": ["entity"], "description": "X is a mammal"},
            {"name": "animal", "arity": 1, "types": ["entity"], "description": "X is an animal"},
            {"name": "cat",    "arity": 1, "types": ["entity"], "description": "X is a cat"},
        ],
        "functions": [],
        "constants": [],
    },
    "facts": [],
    "rules": [
        {
            "forall": "X",
            "type": "entity",
            "body": {
                "implies": [
                    {"pred": "mammal", "args": ["X"]},
                    {"pred": "animal", "args": ["X"]},
                ]
            },
        },
        {
            "forall": "X",
            "type": "entity",
            "body": {
                "implies": [
                    {"pred": "cat", "args": ["X"]},
                    {"pred": "mammal", "args": ["X"]},
                ]
            },
        },
    ],
    "query": {
        "prove": {
            "forall": "X",
            "type": "entity",
            "body": {
                "implies": [
                    {"pred": "cat", "args": ["X"]},
                    {"pred": "animal", "args": ["X"]},
                ]
            },
        }
    },
    "constraints": [],
}
```

You can either keep these as raw JSON fixtures or convert them into `SymIL` instances in code.

### 2.2 SymIL Profiles

**Tasks:**

1. Implement `symprompt/symil/profiles.py`:

   * Define `SymILProfile` dataclass matching v2 architecture (`name`, `predicate_vocabulary`, `allowed_constructs`, `preferred_solver`, `default_level`, `translation_hints`).
   * Hard‑code an initial set of profiles:
     * `syllogism`, `math`, `planning`, `legal`, `uncertain`, aligned with v2 table.
   * Provide a registry and simple helpers:
     * `get_profile(name: str)`, `list_profiles()`.

2. Tests:

   * `tests/test_profiles.py`:
     * Ensure profiles load correctly and use valid SymIL constructs.

---

## 3. Reasoning Layer v2: Multi‑Backend & Portfolio

**Goal:** Extend the v1 Z3‑only reasoning to the v2 portfolio (Z3, Clingo, Scallop, VSA) with a simple portfolio strategy.

### 3.1 Backends

**Tasks:**

1. Keep and refine v1 Z3 path:

   * Move the v1 `symprompt/core/compiler_z3.py` logic into `symprompt/reasoning/fol_compiler.py` + `z3_runner.py`.
   * Ensure it supports L0/L1/L2 SymIL.

   **Reference implementation sketch (`symprompt/reasoning/fol_compiler.py`):**

   ```python
   from __future__ import annotations
   from typing import Dict
   import z3

   from symprompt.symil.model import (
       SymIL, Formula, Atom, Not, And, Or, Implies, ForAll, Exists,
   )


   def symil_to_z3(symil: SymIL) -> dict:
       Entity = z3.DeclareSort("Entity")

       pred_fns: Dict[str, z3.FuncDeclRef] = {}
       for p in symil.ontology.predicates:
           arg_sorts = [Entity] * p.arity
           pred_fns[p.name] = z3.Function(p.name, *arg_sorts, z3.BoolSort())

       def encode_formula(f: Formula, env: Dict[str, z3.ExprRef]) -> z3.BoolRef:
           if isinstance(f, Atom):
               fn = pred_fns[f.pred]
               args = [env.get(a, z3.Const(a, Entity)) for a in f.args]
               for a in f.args:
                   if a not in env:
                       env[a] = z3.Const(a, Entity)
               return fn(*args)
           if isinstance(f, Not):
               return z3.Not(encode_formula(f.formula, env))
           if isinstance(f, And):
               return z3.And(*[encode_formula(sub, env) for sub in f.formulas])
           if isinstance(f, Or):
               return z3.Or(*[encode_formula(sub, env) for sub in f.formulas])
           if isinstance(f, Implies):
               return z3.Implies(
                   encode_formula(f.premise, env),
                   encode_formula(f.conclusion, env),
               )
           if isinstance(f, ForAll):
               var = z3.Const(f.var, Entity)
               new_env = env | {f.var: var}
               return z3.ForAll([var], encode_formula(f.body, new_env))
           if isinstance(f, Exists):
               var = z3.Const(f.var, Entity)
               new_env = env | {f.var: var}
               return z3.Exists([var], encode_formula(f.body, new_env))
           raise TypeError(f"Unknown formula type: {type(f)}")

       axioms: list[z3.BoolRef] = []
       for fact in symil.facts:
           axioms.append(encode_formula(fact, {}))

       for rule in symil.rules:
           var = z3.Const(rule.forall, Entity)
           body_ref = encode_formula(rule.body, {rule.forall: var})
           axioms.append(z3.ForAll([var], body_ref))

       query_ref = None
       if symil.query is not None:
           query_ref = encode_formula(symil.query.prove, {})

       return {
           "Entity": Entity,
           "predicates": pred_fns,
           "axioms": axioms,
           "query": query_ref,
       }


   def solve_symil(symil: SymIL, timeout_ms: int = 1000) -> dict:
       ctx = symil_to_z3(symil)
       solver = z3.Solver()
       solver.set("timeout", timeout_ms)

       for ax in ctx["axioms"]:
           solver.add(ax)

       if ctx["query"] is not None:
           solver.push()
           solver.add(z3.Not(ctx["query"]))
           sat_result = solver.check()
           solver.pop()
           if sat_result == z3.unsat:
               status = "VALID"
           elif sat_result == z3.sat:
               status = "NOT_VALID"
           else:
               status = "UNKNOWN"
       else:
           sat_result = solver.check()
           status = str(sat_result)

       return {"status": status}
   ```

2. Add ASP backend (`symprompt/reasoning/asp_compiler.py`, `clingo_runner.py`):

   * Implement a basic SymIL‑to‑ASP mapping for L1/L2 rules and constraints.
   * Provide `solve_asp(symil, profile)` returning result + diagnostics.

3. Add Scallop backend (`symprompt/reasoning/scallop_compiler.py`, `scallop_runner.py`):

   * Implement SymIL → Scallop program compilation similar to the pseudo‑code in v2 architecture (probabilistic facts, rules).

4. Add VSA backend (`symprompt/reasoning/vsa_encoder.py`, `vsa_runner.py`):

   * Implement basic SymIL → VSA coding as per v2 doc (binding predicates and arguments).

5. Implement a portfolio layer (`symprompt/reasoning/portfolio.py`):

   * Wrap backends to:
     * Run a single backend for Tier 1 (Z3 or Scallop depending on profile).
     * Run a portfolio for Tier 2 (Z3 + Clingo + Scallop + VSA).
     * Implement simple consensus voting and latency measurement.

6. Tests:

   * `tests/test_reasoning_backends.py`:
     * Use small SymIL examples to check correctness of each backend.
     * Ensure portfolio returns consistent answers when backends agree.

---

## 4. Translation Layer v2: Progressive Levels

**Goal:** Extend the v1 TranslationPipeline to support target SymIL levels (L0→L1→L2) and profiles.

**Tasks:**

1. Update `symprompt/translation/preprocess.py`:

   **Reference implementation:**

   ```python
   # symprompt/translation/preprocess.py
   from __future__ import annotations


   class Preprocessor:
       def normalize(self, text: str) -> str:
           # Minimal normalization for v1/v2
           return " ".join(text.strip().split())
   ```

2. Update `symprompt/translation/ontology.py` and `logical.py`:

   Start from the v1 baseline prompts and logic.

   **Ontology extractor prompt (v1 style):**

   ```python
   ONTOLOGY_SYSTEM_PROMPT = """
   You are an ontology designer for a logic reasoning system.
   Your job is to read natural language text and extract a minimal set
   of logical predicates and constants that will be used to formalize
   the text in first-order logic.

   Output ONLY a JSON object:
   {
     "predicates": [
       {"name": "mammal", "arity": 1, "types": ["entity"], "description": "X is a mammal"},
       {"name": "animal", "arity": 1, "types": ["entity"], "description": "X is an animal"}
     ],
     "functions": [],
     "constants": []
   }

   Rules:
   - Use lowercase predicate names without spaces.
   - Assume all predicates have arity 1 and type "entity" for now.
   - Do NOT include comments or extra keys.
   """
   ```

   **Logical translator prompt (v1 style):**

   ```python
   LOGICAL_SYSTEM_PROMPT = """
   You are a translator from natural language into a JSON-based logical
   intermediate language called SymIL.

   Given:
   - A short piece of text describing logical relationships.
   - An ontology listing predicates.

   You must output ONLY a JSON object with the structure:

   {
     "facts": [ ... ],
     "rules": [ ... ],
     "query": { ... }
   }

   Rules:
   - Use only predicates defined in the ontology.
   - Use variable names like "X", "Y" (type "entity").
   - Use "implies" for conditional statements ("all ... are ...").
   - Use "forall" for universal statements.
   - For Level 0, prefer only facts + query.
   - For Level 1, restrict rules to simple Horn clauses.
   - For Level 2, you may emit full SymIL (nested quantifiers, constraints).
   """
   ```

   **Ontology extractor implementation sketch:**

   ```python
   # symprompt/translation/ontology.py
   import json
   from symprompt.symil.model import Ontology, Predicate


   class OntologyExtractor:
       def __init__(self, llm_client):
           self.llm = llm_client

       def extract(self, text: str) -> Ontology:
           prompt = ONTOLOGY_SYSTEM_PROMPT + "\nText:\n" + text
           raw = self.llm.complete(prompt)
           data = json.loads(raw)

           preds = [
               Predicate(
                   name=p["name"],
                   arity=p.get("arity", 1),
                   types=p.get("types", ["entity"]),
                   description=p.get("description"),
               )
               for p in data.get("predicates", [])
           ]
           return Ontology(predicates=preds)
   ```

   **Logical translator implementation sketch (key idea):**

   ```python
   # symprompt/translation/logical.py
   import json
   from symprompt.symil.model import Ontology, SymIL, Atom, Rule, Query, Formula


   class LogicalTranslator:
       def __init__(self, llm_client):
           self.llm = llm_client

       def translate(self, text: str, ontology: Ontology, target_level: int) -> SymIL:
           ontology_json = json.dumps(
               {
                   "predicates": [
                       {
                           "name": p.name,
                           "arity": p.arity,
                           "types": p.types,
                           "description": p.description or "",
                       }
                       for p in ontology.predicates
                   ],
                   "functions": [],
                   "constants": [],
               }
           )
           prompt = LOGICAL_SYSTEM_PROMPT + f"\nOntology:\n{ontology_json}\nText:\n{text}"
           raw = self.llm.complete(prompt)
           data = json.loads(raw)

           # Implement _formula_from_json / _rule_from_json / _query_from_json
           facts = [self._formula_from_json(f) for f in data.get("facts", [])]
           rules = [self._rule_from_json(r) for r in data.get("rules", [])]
           query = self._query_from_json(data.get("query"))

           return SymIL(ontology=ontology, facts=facts, rules=rules, query=query, level=target_level)
   ```

3. Extend `symprompt/translation/pipeline.py`:

   **Reference implementation sketch (v1 → v2):**

   ```python
   # symprompt/translation/pipeline.py
   from dataclasses import dataclass
   from symprompt.symil.model import SymIL
   from symprompt.symil.validator import SymILValidator
   from symprompt.translation.preprocess import Preprocessor
   from symprompt.translation.ontology import OntologyExtractor
   from symprompt.translation.logical import LogicalTranslator


   @dataclass
   class TranslationPipeline:
       preprocessor: Preprocessor
       ontology_extractor: OntologyExtractor
       logical_translator: LogicalTranslator
       validator: SymILValidator

       @classmethod
       def from_llm_client(cls, llm_client) -> "TranslationPipeline":
           return cls(
               preprocessor=Preprocessor(),
               ontology_extractor=OntologyExtractor(llm_client),
               logical_translator=LogicalTranslator(llm_client),
               validator=SymILValidator(),
           )

       def translate(self, nl_prompt: str, level: int) -> SymIL:
           text = self.preprocessor.normalize(nl_prompt)
           ontology = self.ontology_extractor.extract(text)
           symil = self.logical_translator.translate(text, ontology, target_level=level)
           return self.validator.validate(symil)
   ```

   For progressive escalation, wrap this in a helper:

   ```python
   def translate_with_escalation(pipeline: TranslationPipeline, prompt: str, max_level: int = 2):
       from symprompt.reasoning.fol_compiler import solve_symil

       for level in range(max_level + 1):
           symil = pipeline.translate(prompt, level=level)
           result = solve_symil(symil)
           if result["status"] == "VALID":
               return symil, result
           # If backend reports needs-more-structure, escalate; you can encode this explicitly later.
       return symil, result
   ```

4. Tests:

   * `tests/test_translation_tier1.py`:
     * Use synthetic L0/L1 NL prompts (e.g., mammals/cats) and assert correct SymIL structure.
   * `tests/test_translation_tier2.py`:
     * Use more complex examples and ensure escalations work.

---

## 5. Smart Router v2: Two Tiers & Profiles

**Goal:** Implement the two‑tier routing logic and make it evolvable.

**Tasks:**

1. Implement routing features (`symprompt/router/features.py`):

   * Extract features from prompts: length, presence of logical keywords, numerics, constraints, domain hints (math/legal/planning).

2. Implement `SmartRouter` (`symprompt/router/smart_router.py`):

   * Start with a hand‑crafted version based on the v2 pseudo‑code:
     * BYPASS (pure LLM) for trivial prompts.
     * Tier 1 (L0/L1 + single solver) for simple reasoning.
     * Tier 2 (L0→L2 + portfolio) for complex reasoning.
   * Use `SymILProfile` selection based on features.
   * Encapsulate routing rules in a way that OpenEvolve can later mutate (e.g., thresholds in a small config or pure‑Python decision function).

3. Glue to integration layer (`symprompt/integration/router_adapter.py`):

   * Provide a function:
     * `route_and_solve(prompt, llm_client)` that:
       * Runs `SmartRouter` to choose tier/profile.
       * Calls `TranslationPipeline` + reasoning portfolio when symbolic path is chosen.
       * Calls LLM directly (via LiteLLM) when bypass is chosen.

4. Tests:

   * `tests/test_router.py`:
     * Use synthetic prompts to exercise BYPASS, Tier 1, and Tier 2 paths.

---

## 6. OpenEvolve Integration v2: Translation, Router, Profiles

**Goal:** Upgrade the v1 “translation‑only” evolution into the full v2 multi‑target evolution, using best practices from `/Users/webik/projects/evolve`.

### 6.1 LiteLLM Integration for OpenEvolve

**Tasks:**

1. Implement `symprompt/llm/litellm_client.py`:

   * Adapt `LiteLLMLLM` from `evolve/evolution/litellm_client.py` to implement OpenEvolve’s `LLMInterface`.
   * Support:
     * Model name, temperature, top_p, max_tokens, timeout, retries, seed.
     * Markdown code fence stripping where relevant (for code evolution tasks).

2. Update `openevolve_config.yaml`:

   * Configure LiteLLM models as the primary LLM backend.
   * Keep a simple single‑model or two‑model ensemble configuration.

### 6.2 Evaluation & Benchmarks

**Tasks:**

1. Implement `symprompt/evolution/eval_pipeline.py`:

   * Evaluate:
     * Tier 1 accuracy, coverage, latency.
     * Tier 2 accuracy, coverage, latency.
     * Syntactic validity of SymIL across levels.
     * Routing quality (whether router sends prompts to appropriate tier/profile).
   * Follow the v2 fitness function structure:
     * Combine tier‑weighted accuracy, latency score, routing score, syntactic validity.

2. Benchmarks (`benchmarks/`):

   * Reuse/extend v1 `tiny_folio.json`.
   * Add v2 benchmark slices:
     * `v2_syllogism.json`, `v2_math.json`, `v2_planning.json`, `v2_legal.json`.
   * Include a small set of “wild” prompts (held‑out) for anti‑overfitting.

3. Tests:

   * `tests/test_evolution_eval.py` with tiny fixtures, confirming metrics and fitness combine correctly.

### 6.2.1 Example Evaluation Harness (`symprompt/evolution/eval_pipeline.py`)

The concrete v2 evaluation harness lives in `symprompt/evolution/eval_pipeline.py`:

- `evaluate_system(router, pipeline, benchmarks)`:
  - Routes each prompt through `SmartRouter`, performs SymIL level escalation with refinement, and calls the multi-backend portfolio (`run_portfolio`).
  - Returns tiered accuracy/coverage/latency, syntactic validity, and a routing score.
- `evaluate(program_path)`:
  - Dynamically loads a candidate `TranslationPipeline` from a Python file.
  - Runs it on multi-domain benchmarks and computes:
    - Global metrics and tier-weighted accuracy (matching the v2 architecture).
    - Per-domain metrics (`<domain>_accuracy`, `<domain>_coverage`) and a domain-weighted `weighted_accuracy` for analysis and feature dimensions.
    - A scalar `combined_score` that follows the v2 multi-objective fitness:
      - `combined_score = 0.50 * tier_weighted_accuracy + 0.25 * latency_score + 0.15 * routing_score + 0.10 * syntactic_validity`
    - A `latency_score` derived from Tier 1 and Tier 2 p95 latencies with targets of 50ms and 500ms, respectively.
    - `complexity` as the candidate’s code length, used as a mild penalty for overly large programs.

A tiny harness script in `scripts/run_eval_tiny.py` exercises this end to end:

```python
from symprompt.evolution.eval_pipeline import evaluate_system
from symprompt.router.smart_router import SmartRouter
from symprompt.translation.pipeline import TranslationPipeline
from symprompt.llm.sync_client import build_default_sync_client
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
benchmarks = json.loads((root / "benchmarks" / "tiny_folio.json").read_text())

router = SmartRouter()
llm_client = build_default_sync_client()
pipeline = TranslationPipeline.from_llm_client(llm_client)

result = evaluate_system(router, pipeline, benchmarks)
print("Tier1 accuracy:", result.tier1_accuracy)
print("Tier2 accuracy:", result.tier2_accuracy)
print("Syntactic validity:", result.syntactic_validity)
print("Routing score:", result.routing_score)
```

This is the canonical v2 evaluation harness example for developers and for OpenEvolve integration.

### 6.3 Evolution Scripts & Best Practices

**Tasks:**

1. Implement evolution entrypoints:

   * `symprompt/evolution/run_translation_evolution.py`:
     * Evolves `OntologyExtractor` / `LogicalTranslator` logic and prompts.
   * `symprompt/evolution/run_router_evolution.py`:
     * Evolves router thresholds / decision logic.
   * `symprompt/evolution/run_profile_evolution.py`:
     * Evolves profile definitions (vocabulary, constructs, solver preferences).

2. Implement evolution DB backups (`symprompt/evolution/backups.py`):

   * Before each evolution run:
     * Create a timestamped backup of the OpenEvolve program database / archive (mirroring `scripts/backup_state.py` ideas from `/Users/webik/projects/evolve`).
   * Optionally keep a rotating set of backups.

3. Seeding:

   * Implement `symprompt/evolution/seeds.py`:
     * Load top programs (e.g. by fitness) from previous evolution runs.
     * Provide seed sets for the next run (similar to `strategies/seeds` + `seed_archive.py` in `evolve`).

4. Anti‑overfitting:

   * Follow v2 anti‑overfitting measures:
     * Held‑out wild prompts.
     * Synthetic noise / paraphrases.
     * Cross‑benchmark validation.
     * Complexity penalties for overly complex translators.

---

## 7. Integration & CLI: Two‑Tier User Experience

**Goal:** Provide a CLI and simple API that showcase the two‑tier system end‑to‑end.

**Tasks:**

1. CLI (`symprompt/integration/cli.py`):

   * `symprompt` command that:
     * Accepts a prompt.
     * Runs `route_and_solve(...)`.
     * Prints:
       * Tier used (BYPASS, Tier 1, or Tier 2).
       * SymIL (optional flag, e.g. `--show-symil`).
       * Solver result and natural‑language explanation via LLM.

2. Notebook demo (`symprompt/integration/notebook_demo.ipynb`):

   * Walk through:
     * Router decision visualization.
     * SymIL at each level.
     * Reasoning through multiple backends.
     * Mode 1/2/3 examples using the CLI (`query`, `verify`, `refine`).

---

## 8. Hardening, Metrics & Documentation

**Goal:** Make the v2 system robust and understandable, similar in spirit to the evolve project’s maturity.

**Tasks:**

1. Observability:

   * Add lightweight logging for:
     * Routing decisions (prompt, tier, profile).
     * SymIL validation failures.
     * Backend latency per query.
     * Evolution runs (start/end, best fitness, seeds used).

2. Performance:

   * Measure and optimize:
     * Tier 1 P95 latency (< 50ms target where feasible).
     * Tier 2 P95 latency (< 500ms target).

3. Docs:

   * Add / update:
     * `docs/SymPrompt_Architecture_v2.md` (already present; treat as canonical architecture).
     * `docs/SymPrompt_dev_plan_v2.md` (this file) as the implementation roadmap.
     * Short runbooks for:
       * Running CLI.
       * Running evolution.
       * Debugging SymIL or routing issues.

4. Testing:

   * Ensure `pytest` suite covers:
     * SymIL levels and profiles.
     * Translation and routing.
     * Reasoning backends and portfolio.
     * Evolution evaluation logic.

---

## 9. Phase Summary (Execution Order)

1. **Phase 1 – v2 Skeleton & Env:** create/extend project layout, LiteLLM integration, basic config.
2. **Phase 2 – SymIL Levels & Profiles:** implement L0/L1/L2 and profiles with validators and tests.
3. **Phase 3 – Reasoning Backends & Portfolio:** extend Z3 path, add Clingo/Scallop/VSA, portfolio runner.
4. **Phase 4 – Translation v2:** add level‑aware translation, escalation, profile awareness.
5. **Phase 5 – Smart Router:** implement two‑tier routing + profile selection and glue logic.
6. **Phase 6 – OpenEvolve v2:** wire up evaluation, evolution scripts, backups, and seeding.
7. **Phase 7 – CLI & Notebook:** user‑facing tooling for experiments and demos.
8. **Phase 8 – Hardening & Docs:** logging, latency tuning, tests, and documentation polish.

Each phase should end with passing tests and a small set of real prompts exercised through the full pipeline (where applicable), keeping SymPrompt v2 usable throughout its evolution.
