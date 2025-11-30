Here’s a concrete development plan to build **SymPrompt** as a **simple, clean, private, personal tool** rather than a heavy research platform.

I’ll assume:

* Single developer / small setup
* Local machine (laptop / workstation)
* Python 3.11+
* You’re okay using a single main solver at first (Z3), and adding others later if you want.

I’ll structure it in phases you can actually implement.

---

## 0. Scope & Simplifications (what we’ll build first)

Starting from the v1.0 architecture doc, we’ll deliberately **shrink** the initial scope: 

**Keep (v1.0 features):**

* SymIL as the intermediate JSON language (but we’ll use a “Level 0/1” subset)
* A TranslationPipeline with:

  * `Preprocessor`
  * `OntologyExtractor` (evolvable)
  * `LogicalTranslator` (evolvable)
  * `ConstraintMiner` (optional, later)
  * `SymILValidator`
* One main backend: **FOL/SMT via Z3** (drop Prolog/ASP/Datalog for now) 
* OpenEvolve integration to evolve the translation pipeline
* Mode 1: **Pre-processing** integration with your LLM (query translation → solver → explanation) 

**Defer / omit for now:**

* Prolog, ASP, Scallop backends
* Multi-solver portfolio, consensus voting, fallback cascade 
* Post-processing & hybrid modes (we can add later) 
* Full benchmark suite; start with a small curated subset (e.g., a dozen FOLIO/MALLS-style examples)

That keeps the first version **lightweight and private** but still very close in spirit to the spec.

---

## 1. Project Skeleton & Environment

**Goal:** Clean repo layout, reproducible environment.

**Tasks:**

1. **Create repo structure**

   ```text
   symprompt/
     symprompt/
       __init__.py
       config.py
       symil/
         __init__.py
         model.py          # dataclasses for SymIL
         validator.py
         examples.py
       translation/
         __init__.py
         pipeline.py       # TranslationPipeline
         preprocess.py
         ontology.py
         logical.py
         constraints.py
       reasoning/
         __init__.py
         fol_compiler.py   # SymIL -> FOL/SMT
         z3_runner.py
       evolution/
         __init__.py
         eval_pipeline.py  # fitness eval for OpenEvolve
       integration/
         __init__.py
         router.py
         cli.py            # simple CLI
         notebook_demo.ipynb
     tests/
       test_symil.py
       test_translation.py
       test_reasoning.py
     benchmarks/
       tiny_folio.json
     openevolve_config.yaml
     pyproject.toml / requirements.txt
     README.md
   ```

2. **Environment**

   * Python 3.11+
   * Dependencies:

     * `z3-solver`
     * `openai` or your chosen LLM client
     * `pydantic` (or `dataclasses` + `marshmallow`) for SymIL model
     * `pytest` for tests
     * `openevolve` (however you install it)
   * Optionally: `black`, `ruff` for formatting / linting.

3. **Security & privacy defaults**

   * No telemetry / remote logging by default.
   * Config in a local `.env` or `config.py` (LLM keys, paths).
   * Make sure all data (benchmarks, logs) are local files under `symprompt/`.

---

## 2. SymIL Core (Lite Level)

**Goal:** Implement a minimal **SymIL** and validator to support basic FOL-like reasoning. 

### 2.1 Define SymIL datamodel

Implement a **lite subset** of the grammar (facts, rules, query; constraints optional): 

* `Ontology`:

  * predicates: `{name, arity, types}`
  * constants (optional for now)
* `Formula`:

  * Atoms: `{pred, args}`
  * `Not`, `And`, `Or`, `Implies`, `ForAll`, `Exists`
* SymIL object:

  ```python
  class SymIL(BaseModel):
      ontology: Ontology
      facts: list[Formula] = []
      rules: list[Formula] = []
      query: Formula | None = None
      constraints: list[Formula] = []
  ```

Use the example from the doc (“All mammals are animals…”) as a unit test fixture. 

### 2.2 SymILValidator

Implement:

* Schema checks:

  * All predicates in formulas must appear in ontology.
  * Arity & types must match.
* Simple semantic sanity checks:

  * No free variables in rules/query (except quantify them).
  * No unknown formula types.

Add tests:

* Valid example from the doc. 
* Invalid examples (arity/type mismatch, unknown predicate).

---

## 3. Symbolic Reasoning Layer (Z3-only)

**Goal:** SymIL → SMT (Z3) → result.

### 3.1 FOL/SMT Compiler

Implement `fol_compiler.py`:

* Translate SymIL formulas to a **restricted FOL/SMT subset** that Z3 understands:

  * Universe type `Entity`.
  * Predicates as `BoolSort` functions over `Entity`s.
  * Quantifiers via Z3’s `ForAll`, `Exists`.
  * Simple domain: no functions at first.

Use the example in §4.1.2 as a test: compile the SymIL for mammals/cats/animals to Z3 constraints and query. 

### 3.2 z3_runner

Implement:

* `prove(symil: SymIL) -> bool | Unknown`:

  * Build solver.
  * Assert facts & rules.
  * Assert `¬query` and check unsat for validity.
* `satisfy(symil: SymIL) -> model | None` (optional for later tasks).

Add tests:

* A few hard-coded SymIL examples (syllogisms, small puzzles).
* Ensure expected True/False matches.

At this point, you can **manually craft SymIL** in code and prove things via Z3.

---

## 4. Baseline Translation Pipeline (no evolution yet)

**Goal:** A working `TranslationPipeline.translate(nl_prompt) -> SymIL` that uses your LLM with simple prompts. 

### 4.1 Preprocessor

`preprocess.py`:

* Normalize whitespace, fix quotes, maybe sentence-split.
* Strip obvious noise (“Explain step by step…”).
* Very small, deterministic.

### 4.2 OntologyExtractor (baseline)

`ontology.py` baseline:

* Prompt template:
  “Extract ontology for this logic problem. List predicates with names, arity, and types as JSON…”

* Parse LLM JSON into ontology structure.

* For now, **no evolution**; just a clean prompt and a few examples.

### 4.3 LogicalTranslator (baseline)

`logical.py` baseline:

* Prompt template:
  “Given this text and ontology, express facts, rules, and query in the following JSON SymIL format…”

* Ask LLM to output *only* the SymIL JSON (matching your schema).

* Parse JSON → SymIL datamodel.

* Run `SymILValidator` to catch basic issues.

If validation fails:

* Optionally send a very short error hint back to LLM and let it try once more.
* But keep it to 1–2 attempts for simplicity.

### 4.4 ConstraintMiner (optional v1)

You can stub `ConstraintMiner` out:

* For v1, simply return `[]` (no constraints) until you really need them.
  The architecture doc mentions constraints, but they’re an advanced feature. 

### 4.5 TranslationPipeline

`pipeline.py` implements the pipeline from the doc, but minimal: 

```python
class TranslationPipeline:
    def __init__(self, llm_client):
        self.preprocessor = Preprocessor()
        self.ontology_extractor = BaselineOntologyExtractor(llm_client)
        self.logical_translator = BaselineLogicalTranslator(llm_client)
        self.constraint_miner = NoopConstraintMiner()
        self.validator = SymILValidator()

    def translate(self, nl_prompt: str) -> SymIL:
        text = self.preprocessor.normalize(nl_prompt)
        ontology = self.ontology_extractor.extract(text)
        formulas = self.logical_translator.translate(text, ontology)
        constraints = self.constraint_miner.mine(text, formulas)
        symil = SymIL(ontology=ontology,
                      facts=formulas.facts,
                      rules=formulas.rules,
                      query=formulas.query,
                      constraints=constraints)
        return self.validator.validate(symil)
```

Add integration tests:

* Take a few hand-labeled NL→SymIL examples.
* Ensure pipeline output matches (or is semantically equivalent).

---

## 5. Minimal Evaluation Framework (local, tiny benchmarks)

**Goal:** Have a **local fitness function** over a small benchmark set that OpenEvolve can optimize against. 

### 5.1 Build a tiny benchmark JSON

In `benchmarks/tiny_folio.json`, define e.g. 10–20 items:

```json
[
  {
    "id": "cats_mammals_animals",
    "nl": "All mammals are animals. All cats are mammals. Therefore, all cats are animals.",
    "gold_symil": { ... optional ... },
    "expected_result": "VALID"
  },
  ...
]
```

You can:

* Either store full gold SymIL (for direct translation comparison), or
* Just store the expected solver result (VALID/INVALID).

### 5.2 eval_pipeline.py

Implement `evaluate_translation_pipeline(pipeline, benchmark_file)`:

For each item:

1. `symil = pipeline.translate(problem["nl"])`
2. Run `prove(symil)` with Z3.
3. Compare with `expected_result`.
4. Track:

   * `syntax_ok` (validation passed),
   * `solver_success` (Z3 returned expected answer),
   * `latency` per item.

Return an aggregated dict, then a scalar score similar to the doc’s fitness function (but simplified): 

```python
score = (
    0.4 * syntactic_validity +
    0.4 * solver_success_rate +
    0.2 * latency_score
)
```

This gives you a **single number** OpenEvolve can optimize.

---

## 6. OpenEvolve Integration (lightweight, personal)

**Goal:** Let OpenEvolve evolve your translation pipeline in a small way: mainly prompt templates + simple logic inside `OntologyExtractor` and `LogicalTranslator`. 

### 6.1 Define evolvable targets

Decide what OpenEvolve is allowed to mutate:

* The code of:

  * `OntologyExtractor.extract()`
  * `LogicalTranslator.translate()`
* And/or prompt templates stored as strings in `config.py`.

Keep everything else fixed.

### 6.2 Configure OpenEvolve

Use a scaled-down version of the config from the doc: 

```yaml
evolution:
  population_size: 10
  generations: 20
  elite_fraction: 0.2
  mutation_rate: 0.3

llm_ensemble:
  primary: "your_llm_name"
  secondary: null
  ratio: 1.0

evaluation:
  metrics:
    - name: syntactic_validity
      weight: 0.4
    - name: solver_success_rate
      weight: 0.5
    - name: latency_score
      weight: 0.1

  benchmarks:
    - tiny_folio
```

**Personal use tweak:**

* Just use **one LLM backend** (local or via API).
* Use a small population / low number of generations to keep cost + runtime sane.

### 6.3 Evolution loop

Implement a small wrapper script:

```python
# evolution/run_evolution.py
from openevolve import Engine
from symprompt.evolution.eval_pipeline import evaluate_translation_pipeline

engine = Engine(config_path="openevolve_config.yaml")

engine.run(
    target_module="symprompt.translation.pipeline",
    target_class="TranslationPipeline",
    eval_fn=evaluate_translation_pipeline
)
```

This will:

* Generate variants of the translator code,
* Evaluate them on `tiny_folio`,
* Track best candidates.

You then manually inspect the best few, sanity‑check them on your own prompts, and freeze one as your personal translator.

---

## 7. LLM Integration for Personal Use (CLI / notebook)

**Goal:** Make it pleasant to use personally: from a CLI or notebook, you type a question and see a symbolic proof-backed answer.

### 7.1 Router (minimal)

For now, a super simple router:

* If prompt contains logic-y keywords (from the doc’s `SYMBOLIC_INDICATORS` list), send it through SymPrompt; otherwise, call LLM directly. 

```python
class PromptRouter:
    KEYWORDS = [...same as doc...]

    def should_use_symbolic(self, text: str) -> bool:
        return any(k in text.lower() for k in self.KEYWORDS)
```

No LLM classifier yet (keep it simple).

### 7.2 CLI

`integration/cli.py`:

* Read input from stdin.
* If router says yes:

  1. Use `TranslationPipeline.translate()` to get SymIL.
  2. Use `prove()` to get True/False.
  3. Ask LLM:
     “The statement is logically VALID/INVALID. Explain why in natural language.”
* Else: call LLM normally.

Optionally, print the SymIL JSON so you can inspect your neurosymbolic representation directly.

### 7.3 Notebook demo

`integration/notebook_demo.ipynb`:

* Walk through:

  * Visualizing SymIL.
  * Running Z3 on it.
  * Asking LLM to explain the proof.

Great for debugging and personal experimentation.

---

## 8. Nice-to-have Next Steps (after v1)

Once the basic private SymPrompt works, you can gradually add:

1. **More benchmarks** (a small subset of FOLIO, MALLS, ProntoQA) as separate JSON files. 
2. **Post-processing mode** (verify LLM’s answer instead of the question). 
3. **Constraints + ASP backend** if you need planning / default reasoning. 
4. **A neuro‑symbolic backend**: e.g., SymIL → Scallop or a small VSA module for differentiable reasoning. 
5. **Better routing** via a tiny classifier fine‑tuned on your own prompts.

---

### Summary

This plan builds a **SymPrompt‑Lite** that is:

* Simple enough to implement and run on a single personal machine.
* Fully local and private (all data + solvers + code on your box).
* Ready to be extended toward the full SymPrompt vision once the core is stable.

If you’d like, I can next draft some concrete SymIL examples and baseline LLM prompts for `OntologyExtractor` and `LogicalTranslator` so you have something copy‑pasteable to start v1.

++++++++++++++++++++++
Got it — let’s push this from “plan on paper” toward something you can actually start coding.

I’ll give you **three concrete building blocks** you can drop into your repo:

1. A worked **NL → SymIL example** (aligned with your architecture).
2. A **minimal SymIL + Z3 skeleton** in Python.
3. **Baseline LLM prompts** for `OntologyExtractor` and `LogicalTranslator` that implement the v0 TranslationPipeline. 

You can literally paste‑adapt these into your modules.

---

## 1. Concrete NL → SymIL Example

Take the classic one already used in the spec: 

> “All mammals are animals. All cats are mammals. Therefore, all cats are animals.”

### 1.1 SymIL JSON

```json
{
  "ontology": {
    "predicates": [
      {"name": "mammal", "arity": 1, "types": ["entity"], "description": "X is a mammal"},
      {"name": "animal", "arity": 1, "types": ["entity"], "description": "X is an animal"},
      {"name": "cat",    "arity": 1, "types": ["entity"], "description": "X is a cat"}
    ],
    "functions": [],
    "constants": []
  },

  "facts": [],

  "rules": [
    {
      "forall": "X",
      "type": "entity",
      "body": {
        "implies": [
          {"pred": "mammal", "args": ["X"]},
          {"pred": "animal", "args": ["X"]}
        ]
      }
    },
    {
      "forall": "X",
      "type": "entity",
      "body": {
        "implies": [
          {"pred": "cat", "args": ["X"]},
          {"pred": "mammal", "args": ["X"]}
        ]
      }
    }
  ],

  "query": {
    "prove": {
      "forall": "X",
      "type": "entity",
      "body": {
        "implies": [
          {"pred": "cat", "args": ["X"]},
          {"pred": "animal", "args": ["X"]}
        ]
      }
    }
  },

  "constraints": []
}
```

That matches the SymIL grammar in section 4.1.1/4.1.2 exactly. 

---

## 2. Minimal SymIL + Z3 Skeleton (Python)

This is intentionally lightweight for personal use. You can refine types later.

### 2.1 `core/symil.py`

```python
# symprompt/core/symil.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional, Any


# --- Ontology ---

@dataclass
class Predicate:
    name: str
    arity: int
    types: List[str]  # e.g. ["entity"]
    description: str | None = None


@dataclass
class Ontology:
    predicates: List[Predicate] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)

    def get_predicate(self, name: str) -> Predicate | None:
        for p in self.predicates:
            if p.name == name:
                return p
        return None


# --- Formula AST ---

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
    # For now, we just reuse Formula but keep this wrapper for clarity.
    forall: str
    type: str
    body: Formula


@dataclass
class Query:
    # For v0 we only support "prove" queries.
    prove: Formula


@dataclass
class SymIL:
    ontology: Ontology
    facts: List[Formula] = field(default_factory=list)
    rules: List[Rule] = field(default_factory=list)
    query: Optional[Query] = None
    constraints: List[Formula] = field(default_factory=list)
    level: int = 0


# --- Validator ---

class SymILValidationError(Exception):
    pass


class SymILValidator:
    def validate(self, symil: SymIL) -> SymIL:
        self._check_predicates(symil)
        # you can add free-variable & type checks later
        return symil

    def _check_predicates(self, symil: SymIL) -> None:
        def check_formula(f: Formula):
            if isinstance(f, Atom):
                pred = symil.ontology.get_predicate(f.pred)
                if pred is None:
                    raise SymILValidationError(f"Unknown predicate: {f.pred}")
                if len(f.args) != pred.arity:
                    raise SymILValidationError(
                        f"Arity mismatch for {f.pred}: expected {pred.arity}, got {len(f.args)}"
                    )
            elif isinstance(f, (Not,)):
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

### 2.2 `core/compiler_z3.py`

```python
# symprompt/core/compiler_z3.py
from __future__ import annotations
from typing import Dict
import z3

from .symil import (
    SymIL, Formula, Atom, Not, And, Or, Implies, ForAll, Exists,
)


def symil_to_z3(symil: SymIL) -> dict:
    """
    Returns a dict containing:
      - 'ctx': context
      - 'predicates': mapping from predicate name to z3 function
      - 'vars': mapping from var name to z3 Const
      - 'axioms': list of z3 BoolRef (facts + rules)
      - 'query': z3 BoolRef or None
    """
    Entity = z3.DeclareSort("Entity")

    # Create predicate symbols
    pred_fns: Dict[str, z3.FuncDeclRef] = {}
    for p in symil.ontology.predicates:
        # For now we assume all arg types are Entity
        arg_sorts = [Entity] * p.arity
        pred_fns[p.name] = z3.Function(p.name, *arg_sorts, z3.BoolSort())

    def encode_formula(f: Formula, env: Dict[str, z3.ExprRef]) -> z3.BoolRef:
        if isinstance(f, Atom):
            fn = pred_fns[f.pred]
            args = [env.get(a, z3.Const(a, Entity)) for a in f.args]
            # update env for unknown variables
            for a in f.args:
                if a not in env:
                    env[a] = args[f.args.index(a)]
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
        # rules are essentially ForAll bodies already
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

    result = None
    if ctx["query"] is not None:
        # Check if the query is logically valid: assert ¬query and see if UNSAT
        solver.push()
        solver.add(z3.Not(ctx["query"]))
        sat_result = solver.check()
        solver.pop()
        if sat_result == z3.unsat:
            result = "VALID"
        elif sat_result == z3.sat:
            result = "NOT_VALID"
        else:
            result = "UNKNOWN"
    else:
        sat_result = solver.check()
        result = str(sat_result)

    return {"status": result}
```

This is deliberately minimal, but enough to:

* Represent your SymIL in code,
* Compile to Z3,
* Answer “VALID / NOT_VALID / UNKNOWN” for simple proofs.

---

## 3. Baseline LLM Prompts for Translation

Now the fun part: prompts for `OntologyExtractor` and `LogicalTranslator` that align with the SymIL spec. 

### 3.1 Baseline `OntologyExtractor` Prompt

You can keep this in `translate/ontology.py` as a string constant:

```python
ONTOLOGY_SYSTEM_PROMPT = """
You are a symbolic ontology extractor.

Your job is to read a short natural language statement and propose
a minimal logical ontology for it.

Output ONLY a JSON object with this structure:

{
  "predicates": [
    {"name": <string>, "arity": 1, "types": ["entity"], "description": <string>},
    ...
  ],
  "functions": [],
  "constants": []
}

Rules:
- Use lowercase predicate names without spaces, e.g. "mammal", "animal", "cat".
- For now, assume all predicates have arity 1 and type "entity".
- Do NOT include any comments or extra keys.
"""

ONTOLOGY_USER_TEMPLATE = """
Text:
{TEXT}

Extract the ontology as JSON.
"""
```

Usage in code (pseudo):

```python
class OntologyExtractor:
    def __init__(self, llm_client):
        self.llm = llm_client

    def extract(self, text: str) -> Ontology:
        prompt = ONTOLOGY_SYSTEM_PROMPT + ONTOLOGY_USER_TEMPLATE.format(TEXT=text)
        raw = self.llm.chat(prompt)
        data = json.loads(raw)
        preds = [
            Predicate(
                name=p["name"],
                arity=p.get("arity", 1),
                types=p.get("types", ["entity"]),
                description=p.get("description"),
            )
            for p in data["predicates"]
        ]
        return Ontology(predicates=preds)
```

### 3.2 Baseline `LogicalTranslator` Prompt

We want the model to emit the **SymIL fragment** (rules + query) matching the grammar in 4.1.2. 

```python
LOGICAL_SYSTEM_PROMPT = """
You are a translator from natural language into a JSON-based logical
intermediate language called SymIL.

Given:
- A short piece of text describing logical relationships.
- An ontology listing predicates.

You must output ONLY a JSON object with the following structure:

{
  "facts": [ /* usually empty, list of formula objects */ ],
  "rules": [
    {
      "forall": "X",
      "type": "entity",
      "body": {
        "implies": [
          { "pred": "<name>", "args": ["X"] },
          { "pred": "<name>", "args": ["X"] }
        ]
      }
    }
  ],
  "query": {
    "prove": {
      "forall": "X",
      "type": "entity",
      "body": {
        "implies": [
          { "pred": "<name>", "args": ["X"] },
          { "pred": "<name>", "args": ["X"] }
        ]
      }
    }
  }
}

Rules:
- Use only the predicates defined in the ontology.
- Use variable names like "X", "Y" and assume type "entity".
- Use "implies" for conditional statements (IF/THEN, "all ... are ...", etc).
- Use "forall" for universal statements "all X ...".
- For this v0, do not use quantifiers other than "forall".
- Do NOT add comments or extra keys.
"""

LOGICAL_USER_TEMPLATE = """
Text:
{TEXT}

Ontology (JSON):
{ONTOLOGY_JSON}

Produce the SymIL fragment JSON (facts, rules, query) for this text.
"""
```

Example call for the mammals/cats case:

```python
class LogicalTranslator:
    def __init__(self, llm_client):
        self.llm = llm_client

    def translate(self, text: str, ontology: Ontology):
        ontology_json = json.dumps({
            "predicates": [
                {
                    "name": p.name,
                    "arity": p.arity,
                    "types": p.types,
                    "description": p.description or ""
                }
                for p in ontology.predicates
            ],
            "functions": [],
            "constants": []
        }, indent=2)

        prompt = LOGICAL_SYSTEM_PROMPT + LOGICAL_USER_TEMPLATE.format(
            TEXT=text,
            ONTOLOGY_JSON=ontology_json,
        )
        raw = self.llm.chat(prompt)
        data = json.loads(raw)

        # Convert JSON to Formula/Rule/Query objects
        rules = [self._rule_from_json(r) for r in data.get("rules", [])]
        facts = [self._formula_from_json(f) for f in data.get("facts", [])]
        query = self._query_from_json(data.get("query"))

        return facts, rules, query

    # Implement _formula_from_json, _rule_from_json, _query_from_json
    # to build Atom/Implies/ForAll/etc
```

You can seed the model with the exact example from the architecture doc as a **few‑shot** inside `LOGICAL_SYSTEM_PROMPT` to anchor its structure. 

---

## 4. How to use all this right now

Minimal end‑to‑end test you can do once you wire things up:

1. Hard‑code the mammals/cats text:

   ```python
   text = "All mammals are animals. All cats are mammals. Therefore, all cats are animals."
   ```

2. Run:

   ```python
   pipeline = TranslationPipeline(llm_client)
   symil = pipeline.translate(text)
   result = solve_symil(symil)
   print(symil)
   print(result)  # expect: {'status': 'VALID'}
   ```

3. Then call `answer_with_symprompt(llm_client, text)` and compare the explanation to what you’d expect logically.

That gives you a **fully working vertical slice** of SymPrompt‑Lite, with clear spots to later plug in OpenEvolve to mutate the prompts / code in `OntologyExtractor` and `LogicalTranslator` for better accuracy. 

If you’d like next, I can draft the `_formula_from_json` / `_rule_from_json` helpers, or a tiny `evaluate_pipeline` script that you can immediately point OpenEvolve at.

++++++
Nice, let’s turn this into a runnable mini‑framework.

Below you’ll get:

1. A **minimal TranslationPipeline** tying everything together.
2. JSON → SymIL **parsers** (`_formula_from_json`, `_rule_from_json`, `_query_from_json`).
3. A tiny **evaluation harness** that you can plug into OpenEvolve or run by hand.

All of this is consistent with the SymPrompt architecture and SymIL spec from your document. 

---

## 1. TranslationPipeline (glue everything together)

Create `symprompt/translation/pipeline.py`:

```python
# symprompt/translation/pipeline.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List

from symprompt.core.symil import SymIL, Ontology
from symprompt.core.symil import SymILValidator
from symprompt.core.symil import Formula, Rule, Query
from symprompt.translation.preprocess import Preprocessor
from symprompt.translation.ontology_extractor import OntologyExtractor
from symprompt.translation.logical_translator import LogicalTranslator


@dataclass
class TranslationPipeline:
    """
    Minimal NL -> SymIL translation pipeline.

    - Preprocessor: cleans up natural language.
    - OntologyExtractor: LLM-based, builds SymIL ontology.
    - LogicalTranslator: LLM-based, builds facts/rules/query.
    - Validator: checks predicate arity & basic structure.
    """

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

    def translate(self, nl_prompt: str) -> SymIL:
        """
        End-to-end NL -> SymIL (ontology + facts + rules + query).
        Raises SymILValidationError if the result fails validation.
        """
        text = self.preprocessor.normalize(nl_prompt)

        ontology: Ontology = self.ontology_extractor.extract(text)
        facts, rules, query = self.logical_translator.translate(text, ontology)

        symil = SymIL(
            ontology=ontology,
            facts=facts,
            rules=rules,
            query=query,
            constraints=[],  # keep empty in v0
        )

        return self.validator.validate(symil)
```

And a super simple preprocessor in `symprompt/translation/preprocess.py`:

```python
# symprompt/translation/preprocess.py
from __future__ import annotations


class Preprocessor:
    def normalize(self, text: str) -> str:
        # Minimal normalization for v0
        return " ".join(text.strip().split())
```

---

## 2. OntologyExtractor (using the prompt we defined)

`symprompt/translation/ontology_extractor.py`:

```python
# symprompt/translation/ontology_extractor.py
from __future__ import annotations
import json
from typing import Any, Dict, List

from symprompt.core.symil import Ontology, Predicate
from symprompt.prompts.ontology_prompts import build_ontology_prompt


class OntologyExtractor:
    def __init__(self, llm_client):
        """
        llm_client must expose a method:

            llm_client.complete(prompt: str) -> str

        which returns the raw JSON string.
        """
        self.llm = llm_client

    def extract(self, text: str) -> Ontology:
        prompt = build_ontology_prompt(text)
        raw = self.llm.complete(prompt)
        data = json.loads(raw)

        ont = data.get("ontology", data)  # allow either format
        preds_json: List[Dict[str, Any]] = ont.get("predicates", [])

        predicates = [
            Predicate(
                name=p["name"],
                arity=p.get("arity", 1),
                types=p.get("types", ["entity"]),
                description=p.get("description"),
            )
            for p in preds_json
        ]

        return Ontology(predicates=predicates)
```

And the prompts in `symprompt/prompts/ontology_prompts.py` (this is essentially what I gave you already):

```python
# symprompt/prompts/ontology_prompts.py
ONTOLOGY_SYSTEM_PROMPT = """You are an ontology designer for a logic reasoning system.
Your job is to read natural language text and extract a minimal set of logical predicates
and constants that will be used to formalize the text in first-order logic.

The target representation is the SymIL ontology section:

{
  "ontology": {
    "predicates": [
      {"name": "mammal", "arity": 1, "types": ["entity"], "description": "X is a mammal"},
      {"name": "animal", "arity": 1, "types": ["entity"], "description": "X is an animal"}
    ],
    "functions": [],
    "constants": [
      {"name": "socrates", "type": "entity", "description": "Socrates, a person"}
    ]
  }
}

Rules, facts, queries and constraints are NOT part of this step.
You must ONLY return the ontology field as well-formed JSON with keys: "ontology", "predicates", "functions", "constants".
Use simple lowercase names for predicates and constants, no spaces.
Use "entity" as the only type for now.
"""

ONTOLOGY_FEWSHOT_EXAMPLES = """
Example 1
---------
Input:
"All mammals are animals. All cats are mammals. Therefore, all cats are animals."

Output:
{
  "ontology": {
    "predicates": [
      {"name": "mammal", "arity": 1, "types": ["entity"], "description": "X is a mammal"},
      {"name": "animal", "arity": 1, "types": ["entity"], "description": "X is an animal"},
      {"name": "cat",    "arity": 1, "types": ["entity"], "description": "X is a cat"}
    ],
    "functions": [],
    "constants": []
  }
}

Example 2
---------
Input:
"Some dogs are friendly. All friendly things are loved by someone."

Output:
{
  "ontology": {
    "predicates": [
      {"name": "dog",      "arity": 1, "types": ["entity"], "description": "X is a dog"},
      {"name": "friendly", "arity": 1, "types": ["entity"], "description": "X is friendly"},
      {"name": "loves",    "arity": 2, "types": ["entity", "entity"], "description": "X loves Y"}
    ],
    "functions": [],
    "constants": []
  }
}
"""

def build_ontology_prompt(nl_text: str) -> str:
    return (
        ONTOLOGY_SYSTEM_PROMPT
        + "\n\n"
        + ONTOLOGY_FEWSHOT_EXAMPLES
        + "\n\nNow process this input text and output ONLY the ontology JSON:\n\n"
        + nl_text
        + "\n\nOutput JSON:\n"
    )
```

---

## 3. LogicalTranslator + JSON → SymIL parsers

Here’s the key piece: parse the LLM’s SymIL JSON into your `Formula`, `Rule`, `Query` dataclasses.

`symprompt/translation/logical_translator.py`:

```python
# symprompt/translation/logical_translator.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple

from symprompt.core.symil import (
    Formula,
    Atom,
    Not,
    And,
    Or,
    Implies,
    ForAll,
    Exists,
    Rule,
    Query,
    SymILValidationError,
    Ontology,
)
from symprompt.prompts.logic_prompts import build_logic_prompt


def formula_from_json(obj: Dict[str, Any]) -> Formula:
    """
    Convert a SymIL formula JSON object to a Formula AST node.
    Supports:
      - {"pred": "name", "args": ["X"...]}
      - {"not": <Formula>}
      - {"and": [<Formula>, ...]}
      - {"or": [<Formula>, ...]}
      - {"implies": [<Formula>, <Formula>]}
      - {"forall": "X", "type": "entity", "body": <Formula>}
      - {"exists": "X", "type": "entity", "body": <Formula>}
    """
    if "pred" in obj:
        return Atom(pred=obj["pred"], args=obj.get("args", []))

    if "not" in obj:
        return Not(formula=formula_from_json(obj["not"]))

    if "and" in obj:
        return And(formulas=[formula_from_json(x) for x in obj["and"]])

    if "or" in obj:
        return Or(formulas=[formula_from_json(x) for x in obj["or"]])

    if "implies" in obj:
        premise_json, conclusion_json = obj["implies"]
        return Implies(
            premise=formula_from_json(premise_json),
            conclusion=formula_from_json(conclusion_json),
        )

    if "forall" in obj and "body" in obj:
        return ForAll(
            var=obj["forall"],
            type=obj.get("type", "entity"),
            body=formula_from_json(obj["body"]),
        )

    if "exists" in obj and "body" in obj:
        return Exists(
            var=obj["exists"],
            type=obj.get("type", "entity"),
            body=formula_from_json(obj["body"]),
        )

    raise SymILValidationError(f"Unknown formula JSON structure: {obj}")


def rule_from_json(obj: Dict[str, Any]) -> Rule:
    """
    Expecting objects like:
      { "forall": "X", "type": "entity", "body": { ...formula... } }
    or similarly for "exists".
    """
    if "forall" in obj:
        return Rule(
            forall=obj["forall"],
            type=obj.get("type", "entity"),
            body=formula_from_json(obj["body"]),
        )

    # You might also allow existential rules, but for v0 it's fine to
    # treat them as ForAll over an Exists in the body.
    if "exists" in obj:
        # example: exists X. body(X)
        exists_formula = Exists(
            var=obj["exists"],
            type=obj.get("type", "entity"),
            body=formula_from_json(obj["body"]),
        )
        # Wrap with a dummy forall rule if needed
        return Rule(
            forall="_",
            type="entity",
            body=exists_formula,
        )

    raise SymILValidationError(f"Unknown rule JSON structure: {obj}")


def query_from_json(obj: Dict[str, Any]) -> Query:
    """
    Expecting objects like:
      { "prove": <Formula JSON> }
    """
    if obj is None:
        raise SymILValidationError("Query JSON is None, cannot build Query.")
    if "prove" not in obj:
        raise SymILValidationError(f"Query JSON missing 'prove' key: {obj}")

    return Query(prove=formula_from_json(obj["prove"]))


class LogicalTranslator:
    def __init__(self, llm_client):
        """
        llm_client.complete(prompt: str) -> str
        """
        self.llm = llm_client

    def translate(
        self,
        text: str,
        ontology: Ontology,
    ) -> Tuple[List[Formula], List[Rule], Query]:
        """
        Returns (facts, rules, query).
        """
        ontology_json = self._ontology_to_json(ontology)
        prompt = build_logic_prompt(text, ontology_json)
        raw = self.llm.complete(prompt)
        data = json.loads(raw)

        facts_json = data.get("facts", [])
        rules_json = data.get("rules", [])
        query_json = data.get("query", None)

        facts = [formula_from_json(f) for f in facts_json]
        rules = [rule_from_json(r) for r in rules_json]
        query = query_from_json(query_json) if query_json is not None else None

        if query is None:
            raise SymILValidationError("LogicalTranslator: missing 'query' in LLM output.")

        return facts, rules, query

    @staticmethod
    def _ontology_to_json(ontology: Ontology) -> str:
        obj = {
            "ontology": {
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
        }
        return json.dumps(obj, indent=2)
```

And the prompts in `symprompt/prompts/logic_prompts.py` (slightly abbreviated for brevity, but you can use the full version I gave before):

```python
# symprompt/prompts/logic_prompts.py
LOGIC_SYSTEM_PROMPT = """You convert natural language text into SymIL, a JSON-based
intermediate language for first-order logic.

You will be given:
1) The original natural language text.
2) The SymIL ontology section (predicates and constants).

Your job is to produce ONLY the logical part of SymIL:
- "facts": a list of ground atomic facts (often empty)
- "rules": a list of universally or existentially quantified formulas
- "query": the main formula to prove, wrapped in a {"prove": ...} object
- "constraints": a list of hard constraints (often empty)

SymIL formula patterns:

Atom      -> {"pred": "name", "args": ["X", ...]}
Not       -> {"not": <Formula>}
And       -> {"and": [<Formula>, ...]}
Or        -> {"or": [<Formula>, ...]}
Implies   -> {"implies": [<Formula>, <Formula>]}
ForAll    -> {"forall": "X", "type": "entity", "body": <Formula>}
Exists    -> {"exists": "X", "type": "entity", "body": <Formula>}

Return a single JSON object with keys: "facts", "rules", "query", "constraints".
"""

LOGIC_FEWSHOT_EXAMPLES = """
Example 1
---------
Natural language:
"All mammals are animals. All cats are mammals. Therefore, all cats are animals."

Ontology:
{
  "ontology": {
    "predicates": [
      {"name": "mammal", "arity": 1, "types": ["entity"], "description": "X is a mammal"},
      {"name": "animal", "arity": 1, "types": ["entity"], "description": "X is an animal"},
      {"name": "cat",    "arity": 1, "types": ["entity"], "description": "X is a cat"}
    ],
    "functions": [],
    "constants": []
  }
}

Output:
{
  "facts": [],
  "rules": [
    {
      "forall": "X",
      "type": "entity",
      "body": {
        "implies": [
          {"pred": "mammal", "args": ["X"]},
          {"pred": "animal", "args": ["X"]}
        ]
      }
    },
    {
      "forall": "X",
      "type": "entity",
      "body": {
        "implies": [
          {"pred": "cat", "args": ["X"]},
          {"pred": "mammal", "args": ["X"]}
        ]
      }
    }
  ],
  "query": {
    "prove": {
      "forall": "X",
      "type": "entity",
      "body": {
        "implies": [
          {"pred": "cat", "args": ["X"]},
          {"pred": "animal", "args": ["X"]}
        ]
      }
    }
  },
  "constraints": []
}
"""

def build_logic_prompt(nl_text: str, ontology_json: str) -> str:
    return (
        LOGIC_SYSTEM_PROMPT
        + "\n\n"
        + LOGIC_FEWSHOT_EXAMPLES
        + "\n\nNatural language to formalize:\n"
        + nl_text
        + "\n\nOntology:\n"
        + ontology_json
        + "\n\nNow output ONLY the JSON object with keys: facts, rules, query, constraints.\n\nOutput JSON:\n"
    )
```

---

## 4. Tiny evaluation harness (for you + OpenEvolve)

Now, a very small evaluator that:

* Runs the pipeline on some examples,
* Calls Z3,
* Computes basic metrics and a combined score (fitness).

`symprompt/evolution/eval_pipeline.py`:

```python
# symprompt/evolution/eval_pipeline.py
from __future__ import annotations
import time
from statistics import quantiles
from typing import List, Dict, Any, Callable

from symprompt.core.symil import SymILValidationError
from symprompt.core.compiler_z3 import solve_symil
from symprompt.translation.pipeline import TranslationPipeline

# You can import the SymIL examples from a module or load JSON from disk.
from symprompt.symil.examples import TINY_EXAMPLES  # list of dicts


def p95(values: List[float]) -> float:
    if not values:
        return 0.0
    # statistics.quantiles with n=20 approximates percentiles; index 18 ~ 95th
    return quantiles(values, n=20)[18]


def latency_score(p95_latency_sec: float) -> float:
    """
    Map latency to [0,1]:
      - <= 0.2s -> ~1.0
      - >= 1.0s -> ~0.0
    """
    if p95_latency_sec <= 0.2:
        return 1.0
    if p95_latency_sec >= 1.0:
        return 0.0
    # linear interpolation
    return max(0.0, 1.0 - (p95_latency_sec - 0.2) / 0.8)


def evaluate_translation_pipeline(
    pipeline: TranslationPipeline,
    examples: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Simple evaluation:
      - syntactic_validity: fraction of examples that pass SymIL validation
      - solver_success_rate: fraction where solver result == expected_result
        (only for examples that specify VALID/INVALID)
      - p95_latency
      - combined_score: weighted sum

    This can be wrapped as the fitness function for OpenEvolve.
    """
    if examples is None:
        examples = TINY_EXAMPLES

    n = len(examples)
    if n == 0:
        return {"syntactic_validity": 0.0, "solver_success_rate": 0.0,
                "p95_latency": 0.0, "combined_score": 0.0}

    syntactic_ok = 0
    solver_ok = 0
    latencies: List[float] = []

    for ex in examples:
        text = ex["natural_language"]
        expected = ex.get("expected_result")

        t0 = time.time()
        try:
            symil = pipeline.translate(text)
            syntactic_ok += 1
        except SymILValidationError:
            latencies.append(time.time() - t0)
            continue

        result = solve_symil(symil)
        latencies.append(time.time() - t0)

        if expected in ("VALID", "INVALID"):
            if result["status"] == expected:
                solver_ok += 1

    syntactic_validity = syntactic_ok / n
    solver_success_rate = solver_ok / max(1, sum(1 for ex in examples
                                                if ex.get("expected_result") in ("VALID", "INVALID")))
    p95_latency_val = p95(latencies)
    lat_score = latency_score(p95_latency_val)

    combined = 0.4 * syntactic_validity + 0.5 * solver_success_rate + 0.1 * lat_score

    return {
        "syntactic_validity": syntactic_validity,
        "solver_success_rate": solver_success_rate,
        "p95_latency": p95_latency_val,
        "latency_score": lat_score,
        "combined_score": combined,
    }
```

You can then run a quick local test:

```python
# test_run.py (top-level script)
from symprompt.translation.pipeline import TranslationPipeline
from symprompt.evolution.eval_pipeline import evaluate_translation_pipeline
from my_llm_client import MyLLMClient

llm = MyLLMClient(...)
pipeline = TranslationPipeline.from_llm_client(llm)

metrics = evaluate_translation_pipeline(pipeline)
print(metrics)
```

That gives you a quantitative baseline you can immediately start improving (and that OpenEvolve can treat as its fitness function, matching the “multi-objective evaluation” section of your architecture doc). 

---

If you want, next step I can show how to define `TINY_EXAMPLES` as Python objects (instead of JSON files) and wire up a minimal CLI that uses `PromptRouter` + this pipeline so you can play with SymPrompt interactively.
++++++++++
Nice, let’s finish wiring this into something you can actually *use*: tiny benchmark set, router, and a simple CLI that does:

> NL prompt → SymIL → Z3 → answer explanation via LLM

all locally, consistent with the SymPrompt architecture. 

I’ll assume you drop this into the skeleton we already sketched.

---

## 1. Tiny examples (so you can test + evolve)

Create `symprompt/symil/examples.py`:

```python
# symprompt/symil/examples.py
from __future__ import annotations
from typing import List, Dict, Any

# Minimal evaluation set for personal usage and OpenEvolve fitness.
# expected_result is about logical validity of the argument.

TINY_EXAMPLES: List[Dict[str, Any]] = [
    {
        "id": "cats_mammals_animals",
        "natural_language": (
            "All mammals are animals. "
            "All cats are mammals. "
            "Therefore, all cats are animals."
        ),
        "expected_result": "VALID",
    },
    {
        "id": "socrates_mortal",
        "natural_language": (
            "All humans are mortal. "
            "Socrates is a human. "
            "Therefore, Socrates is mortal."
        ),
        "expected_result": "VALID",
    },
    {
        "id": "invalid_inference",
        "natural_language": (
            "All dogs are animals. "
            "All cats are animals. "
            "Therefore, all cats are dogs."
        ),
        "expected_result": "INVALID",
    },
    {
        "id": "exists_some_dog_friendly",
        "natural_language": (
            "Some dogs are friendly. "
            "All friendly things are loved by someone. "
            "Therefore, some dogs are loved by someone."
        ),
        "expected_result": "VALID",
    },
    {
        "id": "no_cats_are_dogs",
        "natural_language": (
            "No cats are dogs. "
            "Tom is a cat. "
            "Therefore, Tom is not a dog."
        ),
        "expected_result": "VALID",
    },
]
```

You can add more as you go, but this is enough to sanity‑check the pipeline and drive an initial OpenEvolve run.

---

## 2. Minimal PromptRouter

This is the little heuristic router described in the architecture doc: it decides when to invoke symbolic reasoning vs just using the LLM directly. 

Create `symprompt/integration/router.py`:

```python
# symprompt/integration/router.py
from __future__ import annotations


class PromptRouter:
    """
    Very simple router: decides if a prompt should go through SymPrompt.

    For now we just use keyword heuristics; you can upgrade this later with
    an LLM-based classifier when you want something smarter.
    """

    SYMBOLIC_INDICATORS = [
        "prove",
        "valid",
        "invalid",
        "follows",
        "implies",
        "therefore",
        "if ",
        "if and only if",
        "all ",
        "every ",
        "some ",
        "no ",
        "none ",
        "must be",
        "cannot be",
        "deduce",
        "infer",
        "conclude",
        "logical",
        "contradiction",
    ]

    def should_use_symbolic(self, prompt: str) -> bool:
        text = prompt.lower()
        return any(ind in text for ind in self.SYMBOLIC_INDICATORS)
```

---

## 3. Simple LLM client stub

You probably already have a client for your favorite model. Here’s a tiny adapter so the rest of the code can treat it generically.

Create `symprompt/integration/llm_client.py`:

```python
# symprompt/integration/llm_client.py
from __future__ import annotations
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        """
        Return a raw string completion for the given prompt.
        This is the only method SymPrompt cares about.
        """
        ...


# Example adapter for an OpenAI-like client
class OpenAIClient:
    def __init__(self, openai_client, model: str = "openrouter/x-ai/grok-4.1-fast:free"):
        self.client = openai_client
        self.model = model

    def complete(self, prompt: str) -> str:
        # Adjust to your SDK; this is just a sketch.
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content
```

Swap this with whatever library you use (OpenAI, local LLM, etc.).

---

## 4. CLI: ask a question, get a solver‑backed answer

Now a tiny CLI that stitches everything together:

Create `symprompt/integration/cli.py`:

```python
# symprompt/integration/cli.py
from __future__ import annotations
import argparse
import sys
from typing import Optional

from symprompt.translation.pipeline import TranslationPipeline
from symprompt.core.compiler_z3 import solve_symil
from symprompt.integration.router import PromptRouter
from symprompt.integration.llm_client import LLMClient


class SymPromptCLI:
    def __init__(self, llm_client: LLMClient):
        self.router = PromptRouter()
        self.pipeline = TranslationPipeline.from_llm_client(llm_client)
        self.llm = llm_client

    def run_once(self, prompt: str) -> None:
        if not self.router.should_use_symbolic(prompt):
            # Just pass to LLM
            answer = self.llm.complete(prompt)
            print("\n[LLM ANSWER]\n")
            print(answer)
            return

        print("\n[SymPrompt] Using symbolic pipeline...\n")
        try:
            symil = self.pipeline.translate(prompt)
        except Exception as e:
            print("[SymPrompt] Translation failed, falling back to LLM.")
            print(f"Error: {e}")
            answer = self.llm.complete(prompt)
            print("\n[LLM ANSWER]\n")
            print(answer)
            return

        # Solve with Z3
        result = solve_symil(symil)
        status = result["status"]

        print(f"[SymPrompt] Solver status: {status}")

        # Ask LLM to explain the solver result and the argument
        explanation_prompt = (
            "You are a logic tutor.\n"
            f"The original problem was:\n{prompt}\n\n"
            f"The logic solver determined the argument is: {status}.\n"
            "Explain to the user in clear natural language why this is the case."
        )

        explanation = self.llm.complete(explanation_prompt)
        print("\n[EXPLANATION]\n")
        print(explanation)


def main(llm_client: Optional[LLMClient] = None) -> None:
    parser = argparse.ArgumentParser(description="SymPrompt CLI")
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        help="Prompt to send. If omitted, read from stdin."
    )
    args = parser.parse_args()

    if llm_client is None:
        # You must construct your real client here, e.g.:
        # from openai import OpenAI
        # client = OpenAI(api_key="...")
        # llm_client = OpenAIClient(client)
        raise RuntimeError("You must pass an LLM client instance to main().")

    cli = SymPromptCLI(llm_client)

    if args.prompt:
        prompt = args.prompt
    else:
        print("Enter prompt (Ctrl-D to finish):")
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("No prompt provided.")
        return

    cli.run_once(prompt)


if __name__ == "__main__":
    main()
```

To use it, you just:

```bash
python -m symprompt.integration.cli -p "All mammals are animals. All cats are mammals. Therefore, all cats are animals."
```

(after wiring your real `llm_client` in `main`).

---

## 5. Quick smoke test: full stack

Once you’ve pasted in the pieces from previous messages plus this one, here’s a simple Python snippet to verify the *whole* pipeline without CLI:

```python
from symprompt.translation.pipeline import TranslationPipeline
from symprompt.core.compiler_z3 import solve_symil
from symprompt.integration.llm_client import OpenAIClient
from openai import OpenAI  # or your own lib

text = (
    "All mammals are animals. "
    "All cats are mammals. "
    "Therefore, all cats are animals."
)

openai_client = OpenAI(api_key="YOUR_KEY")
llm = OpenAIClient(openai_client)
pipeline = TranslationPipeline.from_llm_client(llm)

symil = pipeline.translate(text)
print("SymIL:", symil)

result = solve_symil(symil)
print("Solver status:", result["status"])
```

You should see `VALID` for that example if the prompts and parsing are working.

---

## 6. Hooking into OpenEvolve later

You already have `evaluate_translation_pipeline(pipeline, TINY_EXAMPLES)` from the last step. That function is exactly what the architecture calls the **fitness function** for evolving the translation layer. 

When you’re ready:

* Point OpenEvolve at `symprompt/translation/ontology_extractor.py` and `logical_translator.py` as evolvable code.
* Use `evaluate_translation_pipeline` as the evaluation entry point with `TINY_EXAMPLES` or a richer benchmark file.

For now, though, you’ve got:

* SymIL spec + validator
* Z3 compiler/runner
* Translation pipeline (preprocess → ontology → logic → SymIL)
* Tiny benchmark set
* Router + CLI for interactive personal use

That’s a clean, private SymPrompt‑Lite implementation you can start playing with and iterating on.
