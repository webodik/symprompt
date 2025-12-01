# SymPrompt Solvers Guide

This document explains the role of different solvers (Z3, Clingo, Scallop, VSA) in the SymPrompt pipeline.

## Overall Pipeline Flow

```
NL Prompt → TranslationPipeline → SymIL → Compiler → Solver → VALID/NOT_VALID/UNKNOWN
                                    │
                           ┌────────┼────────┐
                           ▼        ▼        ▼
                       fol_compiler asp_compiler scallop_compiler
                           │        │        │
                           ▼        ▼        ▼
                          Z3    Clingo    Scallop
```

## Solver Backends

### Z3 (SMT Solver)

**What it is:** Satisfiability Modulo Theories solver from Microsoft Research.

**Implementation:** `symprompt/reasoning/z3_runner.py` + `symprompt/reasoning/fol_compiler.py`

**Best for:**
- Math/arithmetic constraints (`equals`, `greater_than`, `sum`)
- Full first-order logic with quantifiers (`ForAll`, `Exists`)
- Complex nested formulas with `Implies`, `And`, `Not`

**How it works in SymPrompt:**

1. SymIL is compiled to Z3 expressions via `fol_compiler.py`
2. Creates an abstract `Entity` sort for all variables
3. Predicates become Z3 functions: `is_mammal(X) → Bool`
4. To prove query Q: adds `Not(Q)` and checks satisfiability
   - If UNSAT → Q is VALID (can't find counterexample)
   - If SAT → Q is NOT_VALID (found counterexample)

**Example translation:**

```python
# SymIL
ForAll X: is_mammal(X) → is_animal(X)

# Z3
ForAll([x], Implies(is_mammal(x), is_animal(x)))
```

**Code flow:**
```python
# z3_runner.py
context = symil_to_z3(symil)  # Compile to Z3 expressions
solver = z3.Solver()
solver.add(z3.Not(query_ref))  # Try to find counterexample
if solver.check() == z3.unsat:
    status = "VALID"  # No counterexample exists
```

---

### Clingo (ASP Solver)

**What it is:** Answer Set Programming solver for logic programs with stable model semantics.

**Implementation:** `symprompt/reasoning/clingo_runner.py` + `symprompt/reasoning/asp_compiler.py`

**Best for:**
- Planning problems (actions, preconditions, effects)
- Default reasoning and non-monotonic logic
- Rule-based inference with negation-as-failure

**How it works in SymPrompt:**

1. SymIL is compiled to ASP syntax via `asp_compiler.py`
2. Facts become ground atoms: `is_mammal(cat).`
3. Rules become Horn clauses: `is_animal(X) :- is_mammal(X).`
4. To prove query Q: runs two programs
   - Program 1: base + `:- not Q.` (Q must be true)
   - Program 2: base + `:- Q.` (Q must be false)
   - If only Program 1 has model → VALID
   - If only Program 2 has model → NOT_VALID

**Example translation:**

```python
# SymIL
Atom(pred="is_mammal", args=["cat"])
Rule: ForAll X: is_mammal(X) → is_animal(X)

# ASP
is_mammal(cat).
is_animal(X) :- is_mammal(X).
```

**Code flow:**
```python
# clingo_runner.py
base_program = symil_to_asp(symil)
program_valid = f"{base_program}\n:- not {query_str}.\n"
program_notvalid = f"{base_program}\n:- {query_str}.\n"

if has_model(program_valid) and not has_model(program_notvalid):
    return {"status": "VALID"}
```

---

### Scallop (Probabilistic Datalog)

**What it is:** Differentiable Datalog engine supporting probabilistic reasoning.

**Implementation:** `symprompt/reasoning/scallop_runner.py` + `symprompt/reasoning/scallop_compiler.py`

**Best for:**
- Simple syllogistic reasoning
- Categorical statements ("all A are B", "some A are B")
- Soft/probabilistic facts with confidence scores

**How it works:**
- Similar to Clingo but with support for weighted facts
- Uses Datalog semantics (subset of ASP)

---

### VSA (Vector Symbolic Architecture)

**What it is:** High-dimensional vector encoding for semantic similarity matching.

**Implementation:** `symprompt/reasoning/vsa_runner.py` + `symprompt/reasoning/vsa_encoder.py`

**Best for:**
- Semantic similarity tiebreaking
- Fast approximate reasoning
- When formal solvers return UNKNOWN

**How it works:**
- Encodes SymIL facts and query as high-dimensional vectors
- Computes cosine similarity between query and facts
- Returns similarity score used for tiebreaking in portfolio

---

## Domain Profile → Solver Selection

| Profile | Preferred Solver | Rationale |
|---------|------------------|-----------|
| `math` | Z3 | Full arithmetic, equality, inequalities |
| `syllogism` | Scallop | Simple categorical Horn clauses |
| `planning` | Clingo | Actions/effects need stable model semantics |
| `legal` | Clingo | Default reasoning for exceptions |
| `uncertain` | Scallop | Soft facts with confidence |

Profiles are defined in `symprompt/symil/profiles.py`.

---

## Portfolio Strategy

### Tier 1 (Fast Path)

Used for simpler problems where single solver suffices:

```python
# portfolio.py
if decision.tier == 1:
    if solver in {"asp", "clingo"}:
        result = run_asp(symil)
    elif solver in {"scallop", "datalog"}:
        result = run_scallop(symil)

    if result.get("status") != "UNKNOWN":
        return result
    # Fallback to Z3
    return solve_symil(symil)
```

### Tier 2 (Full Verification)

Runs ALL solvers in parallel for maximum confidence:

```python
# portfolio.py
z3_result = solve_symil(symil)
asp_result = run_asp(symil)
scallop_result = run_scallop(symil)
vsa_result = run_vsa(symil)

statuses = [z3, asp, scallop, vsa]

# Consensus voting
if statuses.count("VALID") >= 2:
    return {"status": "VALID"}
if statuses.count("NOT_VALID") >= 2:
    return {"status": "NOT_VALID"}

# VSA tiebreaker
if vsa_similarity > tiebreaker_threshold:
    return {"status": "VALID"}

# Final fallback
return z3_result
```

---

## Solver Comparison

| Aspect | Z3 | Clingo | Scallop | VSA |
|--------|----|----|---------|-----|
| **Logic** | First-order | Answer Set | Datalog | Semantic vectors |
| **Quantifiers** | Full support | Limited | Limited | N/A |
| **Negation** | Classical | Negation-as-failure | Stratified | N/A |
| **Arithmetic** | Full | Limited | No | N/A |
| **Speed** | Fast | Fast | Fast | Very fast |
| **Confidence** | High | High | Medium | Low |

---

## Adding a New Solver

1. Create compiler in `symprompt/reasoning/` (e.g., `new_compiler.py`)
2. Create runner in `symprompt/reasoning/` (e.g., `new_runner.py`)
3. Add to portfolio in `symprompt/reasoning/portfolio.py`
4. Update profile preferences in `symprompt/symil/profiles.py`
