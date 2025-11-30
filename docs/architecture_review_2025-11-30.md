# Architecture Review Findings

**Date:** 2025-11-30
**Reviewer:** Claude
**Scope:** Comparison of `docs/SymPrompt_Architecture.md`, `docs/SymPrompt_Architecture_v2.md`, and `docs/SymPrompt_dev_plan_v2.md` against actual implementation.

---

## 1. Missing Features / Incomplete Implementations

### A. LogicalTranslator missing `Not` and `Or` parsing
- **Location:** `symprompt/translation/logical.py:99-124`
- **Issue:** The `_formula_from_json` method handles `pred`, `implies`, `and`, `forall`, `exists` but NOT `not` or `or` JSON keys
- **Impact:** If LLM outputs JSON with `{"not": {...}}` or `{"or": [...]}`, parsing fails with `ValueError`
- **Note:** The FOL compiler (`fol_compiler.py`) correctly supports all formula types including `Not` and `Or`

### B. ConstraintMiner never implemented
- **Location:** Referenced in v1 architecture §4.2.1
- **Issue:** The implementation passes constraints through from LLM output but never mines implicit constraints from text
- **Impact:** Lower constraint completeness than architecture promises

### C. Notebook demo missing
- **Location:** Dev plan lists `symprompt/integration/notebook_demo.ipynb`
- **Issue:** File doesn't exist

### D. VSA backend very limited
- **Location:** `symprompt/reasoning/vsa_runner.py`
- **Issue:** Only supports Level 0 with atomic queries
- **Impact:** Architecture promises "analogical reasoning" and "mental simulation" but implementation is just similarity matching

### E. LLM-based routing classifier not implemented
- **Location:** v1 architecture §5.2 shows `self.classifier.predict(prompt) > 0.7`
- **Issue:** Actual SmartRouter only uses heuristic keyword matching

---

## 2. Logical Errors / Bugs

### A. Level 0 query enforcement inconsistency
- **Location:** `symprompt/symil/validator.py:70-73`
- **Issue:** Validator enforces L0 queries must be `Atom`, but LogicalTranslator might produce `ForAll`-wrapped queries
- **Impact:** Validation happens after translation, potential for inconsistent state

### B. Scallop rule syntax uses wrong operator
- **Location:** `symprompt/reasoning/scallop_runner.py:96`
- **Issue:** Generates `"{head_str} = {body_str}"` but Scallop Datalog uses `:-` not `=`
- **Impact:** Silent failures when scallopy is present

### C. VSA thresholds inconsistent
- **Locations:**
  - `vsa_runner.py:28-31`: uses `0.9` and `0.3` thresholds
  - `portfolio.py:65-68`: uses `0.8` for VSA tie-breaker
- **Impact:** Inconsistent behavior between direct VSA calls and portfolio

---

## 3. Inconsistencies Between Doc and Implementation

### A. Fitness function weights differ from spec
- **Doc (v2 architecture §6.2):** `0.50 * accuracy + 0.25 * latency + 0.15 * routing + 0.10 * syntactic`
- **Implementation:** Uses `tier_weighted_accuracy` = `0.6 * tier1 + 0.4 * tier2`

### B. Level 1 validator allows Or/Not in unexpected places
- **Location:** `symprompt/symil/validator.py:100-117`
- **Issue:** `check_horn` function doesn't explicitly reject `Or` or `Not`, relies on fall-through

### C. Profile default levels vs router decisions
- **Issue:** Profiles define `default_level` but SmartRouter computes `symil_level` independently in `features.py:61`

---

## 4. Duplicated Code

### A. Atom-to-string conversion (minor)
- `asp_compiler.py:8-10`: `_atom_to_asp`
- `scallop_compiler.py:8-10`: `_atom_to_scallop`
- Identical implementation

### B. Rule body parsing logic
- `logical.py:126-136`: `_rule_from_json`
- `scallop_runner.py:73-101`: `_rule_to_scallop`
- Both parse Implies/And/Atom structures similarly

---

## 5. Hardcoded Values

| Location | Value | Purpose |
|----------|-------|---------|
| `config.py:11` | `"openrouter/x-ai/grok-4.1-fast:free"` | Default LLM model |
| `run_translation_evolution.py:24` | Same model | Fallback model |
| `vsa_encoder.py:14`, `vsa_runner.py:11` | `1024` | VSA dimension |
| `vsa_runner.py:28-31` | `0.9`, `0.3` | Similarity thresholds |
| `portfolio.py:65` | `0.8` | VSA tie-breaker threshold |
| `z3_runner.py:11` | `1000` ms | Z3 timeout |
| `eval_pipeline.py:239-240` | `50.0`, `500.0` ms | Latency targets |
| `eval_pipeline.py:215-220` | Domain weights dict | Fixed evaluation weights |
| `features.py:61` | `40.0`, `0.2` | Complexity scaling factors |
| `verification.py:43` | `2` | Max refinement attempts |

---

## 6. Missing Tests

- `tests/test_profiles.py` - Dev plan mentions this but doesn't exist
- Missing coverage for `Or`/`Not` formula parsing
- Missing coverage for ASP compilation of constraints
- Missing coverage for Level 2 full FOL features

---

## 7. Phase Status vs Execution Status

| Phase | Doc Status | Actual |
|-------|------------|--------|
| Phase 1 – v2 Skeleton & Env | "in progress" | Complete |
| Phase 2 – SymIL Levels & Profiles | "partially implemented" | Missing `test_profiles.py` |
| Phase 3 – Reasoning Backends | Documented as complete | Scallop has syntax bug |
| Phase 4 – Translation v2 | Complete | Missing `Not`/`Or` parsing |
| Phase 5 – Smart Router | Complete | Missing LLM classifier |
| Phase 6 – OpenEvolve v2 | Complete | Fitness weights differ from doc |
| Phase 7 – CLI & Notebook | "notebook demo not started" | Correct |
| Phase 8 – Hardening & Docs | "not started" | Correct |

---

## Recommendations (Priority Order)

1. **[HIGH]** Fix LogicalTranslator to parse `not` and `or` JSON keys
2. **[HIGH]** Fix Scallop rule syntax from `=` to `:-`
3. **[MEDIUM]** Add missing test file `tests/test_profiles.py`
4. **[MEDIUM]** Consolidate hardcoded values into `config.py`
5. **[LOW]** Make VSA thresholds consistent across files
6. **[LOW]** Update Execution Status in dev plan to reflect actual state
