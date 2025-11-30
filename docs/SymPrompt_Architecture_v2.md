

# **SymPrompt v2.0**

A Two-Tier Framework for NL-to-Neuro-Symbolic Translation

with Progressive SymIL and Evolvable Routing

*Technical Architecture Document*

OpenEvolve Integration for Algorithm Discovery

Version 2.0 (Revised)

November 2025

# **Revision Summary (v1.0 → v2.0)**

This revision addresses key architectural concerns identified in design review, specifically: latency vs complexity trade-offs, IL translation difficulty, benchmark overfitting risks, and the need for stronger neuro-symbolic integration.

## **Key Changes in v2.0**

| Change | Rationale |
| ----- | ----- |
| **Two-Tier Pipeline** | Tier 1 (fast path) for simple queries; Tier 2 (full SymPrompt) for complex reasoning. Addresses latency concerns. |
| **Progressive SymIL** | Three levels (L0, L1, L2) of IL complexity. Start simple, escalate if needed. Reduces translation difficulty. |
| **Evolvable Router** | Router is now an OpenEvolve target, not just heuristics. Discovers optimal routing strategies. |
| **SymIL Profiles** | Domain-specific IL configurations (legal, math, planning). Evolvable alongside translation. |
| **First-Class Neuro-Symbolic** | Scallop and VSA backends promoted to primary targets. Enables soft/differentiable reasoning. |
| **Reduced Solver Portfolio** | Default to 2 backends (SMT \+ ASP/Scallop). Full portfolio only for Tier 2 high-stakes queries. |

# **Table of Contents**

# **1\. Executive Summary**

SymPrompt v2.0 is a **two-tier neuro-symbolic framework** for translating natural language prompts into formal symbolic representations during LLM inference. The framework addresses a fundamental trade-off: *high accuracy requires rich intermediate representations, but rich ILs are harder to translate and slower to process*.

The key insight is that most queries don't need the full power of the system. By introducing a **fast path (Tier 1\)** for simple reasoning and a **full path (Tier 2\)** for complex problems, we achieve both low latency and high capability.

## **1.1 Core Innovations**

* **Two-Tier Architecture:** Fast path (\<50ms) for 70% of queries; full pipeline for complex reasoning  
* **Progressive SymIL:** Three complexity levels (L0→L1→L2) with automatic escalation  
* **Evolvable Everything:** OpenEvolve optimizes translator, router, AND SymIL profiles  
* **True Neuro-Symbolic:** Scallop (differentiable) and VSA (vector-symbolic) as first-class backends

## **1.2 Performance Targets (Revised)**

| Metric | Tier 1 Target | Tier 2 Target | Current SOTA |
| ----- | :---: | :---: | :---: |
| P95 Latency | **\<50ms** | **\<500ms** | 500-2000ms |
| Syntactic Validity | **\>95%** | **\>90%** | \~75% |
| Logical Equivalence | **\>80%** | **\>88%** | \~70% |
| Query Coverage | **\~70%** | **\~30%** | N/A |

# **2\. Two-Tier Architecture**

The fundamental insight: **most reasoning queries don't need the full machinery**. Simple syllogisms, basic math, and toy puzzles can be handled with a lightweight translator and single solver. Only complex multi-hop reasoning, constraint satisfaction, or high-stakes verification needs the full SymIL pipeline.

## **2.1 Tier Overview**

| Aspect | Tier 1: Fast Path | Tier 2: Full SymPrompt |
| ----- | ----- | ----- |
| **IL Level** | SymIL-Lite (L0 or L1 only) | Full SymIL (L0→L1→L2 escalation) |
| **Solvers** | Single backend (Z3 or Scallop) | Portfolio (Z3 \+ Clingo \+ Scallop \+ VSA) |
| **Latency** | P95 \< 50ms | P95 \< 500ms (relaxed for accuracy) |
| **Use Cases** | Syllogisms, simple math, basic constraints, fact checking | Multi-hop reasoning, planning, abduction, high-stakes verification |
| **Evolution** | Hand-designed \+ mild OpenEvolve tuning | Full OpenEvolve optimization |

## **2.2 Architecture Diagram**

| ┌─────────────────────────────────────────────────────────────────┐ │                     SYMPROMPT v2.0 ARCHITECTURE                 │ ├─────────────────────────────────────────────────────────────────┤ │                                                                 │ │  ┌─────────────────┐                                           │ │  │   NL Prompt     │                                           │ │  └────────┬────────┘                                           │ │           ▼                                                    │ │  ┌─────────────────┐     ┌──────────────────┐                  │ │  │  SMART ROUTER  │◀────│   OpenEvolve     │                  │ │  │   (Evolvable)   │     │   (Evolves All)  │                  │ │  └───────┬─────────┘     └────────┬─────────┘                  │ │          │                        │                            │ │    ┌─────┴─────┐                  │ Feedback                   │ │    ▼           ▼                  │                            │ │ ╔═══════════╗  ╔═══════════════╗  │                            │ │ ║  TIER 1   ║  ║    TIER 2     ║  │                            │ │ ║ Fast Path ║  ║ Full SymPrompt║  │                            │ │ ╠═══════════╣  ╠═══════════════╣  │                            │ │ ║SymIL-Lite ║  ║Progressive    ║◀─┘                            │ │ ║ (L0/L1)   ║  ║SymIL (L0→L2) ║                               │ │ ╠═══════════╣  ╠═══════════════╣                               │ │ ║ Single    ║  ║ Multi-Solver  ║                               │ │ ║ Solver    ║  ║ Portfolio     ║                               │ │ ║(Z3/Scallop║  ║(Z3+Clingo+    ║                               │ │ ║   )       ║  ║ Scallop+VSA)  ║                               │ │ ╚═════╤═════╝  ╚═══════╤═══════╝                               │ │       │                │                                       │ │       └────────┬───────┘                                       │ │                ▼                                               │ │       ┌─────────────────┐                                      │ │       │  LLM Integration │───▶ Augmented Response              │ │       └─────────────────┘                                      │ └─────────────────────────────────────────────────────────────────┘ |
| :---: |

## **2.3 Tier Selection Logic**

class SmartRouter:  \# Evolvable by OpenEvolve

    def route(self, prompt: str, context: Context) \-\> RoutingDecision:

        features \= self.extract\_features(prompt)

        

        \# Fast bypass: pure CoT is sufficient

        if features.complexity \< 0.3 and features.logical\_depth \< 2:

            return RoutingDecision.BYPASS\_SYMBOLIC

        

        \# Tier 1: Simple symbolic (syllogisms, basic math)

        if features.complexity \< 0.6 and features.constraint\_count \< 3:

            return RoutingDecision(

                tier=1,

                symil\_level=0 if features.is\_factual else 1,

                solver='z3' if features.is\_arithmetic else 'scallop',

                profile=self.select\_profile(features)

            )

        

        \# Tier 2: Full SymPrompt (complex reasoning)

        return RoutingDecision(

            tier=2,

            symil\_level=0,  \# Start low, escalate if needed

            solver='portfolio',

            profile=self.select\_profile(features)

        )

# **3\. Progressive SymIL**

SymIL v2.0 introduces **three progressive complexity levels**. Instead of requiring the translator to emit the full IL structure for every query, it starts simple and escalates only when solver feedback indicates missing structure.

## **3.1 Level Definitions**

| Level | Structure | Example Use | Translation Difficulty |
| :---: | ----- | ----- | ----- |
| **L0** | facts \+ query only | Fact checking, simple Q\&A | Easy (just extract entities) |
| **L1** | facts \+ query \+ simple rules (∀x P→Q) | Syllogisms, implications | Medium (quantifiers, scope) |
| **L2** | full: ontology \+ rules \+ constraints \+ nested quantifiers | Planning, abduction, complex proof | Hard (full semantic alignment) |

## **3.2 Level Grammar Specifications**

### **3.2.1 Level 0 (SymIL-Lite)**

SymIL\_L0 ::= {

  "level": 0,

  "facts": \[Atom\],          // Ground facts only

  "query": Atom | Boolean   // Single predicate or yes/no

}

// Example: "Is Paris the capital of France?"

{

  "level": 0,

  "facts": \[{"pred": "capital\_of", "args": \["Paris", "France"\]}\],

  "query": {"pred": "capital\_of", "args": \["Paris", "France"\]}

}

### **3.2.2 Level 1 (Horn Clauses)**

SymIL\_L1 ::= SymIL\_L0 \+ {

  "rules": \[HornClause\]     // ∀x (P₁ ∧ P₂ → Q)

}

// Example: "All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?"

{

  "level": 1,

  "facts": \[{"pred": "cat", "args": \["Whiskers"\]}\],

  "rules": \[{

    "forall": "X",

    "if": {"pred": "cat", "args": \["X"\]},

    "then": {"pred": "mammal", "args": \["X"\]}

  }\],

  "query": {"pred": "mammal", "args": \["Whiskers"\]}

}

### **3.2.3 Level 2 (Full SymIL)**

SymIL\_L2 ::= SymIL\_L1 \+ {

  "ontology": Ontology,     // Explicit predicate/type signatures

  "constraints": \[Constraint\],  // Hard constraints, integrity rules

  "rules": \[Formula\]        // Full FOL (∀∃ nesting, negation)

}

## **3.3 Automatic Level Escalation**

If a lower level fails (solver returns UNKNOWN or error), the system automatically escalates:

def translate\_with\_escalation(prompt, max\_level=2):

    for level in range(max\_level \+ 1):

        symil \= translate(prompt, target\_level=level)

        result \= solver.execute(symil)

        

        if result.status \== 'SUCCESS':

            return result

        elif result.status \== 'NEEDS\_MORE\_STRUCTURE':

            continue  \# Escalate to next level

        elif result.status \== 'SYNTAX\_ERROR':

            symil \= refine\_translation(prompt, symil, result.error)

    

    return Result(status='FAILED', reason='max\_level\_exceeded')

# **4\. SymIL Profiles (Domain-Specific Configurations)**

Different reasoning domains benefit from different IL configurations. SymIL Profiles define domain-specific predicate vocabularies, allowed constructs, and preferred solver backends. **Profiles themselves are evolvable by OpenEvolve.**

## **4.1 Profile Structure**

SymILProfile ::= {

  "name": string,           // e.g., "legal", "math", "planning"

  "predicate\_vocabulary": \[PredicateSpec\],  // Domain predicates

  "allowed\_constructs": \[Construct\],        // Subset of SymIL

  "preferred\_solver": SolverType,

  "default\_level": 0 | 1 | 2,

  "translation\_hints": \[Hint\]               // Domain-specific prompts

}

## **4.2 Example Profiles**

| Profile | Predicate Examples | Preferred Solver | Construct Restrictions |
| ----- | ----- | ----- | ----- |
| **syllogism** | is\_a, has\_property, subset\_of | Scallop (fast Datalog) | L1 max, Horn clauses only |
| **math** | equals, greater\_than, sum, product | Z3 (arithmetic theories) | L0-L1, numeric constraints |
| **planning** | action, precondition, effect, goal | Clingo (ASP planning) | L2, non-monotonic reasoning |
| **legal** | obligation, permission, prohibition | Clingo (defeasible logic) | L2, deontic modalities |
| **uncertain** | probably, likely, confidence | Scallop (probabilistic) | L1-L2, soft facts with weights |

# **5\. First-Class Neuro-Symbolic Backends**

v2.0 elevates **Scallop** and **Vector Symbolic Architectures (VSA)** to first-class backends, enabling true neuro-symbolic reasoning: soft/differentiable logic and vector-space symbolic computation.

## **5.1 Backend Portfolio**

| Backend | Type | Strengths | Best For |
| ----- | ----- | ----- | ----- |
| **Z3** | Hard Logic (SMT) | Sound, complete for decidable theories | Verification, arithmetic, SAT |
| **Clingo** | Hard Logic (ASP) | Non-monotonic, defaults, constraints | Planning, abduction, defaults |
| **Scallop** | Neuro-Symbolic (Diff) | Differentiable, probabilistic, fast | Uncertain facts, neural integration |
| **VSA Engine** | Neuro-Symbolic (Vector) | Compositional, analogical reasoning | Analogy, mental simulation |

## **5.2 Scallop Integration**

Scallop provides differentiable Datalog with provenance semirings, enabling:

* **Probabilistic facts:** Attach confidence scores to extracted facts  
* **Neural predicates:** Integrate neural network outputs as soft facts  
* **Gradient flow:** Backpropagate through reasoning for end-to-end training

\# SymIL → Scallop compilation with probabilities

def compile\_to\_scallop(symil: SymIL) \-\> ScallopProgram:

    prog \= ScallopProgram()

    

    for fact in symil.facts:

        prob \= fact.get('confidence', 1.0)

        prog.add\_fact(fact.pred, fact.args, probability=prob)

    

    for rule in symil.rules:

        prog.add\_rule(compile\_rule(rule))

    

    return prog

## **5.3 VSA Engine Integration**

Vector Symbolic Architectures encode symbolic structures in high-dimensional vectors, enabling:

* **Compositional binding:** Bind predicates and arguments into single vectors  
* **Analogical reasoning:** "A is to B as C is to ?" via vector arithmetic  
* **Mental simulation:** Run forward inference as vector operations

\# SymIL → VSA encoding

def compile\_to\_vsa(symil: SymIL, codebook: VSACodebook) \-\> VSAState:

    state \= VSAState(dim=10000)

    

    for fact in symil.facts:

        \# Bind predicate with arguments

        vec \= codebook.get(fact.pred)

        for i, arg in enumerate(fact.args):

            vec \= vsa\_bind(vec, vsa\_permute(codebook.get(arg), i))

        state.memory \= vsa\_bundle(state.memory, vec)

    

    return state

# **6\. OpenEvolve Integration (Expanded)**

In v2.0, OpenEvolve optimizes **three evolvable targets**: the Translation Pipeline (as before), the Smart Router, and SymIL Profiles. This enables holistic system optimization.

## **6.1 Evolvable Components**

| Target | What Evolves | Fitness Signal |
| ----- | ----- | ----- |
| **TranslationPipeline** | Prompt templates, parsing strategies, quantifier handling | Syntactic validity, logical equivalence, solver success |
| **SmartRouter** | Feature thresholds, tier selection rules, profile selection | End-to-end accuracy × 1/latency (Pareto) |
| **SymILProfiles** | Predicate vocabularies, construct subsets, solver preferences | Per-domain accuracy on benchmark slices |

## **6.2 Multi-Objective Fitness Function**

def fitness(system: SymPromptSystem, benchmarks: BenchmarkSuite) \-\> float:

    results \= evaluate\_all(system, benchmarks)

    

    \# Tier-weighted accuracy

    tier1\_acc \= results.tier1.accuracy \* results.tier1.coverage

    tier2\_acc \= results.tier2.accuracy \* results.tier2.coverage

    accuracy \= 0.6 \* tier1\_acc \+ 0.4 \* tier2\_acc

    

    \# Latency score (Tier 1 must be fast)

    latency\_score \= (

        0.7 \* score\_latency(results.tier1.p95\_latency, target=50) \+

        0.3 \* score\_latency(results.tier2.p95\_latency, target=500)

    )

    

    \# Routing quality (did router choose correctly?)

    routing\_score \= evaluate\_routing\_decisions(results)

    

    \# Combined fitness

    return (

        0.50 \* accuracy \+

        0.25 \* latency\_score \+

        0.15 \* routing\_score \+

        0.10 \* results.syntactic\_validity

    )

## **6.3 Anti-Overfitting Measures**

To prevent benchmark overfitting, we implement:

1. **Held-out wild prompts:** 20% of fitness evaluation uses real user prompts not in evolution loop  
2. **Synthetic noise:** Add paraphrases, typos, and domain-shifted versions of benchmark examples  
3. **Cross-benchmark validation:** Train on subset A, validate on subset B, rotate  
4. **Complexity penalty:** Penalize overly complex translation pipelines (regularization)

# **7\. Implementation Roadmap (Revised)**

## **7.1 Phase 1: SymPrompt-Lite (Weeks 1-4)**

**Goal:** Implement Tier 1 fast path as standalone, usable system.

* Implement SymIL L0 and L1 only  
* Single solver backend (Z3 for arithmetic, Scallop for relational)  
* Hand-designed translation prompts (no evolution yet)  
* Basic heuristic router  
* **Success Criteria:** P95 \< 50ms, \>75% accuracy on ProofWriter-easy

## **7.2 Phase 2: OpenEvolve \+ Profiles (Weeks 5-8)**

**Goal:** Integrate OpenEvolve, define SymIL profiles, begin evolution.

* Set up OpenEvolve infrastructure  
* Define 3-5 initial SymIL profiles (syllogism, math, planning)  
* Evolve translation pipeline for Tier 1  
* Begin evolving router  
* **Success Criteria:** \+10% accuracy over hand-designed baseline

## **7.3 Phase 3: Full Tier 2 (Weeks 9-12)**

**Goal:** Add full SymIL (L2), multi-solver portfolio, progressive escalation.

* Implement SymIL L2 with full ontology/constraints  
* Add Clingo (ASP) and VSA backends  
* Implement automatic level escalation  
* Multi-solver consensus voting  
* **Success Criteria:** \>85% on FOLIO, \>90% on ProofWriter

## **7.4 Phase 4: Production Hardening (Weeks 13-16)**

**Goal:** Optimize latency, add LLM integration modes, prepare for release.

* Optimize critical path (caching, parallelization)  
* Implement pre/post/hybrid LLM integration modes  
* Run evolution on wild prompts for generalization  
* Documentation, API design, release  
* **Success Criteria:** Production-ready, \<50ms Tier 1, \<500ms Tier 2

# **8\. Conclusion**

SymPrompt v2.0 addresses the key tensions in neuro-symbolic NL translation: *richness vs. translatability*, *accuracy vs. latency*, and *benchmark performance vs. real-world generalization*.

The key architectural decisions:

1. **Two-Tier Design:** Fast path for 70% of queries, full pipeline for complex reasoning  
2. **Progressive SymIL:** Start simple (L0), escalate only when needed (L1→L2)  
3. **Evolvable Everything:** Translation, routing, and profiles all optimized by OpenEvolve  
4. **True Neuro-Symbolic:** Scallop and VSA as first-class backends for soft/differentiable reasoning  
5. **Domain Profiles:** Specialized IL configurations for different reasoning types

This architecture provides a practical path from SymPrompt-Lite (immediately usable) to full SymPrompt (research-grade). The two-tier design ensures the system is **useful from day one** while evolution discovers increasingly sophisticated translation strategies over time.

**Next Steps:** Implement SymPrompt-Lite (Phase 1\) to validate the core architecture before investing in full OpenEvolve integration.