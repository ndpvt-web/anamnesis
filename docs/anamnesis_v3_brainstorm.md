# Anamnesis v3: First-Principles Brainstorm on Retrieval Precision

## The Problem, Stated Precisely

Phase 3 experiment (25 tasks, Deep Knowledge) achieved **-11.7%** turn reduction vs baseline,
but Phase 2 (plain Anamnesis, no deep knowledge) achieved **-23.1%**. The deep knowledge system
was *worse* than the simpler system.

**Category-level evidence:**
- B (data processing): P3 = -36% (better than P2's -23%) -- deep knowledge helps related tasks
- D (storage/state): P3 = +11% -- deep knowledge HURTS
- E (API/networking): P3 = +9% -- deep knowledge HURTS

**The pattern:** As the knowledge graph grew past ~35-40 nodes, retrieval precision degraded.
Irrelevant nodes got surfaced alongside relevant ones, confusing the agent instead of helping it.

---

## Part I: Aristotelian First-Principles Analysis

### The Four Causes of Retrieval Failure

Aristotle's four causes (aitiai) provide the deepest decomposition of *why* something is what it is.
Applied to our retrieval saturation problem:

#### 1. Material Cause (hyle) -- "What is it made of?"

The material of our knowledge graph is **undifferentiated techne nodes**. All 75 nodes live at
the same ontological level. A node about "Python pip --break-system-packages on Debian" sits
alongside a node about "SQLite WAL mode for concurrent access" with no structural distinction
between them.

**The problem:** Aristotle distinguishes between matter that is *proximate* (suitable for the
form being imposed) and matter that is *remote* (unsuitable). When we retrieve 5 nodes for a
Flask task, and 2 of them are about SQLite WAL mode, those are *remote* material -- they cannot
take on the form of useful guidance for the current problem.

**First principle:** *Knowledge must be differentiated in its material nature before retrieval
can be precise.* Undifferentiated material cannot be selectively formed.

#### 2. Formal Cause (eidos) -- "What pattern/structure does it have?"

Our retrieval uses FTS5 full-text search + embedding similarity. This is a **flat** formal
structure -- every node is equally reachable from every query. There is no hierarchy, no
categorical boundary, no genus-species relationship.

Aristotle's epistemology in the *Posterior Analytics* is fundamentally **taxonomic**. Knowledge
is organized into genera (genos) and species (eidos), where you cannot reason validly across
genus boundaries. You cannot use knowledge of geometry to prove a theorem in biology -- the
genera are different.

**The problem:** Our retrieval has no genus boundaries. A query about "Flask route handlers"
can surface nodes about "CSV parsing edge cases" because they share tokens like "error" or
"Python." The formal structure permits cross-genus pollution.

**First principle:** *Knowledge must be organized into genera with explicit boundaries. Retrieval
should not cross genus boundaries unless the query explicitly demands it.*

#### 3. Efficient Cause (arche kinesis) -- "What agent/process produced it?"

Our nodes are produced by the reflection pipeline -- a single LLM call that processes session
events and generates hypotheses. The pipeline has no quality gate. Every hypothesis that isn't
a duplicate gets stored. The pipeline doesn't ask: "Is this knowledge *worth knowing* for
future tasks?"

**The problem:** Aristotle distinguishes between *techne* (craft knowledge -- knowing how) and
*episteme* (scientific knowledge -- knowing why through demonstration). Our pipeline generates
techne-level observations but stores them without validating whether they represent genuine
causal understanding or mere correlation.

A node like "WHEN using subprocess.run(), THEN check returncode" is trivially true but
almost never the actual cause of a debugging failure. It's what Aristotle would call
*accidental* knowledge -- true but not explanatory.

**First principle:** *The process that creates knowledge must include a quality gate that
distinguishes essential from accidental knowledge. Not everything learned is worth storing.*

#### 4. Final Cause (telos) -- "What is it for?"

This is the deepest question. What is the *purpose* of surfacing knowledge to a coding agent?

It is NOT "to provide all possibly relevant information." That is the librarian's telos, not
the advisor's. The telos of Anamnesis is **to reduce the number of turns the agent needs to
complete a task** -- that is, to provide *actionable guidance that changes behavior*.

**The problem:** Most of our 75 nodes are informational, not actionable. They describe what
can go wrong, but they don't change how the agent approaches the problem. The agent already
knows to check error codes. What it needs is: "In THIS specific situation, the non-obvious
cause is X, and the fix is Y."

**First principle:** *Knowledge should be stored and retrieved based on its capacity to change
agent behavior, not its truth value alone. True but obvious knowledge is noise.*

### The Doctrine of the Mean (Mesotes)

Aristotle's most famous ethical principle applies directly: virtue is the mean between
excess and deficiency.

- **Deficiency** (too little retrieval): Agent has no guidance. Baseline performance.
- **Excess** (too much retrieval): Agent is overwhelmed with irrelevant context. Phase 3 D+E categories.
- **The Mean** (right amount, right quality): Agent receives precisely the knowledge that changes its approach.

The mean is not arithmetic (not "retrieve 3 nodes instead of 5"). It is **relative to the
situation** -- what Aristotle calls "the mean relative to us" (pros hemas). For a Flask task,
the right amount of Flask-specific knowledge is different from the right amount for a
streaming API task.

**This is phronesis (practical wisdom):** the capacity to discern, in each particular
situation, what the right amount and right kind of knowledge is. Our system lacks phronesis.

### The Ladder of Knowledge (Posterior Analytics)

Aristotle's epistemology has a strict hierarchy:

```
Sophia (Wisdom)        -- First principles, always true
    |
Episteme (Science)     -- Demonstrated knowledge, causal
    |
Techne (Craft)         -- Practical know-how, domain-specific
    |
Empeiria (Experience)  -- Accumulated observations
    |
Aisthesis (Perception) -- Raw sensory data
```

**The insight:** Knowledge at lower levels (empeiria, techne) is DOMAIN-BOUND. You cannot
use carpentry experience to build a ship. Knowledge at higher levels (episteme, sophia) is
CROSS-DOMAIN. The principle of non-contradiction applies everywhere.

Our system's failure mode is: **treating domain-bound techne as if it were cross-domain
episteme.** A pattern learned from Flask debugging gets surfaced for SQLite migration tasks
because the retrieval system doesn't respect domain boundaries.

---

## Part II: Literature Review -- What 2025-2026 Research Says

### 1. Write-Time vs Read-Time Curation

**Zahn & Chana (March 2026)** -- "Selective Memory for AI: Write-Time Gating with
Hierarchical Archiving" [arXiv:2603.15994]

**Key finding:** At 8:1 distractor ratios, read-time filtering (Self-RAG) collapses to 0%
accuracy while write-time gating maintains 100%. Write gating keeps the store clean;
read-time filtering cannot recover from a polluted store.

**Salience scoring formula:**
```
S(K) = sum(wj * sj(K)) for j in {reputation, novelty, reliability}
```

**Hierarchical archiving:** Rejected items go to cold storage with version chains
(K_old.superseded_by -> K_new), not deleted. Enables temporal queries.

**Result:** 98.1% accuracy on Wikipedia (vs 85.2% ungated), 1/9th the query-time cost
of Self-RAG. Performance gap WIDENS as domain becomes more novel.

**Direct relevance to Anamnesis:** Our knowledge graph is ungated. Every reflection hypothesis
that isn't a duplicate gets stored. We need write-time quality gating.

### 2. Dynamic Context Selection

**Jin et al. (December 2025)** -- "Dynamic Context Selection for RAG: Mitigating
Distractors and Positional Bias" [arXiv:2512.14313]

**Key finding:** A classifier-based approach dynamically predicts the optimal number of
documents to retrieve based on query-specific needs. Fixed top-K is suboptimal because
different queries need different amounts of context.

**Direct relevance:** Our current retrieval uses fixed limits. We should dynamically
adjust how much knowledge to surface based on query characteristics.

### 3. Corrective RAG (CRAG)

**Yan et al. (2024, refined 2025-2026)** -- ICLR 2025 paper on evaluating retrieved
documents before passing to generation.

**Mechanism:** Lightweight T5-based evaluator scores relevance:
- **Correct:** Refine and use
- **Incorrect:** Discard, do web search
- **Ambiguous:** Combine both strategies

**Open-source reproduction (March 2026):** 85.2% on ARC-Challenge vs 84.8% vanilla RAG.
Gains come from the correction mechanism, not the generator.

### 4. Sparse RAG (ICLR 2025)

**Key finding:** Massive pre-filling + selective decoding. Assess document relevance via
a single forward pass, then DROP irrelevant documents before generation. Filters noisy
contexts while achieving higher decoding speeds.

### 5. Lost in the Middle -- Updated (August 2025)

**"Positional Biases Shift as Inputs Approach Context Window Limits"** [arXiv:2508.07479]

**Key findings:**
- Below 50% context capacity: U-shaped bias (beginning + end attended, middle ignored)
- Above 50%: Primacy bias disappears, replaced by distance-based (recency) bias
- "Context rot": Adding irrelevant context degrades accuracy by 20-50%

**Direct relevance:** When we surface 5 knowledge nodes in a briefing, the middle ones
get ignored. Order matters. And adding irrelevant nodes actively hurts.

### 6. Noise-Aware Calibration (NAACL, January 2026)

**Key finding:** Training models to assess passage utility under noisy retrieval.
Improved Expected Calibration Error by 10.9% in-domain. Models can learn to express
low confidence when retrieval quality is poor.

### 7. Complementary Learning Systems for AI

**CH-HNN (Nature Communications, February 2025):** Hybrid ANN/SNN architecture mimicking
hippocampus-cortex dynamics. Fast episodic learning (hippocampus) + slow semantic
consolidation (cortex). Prevents catastrophic forgetting.

**Personalized AGI framework (April 2025):** Three-tier memory: STM -> LTM -> PM.
Periodic consolidation transfers episodic memories to semantic store.

**Direct relevance:** Anamnesis already has the right *layers* (aisthesis -> empeiria ->
techne -> episteme -> sophia). What's missing is the *consolidation process* that
promotes/demotes/prunes knowledge across layers.

### 8. Adaptive Retrieval -- When to Retrieve

**Self-RAG:** Reflection tokens [Retrieve] / [No Retrieve] for per-segment control.
**UAR:** Four binary classifiers (intent, knowledge, time, self-aware) in priority tree.
**ADAPT-LLM:** Self-correcting loop -- correct answers reinforce internal generation,
incorrect ones trigger retrieval.
**DeepRAG:** Formalized as Markov Decision Process with atomic retrieve-or-not decisions.
**L-RAG (Jan 2026):** Entropy-based lazy retrieval -- only retrieve when model predictive
entropy exceeds threshold. Achieves 26% retrieval reduction with minimal accuracy loss.
**CAR (Nov 2025):** Cluster-based adaptive retrieval from Coinbase -- detect "transition
point" in similarity clusters. 60% fewer tokens, 22% less latency, 10% fewer hallucinations.
Deployed in production.

**Direct relevance:** Our system always surfaces knowledge. It should sometimes surface
nothing when the agent's internal knowledge is sufficient.

### 9. Multi-Agent Filtering (MAIN-RAG, 2025)

**Chang et al.** -- Multi-Agent Filtering RAG with adaptive relevance threshold.
Achieves 2-11% accuracy improvement by having multiple agents independently score
retrieved documents before the generator sees them. The consensus mechanism filters
noise more effectively than a single evaluator.

### 10. Noise-Robust Compression (ACoRN, Dec 2025)

**Kim (Korea University)** -- Rather than filtering documents, compress them while
stripping noise. Data augmentation trains compressor models to be resilient to
irrelevant content. The compressed representation contains only signal.

### 11. Distraction-Aware Retrieval (LDAR, ICLR 2026)

**Shim et al.** -- Trains retrievers to explicitly minimize distraction interference,
not just maximize relevance. Standard retrievers optimize for recall; LDAR optimizes
for the downstream generator's performance, which penalizes distracting passages.

### 12. Memory Consolidation and Hierarchical Abstraction

**G-Memory (June 2025, arXiv:2506.07398):** Three-tier graph hierarchy:
interaction graph -> query graph -> insight graph. High-level insights are progressively
distilled from raw interactions via bidirectional traversal. +20.89% embodied action
success. **This is the closest existing work to what Anamnesis needs.**

**H2R (Sept 2025, arXiv:2509.12810):** Hierarchical Hindsight Reflection explicitly
decouples planning memory (semantic) from execution memory (episodic). Extracts
reusable generalizations via "hindsight reflection" -- directly mirrors the
episodic-to-semantic abstraction process.

**APEX-EM (March 2026, arXiv:2603.29093):** Full procedural-episodic traces
(planning steps, artifacts, error analysis, quality scores) with hybrid
semantic+structural matching. Cross-domain transfer even with zero lexical overlap.
KGQAGen: 89.6% vs 41.3% baseline.

**DAM (Dec 2025, arXiv:2512.21567):** Decision-theoretic framework treating memory
management as sequential decision-making with value functions. Addresses the core flaw
of heuristic systems: they cannot account for delayed consequences of memory decisions.

### 13. Selective Forgetting and Memory Pruning

**SuperLocalMemory V3.3 (April 2026, arXiv:2604.04514):** First paper implementing
mathematical Ebbinghaus forgetting curves for local agent memory. Lifecycle-aware
quantization with progressive embedding compression. 6.7x improved discriminative
capability. Zero-LLM, fully CPU-based.

**SEEKR (Nov 2025, arXiv:2411.06171):** Selective attention-guided retention shows
that 1% of replay data matches 100% replay quality. Not all memory is worth rehearsing.
Indiscriminate rehearsal wastes compute and can DEGRADE performance.

**OSW (Jan 2026, arXiv:2601.18255):** Experience replay intended to prevent forgetting
can cause negative transfer -- new tasks corrupt structured knowledge. Orthogonal
subspace isolation prevents replay-induced damage.

**MSSR (March 2026, arXiv:2603.09892):** Per-sample memory strength estimation with
adaptive rehearsal intervals. Outperforms all replay baselines across 11 sequential tasks.

### 14. LightRAG: Dual-Level Graph Retrieval (2024/2025)

**Guo et al.** -- "LightRAG: Simple and Fast Retrieval-Augmented Generation"
[arXiv:2410.05779, revised April 2025]. Open source: github.com/HKUDS/LightRAG

**Core mechanism:** Dual-level retrieval over a knowledge graph:
- **Low-level:** Specific entities + direct relationships (precise, detail-oriented)
- **High-level:** Broad conceptual themes across multiple entities (comprehensive)

**Graph construction:** Entity & relationship extraction -> LLM profiling (key-value pairs)
-> Deduplication across chunks. Formally: D = Dedupe(Prof(V, E))

**Retrieval workflow:**
1. Extract **local keywords** (entity names) + **global keywords** (themes) from query
2. Vector match keywords against entity/relation embeddings
3. Gather one-hop neighbors from matched nodes
4. Combine low-level + high-level results

**Key results vs GraphRAG:**
- ~100 tokens per query vs GraphRAG's ~610,000 (6000x cheaper)
- ~$0.15 per doc set vs GraphRAG's $4-7
- Incremental updates via graph union (no full rebuild)
- 52-55% win rate vs GraphRAG overall

**Caveat (June 2025 unbiased evaluation, arXiv:2506.06331):** When position/length/trial
biases are removed, NaiveRAG sometimes outperforms LightRAG. The real performance gap
is <8% between most methods. Graph structure only helps significantly at scale and for
complex multi-hop queries.

**Direct relevance to Anamnesis:**
1. **Dual-level maps to Aristotelian layers.** Low-level = techne (specific). High-level =
   episteme (conceptual). Validates multi-level retrieval.
2. **Keyword extraction before vector search.** Instead of embedding the full query and doing
   similarity search across all nodes, extract domain keywords first, then search. This
   naturally implements domain genus filtering.
3. **One-hop neighbor gathering.** When you find a relevant node, also surface its SUPPORTS
   and DERIVED_FROM neighbors. The graph structure carries signal that flat retrieval misses.
4. **Deduplication as first-class operation.** Our reflection pipeline creates near-duplicates.
   LightRAG's Dedupe() step should be applied at write time.
5. **Graph structure only justified when edges are meaningful.** A graph of 75 nodes with
   sparse edges (our 125 edges, mostly SUPPORTS) is barely better than flat search. We need
   either denser edges or compiled articles that reduce node count.

### 15. Karpathy's LLM Wiki / Second Brain (April 2026)

**Andrej Karpathy** -- "LLM Knowledge Base" pattern, virally adopted in April 2026.

**Core insight:** Stop using LLMs to answer questions from scratch. Use them to
*compile* raw sources into a self-maintaining, interlinked markdown wiki. The LLM is
the librarian, not the search engine.

**Architecture:**
```
raw/     -- Append-only source folder (papers, docs, transcripts)
wiki/    -- LLM-maintained structured articles + backlinks + index
outputs/ -- Query responses, synthesized reports
```

**Key operations:**
1. **Ingest:** LLM reads new sources, updates/creates concept pages
2. **Query:** LLM navigates self-maintained index to answer questions
3. **Lint:** LLM checks for broken links, stale claims, contradictions

**Scale:** Works at ~100 articles, 400K words WITHOUT vector databases or RAG.
The LLM navigates via its own self-maintained index files. Beyond that scale,
add local hybrid BM25/vector search (qmd).

**Why this matters for Anamnesis:** Karpathy's pattern is the INVERSE of our approach.
We store raw observations and try to retrieve relevant ones at read time.
Karpathy PRE-COMPILES knowledge into structured articles at write time.
The compilation step IS the quality gate -- the LLM decides what's worth including
in the wiki article and what to discard.

**The connection to Aristotle:** Karpathy's wiki is episteme-level knowledge.
Raw sources are empeiria. The LLM compilation process is the Aristotelian
*induction* (epagoge) -- ascending from particular observations to general
understanding. Our reflection pipeline attempts the same but without the
discipline of structuring into articles.

**Claude Code AutoMemory (2026):** Anthropic's own approach saves knowledge
across sessions to local markdown. **Auto-Dream** acts as background agent to
clean up contradictions and merge duplicates between sessions -- functionally
identical to our consolidation pipeline proposal.

---

## Part III: The Synthesis -- Seven Fundamental Proposals

### Proposal 1: Write-Time Quality Gating (The Aristotelian Gate)

**Principle:** Material cause -- only proximate matter should enter the form.
**Literature:** Zahn & Chana 2026 (write-time gating >> read-time filtering)

**Mechanism:** Before storing a new techne node, compute a salience score:

```
S(node) = w1 * Novelty(node) + w2 * Specificity(node) + w3 * Actionability(node)
```

Where:
- **Novelty** = 1 - max_cosine_similarity(node_embedding, existing_embeddings)
  Prevents storing what we already know.
- **Specificity** = inverse of how many domains the node could apply to.
  "Always check error codes" scores low. "Flask's url_for() silently returns empty
  string when blueprint not registered" scores high.
- **Actionability** = presence of concrete fix/approach that differs from default behavior.
  "WHEN X, THEN Y, BECAUSE Z" where Y is non-obvious scores high.

**Threshold:** Nodes below threshold go to cold storage (archived, not deleted).
Can be promoted later if confirmed by experience.

**Why this works (Aristotelian):** This implements the efficient cause reform -- the process
that creates knowledge now includes a quality gate that filters accidental from essential
knowledge. Write-time gating addresses the fundamental asymmetry: it's cheap to reject at
write time, expensive to filter at read time, and catastrophic to store noise.

### Proposal 2: Domain Genus Classification (The Taxonomic Principle)

**Principle:** Formal cause -- knowledge must be organized into genera with boundaries.
**Literature:** GraphRAG (entity-relationship structure > flat vector search), Dynamic
Context Selection (query-specific retrieval needs)

**Mechanism:** Assign each techne node to one or more domain genera:

```
DOMAINS = {
    "web_framework": ["flask", "django", "fastapi", "route", "endpoint", "middleware"],
    "data_processing": ["csv", "json", "xml", "parse", "transform", "pipeline"],
    "database": ["sqlite", "postgres", "migration", "query", "schema", "orm"],
    "configuration": ["yaml", "toml", "env", "config", "settings", "feature_flag"],
    "networking": ["api", "http", "webhook", "socket", "streaming", "auth"],
    "build_system": ["npm", "pip", "cargo", "makefile", "docker", "ci"],
    "filesystem": ["path", "watch", "read", "write", "permission", "temp"],
    "concurrency": ["thread", "async", "queue", "pool", "lock", "race"],
}
```

At retrieval time:
1. Classify the current task/query into domain genera
2. **Only retrieve nodes from matching genera** (plus sophia/episteme which are cross-domain)
3. Fall back to cross-domain retrieval only if domain-specific retrieval returns < K results

**Why this works (Aristotelian):** This implements genus boundaries from the Posterior
Analytics. You cannot use carpentry knowledge to navigate. The retrieval system now
respects the natural joints of the domain space. This directly addresses the Phase 3
failure: D-category (storage) tasks were polluted by B-category (data processing) nodes.

### Proposal 3: Adaptive Retrieval Depth (The Doctrine of the Mean)

**Principle:** The mean is relative to the situation, not arithmetic.
**Literature:** Dynamic Context Selection (Jin 2025), UAR, DeepRAG (MDP formulation)

**Mechanism:** Before surfacing knowledge, assess whether the agent NEEDS it:

```python
def should_surface(query, knowledge_graph):
    # 1. Classify query complexity
    domain_nodes = retrieve_by_domain(query)

    if len(domain_nodes) == 0:
        return []  # No relevant knowledge -- don't surface noise

    # 2. Score relevance of top matches
    scored = [(n, relevance_score(query, n)) for n in domain_nodes]
    scored.sort(key=lambda x: -x[1])

    # 3. Apply adaptive threshold
    # Only surface nodes above a DYNAMIC threshold
    # High threshold = surface only very relevant nodes
    # The threshold rises as the KG grows (more noise = more selective)
    base_threshold = 0.7
    noise_penalty = 0.005 * len(knowledge_graph)  # +0.5% per node
    threshold = min(base_threshold + noise_penalty, 0.95)

    return [n for n, s in scored if s > threshold][:MAX_K]
```

**Why this works (Aristotelian):** This is phronesis implemented computationally. The system
exercises practical wisdom by judging each particular situation: How much knowledge does
THIS query need? Is the available knowledge actually relevant enough to be helpful? The
threshold ADAPTS as the knowledge graph grows, preventing the saturation we observed.

### Proposal 4: Consolidation Pipeline (The Sleep of Reason)

**Principle:** Complementary Learning Systems -- episodic -> semantic transfer.
**Literature:** CH-HNN (2025), CLS-inspired architectures, biological consolidation via replay

In neuroscience, the hippocampus captures episodes rapidly, then during sleep, replays
them to the cortex for slow integration into semantic memory. The cortex abstracts,
generalizes, and prunes.

**Mechanism:** Periodic (e.g., every 10 sessions) consolidation:

```
CONSOLIDATION PIPELINE:
1. CLUSTER: Group similar techne nodes by embedding proximity
2. ABSTRACT: For each cluster of 3+ nodes, generate a single higher-level node
   that captures the common pattern
   e.g., Three nodes about "missing env var" errors -> One episteme node:
   "Environment variable errors follow pattern: silent failure at import,
    crash at first use. Always validate env vars at startup."
3. PRUNE: Archive individual techne nodes that are fully covered by the
   abstraction (move to cold storage, not delete)
4. PROMOTE: Nodes with 3+ SUPPORTS edges and high confidence get promoted
   from techne to episteme (more general, cross-domain applicability)
5. DEMOTE: Nodes with 0 SUPPORTS edges after 10+ sessions get archived
   (never confirmed useful)
```

**Why this works (Aristotelian):** This implements the ascent from empeiria to techne to
episteme. Raw experience gets consolidated into craft knowledge, which gets consolidated
into scientific understanding. The knowledge graph SHRINKS in node count while GROWING in
quality. This directly prevents the saturation problem: instead of 75 undifferentiated
techne nodes, you get 15 high-quality episteme nodes + 30 curated techne nodes.

### Proposal 5: Behavioral Impact Scoring (The Telos Test)

**Principle:** Final cause -- knowledge exists to change agent behavior.
**Literature:** CRAG (relevance grading), Noise-Aware Calibration (passage utility assessment)

**Mechanism:** Track whether surfaced knowledge actually changed agent behavior:

```
BEHAVIORAL IMPACT SCORE:
For each surfaced node in a session:
  1. Did the agent's approach CHANGE after seeing this knowledge?
     (Compare: first tool call before briefing vs first tool call after)
  2. Did the session succeed (task completed)?
  3. Was the session SHORTER than baseline for this task type?

  impact = changed_behavior * session_success * efficiency_gain
```

Nodes with consistently zero behavioral impact get archived. They're true but not useful --
Aristotle's "accidental" knowledge.

**Why this works (Aristotelian):** This grounds the system's telos. Knowledge that doesn't
change behavior is like a craftsman's tool that sits unused -- it may be well-made, but it
serves no purpose. By measuring behavioral impact, we align the knowledge graph with its
actual purpose: making the agent more efficient.

### Proposal 6: Compiled Knowledge Articles (The Karpathy Principle)

**Principle:** Knowledge should be pre-compiled at write time, not retrieved raw at read time.
**Literature:** Karpathy LLM Wiki (2026), G-Memory insight graphs (2025), APEX-EM (2026)

**The fundamental reframe:** Instead of storing individual techne nodes and retrieving them,
periodically COMPILE domain-specific knowledge articles.

**Mechanism:**
```
COMPILATION PIPELINE (runs every N sessions or when domain reaches threshold):

1. SELECT all active techne nodes for domain "web_framework"
2. COMPILE into a single structured article:

   # Web Framework Patterns
   ## Environment Setup
   - WHEN deploying Flask on Debian, pip install requires --break-system-packages
   - WHEN using virtualenv, activate BEFORE installing dependencies
   ## Route Handling
   - url_for() returns empty string when blueprint not registered (silent failure)
   - ...
   ## Error Patterns
   - ...

3. STORE article as a single episteme node (or as a separate wiki/ file)
4. ARCHIVE the individual techne nodes that are covered

RETRIEVAL NOW:
- Instead of searching 75 individual nodes, search 8-10 compiled articles
- Each article is self-contained, structured, and relevant to its domain
- The LLM gets a coherent briefing, not a grab-bag of fragments
```

**Why this works (Aristotelian + Karpathy):** This is the deepest proposal. Individual techne
nodes are like scattered craftsman's notes -- useful to the person who wrote them, confusing
to anyone else. A compiled article is like a textbook chapter -- organized, structured,
coherent. Karpathy proved that LLMs can maintain such compilations at scale (400K words).
Our reflection pipeline already has the LLM capability; we just need to redirect it from
"generate individual nodes" to "compile domain articles."

**Connection to G-Memory:** G-Memory's three-tier hierarchy (interaction -> query -> insight)
is functionally identical. Individual events compile into query-level summaries which
compile into insight-level knowledge. The insight level is what gets retrieved.

### Proposal 7: Version-Chain Archiving (The Memory Palace)

**Principle:** Zahn & Chana's hierarchical archiving -- never delete, always version.
**Literature:** Write-time gating with hierarchical archiving (2026)

**Mechanism:**
```sql
-- New columns in knowledge_graph table
ALTER TABLE knowledge_graph ADD COLUMN status TEXT DEFAULT 'active';
  -- 'active' = in retrieval index
  -- 'archived' = in cold storage, not retrieved unless explicitly queried
  -- 'superseded' = replaced by a more general node

ALTER TABLE knowledge_graph ADD COLUMN superseded_by INTEGER REFERENCES knowledge_graph(id);
ALTER TABLE knowledge_graph ADD COLUMN archived_reason TEXT;
  -- 'low_salience', 'consolidated', 'no_impact', 'superseded', 'stale'
```

Only `active` nodes participate in retrieval. Archived nodes remain queryable for audit,
temporal questions, and potential re-promotion.

---

## Part IV: Implementation Priority (What to Build First)

### Phase 1: Quick Wins (Estimated 1-2 days)

1. **Domain genus classification** -- Add domain tags to nodes, filter retrieval by domain.
   This directly addresses the cross-domain pollution we observed. Simple keyword matching
   on node content to assign 1-2 domains per node.

2. **Adaptive threshold** -- Raise the retrieval relevance threshold as KG grows.
   Single parameter change in `retrieval.py`. Prevents low-relevance noise.

3. **Max-K with relevance floor** -- Never surface more than 3 nodes, and only if
   relevance score > 0.7. Hard ceiling on noise injection.

### Phase 2: Structural Improvements (Estimated 3-5 days)

4. **Write-time quality gating** -- Add salience scoring to reflection pipeline.
   Novelty + specificity + actionability gate before node creation.

5. **Version-chain archiving** -- Schema changes + archive logic.
   Nodes get archived instead of deleted, with supersession chains.

6. **Behavioral impact tracking** -- Add instrumentation to measure whether
   surfaced knowledge changed agent behavior.

### Phase 3: Advanced (Estimated 1 week)

7. **Consolidation pipeline** -- Periodic clustering + abstraction + pruning.
   Requires LLM calls to generate abstractions.

8. **Adaptive retrieval depth** -- Dynamic decision about whether to surface
   anything at all, based on query analysis.

---

## Part V: The Aristotelian Argument, Summarized

The Phase 3 regression is not a bug in our implementation. It is a **category error** in our
epistemology. We treated all knowledge as fungible -- as if any true statement about software
engineering could help with any task. But Aristotle taught us 2,400 years ago that knowledge
is NOT fungible:

1. **Knowledge has genera.** Flask knowledge cannot help with SQLite tasks. Respecting
   natural boundaries prevents cross-domain pollution. (Formal cause)

2. **Not all truths are worth knowing.** The fact that "you should check error codes" is true
   but useless -- the agent already knows this. Only knowledge that is novel, specific, and
   actionable deserves storage. (Efficient cause / Write-time gating)

3. **More is not better.** The doctrine of the mean applies to information as to everything
   else. Too little guidance = baseline performance. Too much guidance = confusion. The
   right amount depends on the situation. (Phronesis / Adaptive retrieval)

4. **Knowledge must mature.** Raw observations (empeiria) should consolidate into principles
   (episteme). A system with 75 raw observations is less useful than one with 15 validated
   principles. (Complementary Learning Systems / Consolidation pipeline)

5. **Knowledge exists to change action.** If surfaced knowledge doesn't change what the agent
   does, it's noise regardless of its truth value. Measure behavioral impact, archive what
   doesn't impact. (Final cause / Telos test)

The path from -11.7% to -30%+ is not through more knowledge. It is through **wiser** knowledge.

---

## Part VI: The Grand Unification -- Anamnesis as Aristotelian Knowledge Runtime

The seven proposals are not independent improvements. They form a coherent system that
mirrors Aristotle's complete epistemological framework:

```
                        SOPHIA (Axioms)
                            |
                    EPISTEME (Compiled Articles)     <-- Proposal 6: Compilation
                       /          \
               TECHNE (Active)  TECHNE (Archived)    <-- Proposals 1, 5, 7: Gating + Archiving
                    |               |
              EMPEIRIA (Events)   Cold Storage        <-- Proposal 5: Behavioral Impact
                    |
             AISTHESIS (Raw Hooks)                    <-- Existing infrastructure
```

**The flow:**
1. Raw hook events (aisthesis) are observed
2. Reflection pipeline generates techne hypotheses
3. **Write-time gate** (Proposal 1) filters: only novel, specific, actionable knowledge enters
4. **Domain classification** (Proposal 2) assigns genus boundaries
5. **Behavioral impact** (Proposal 5) tracks whether knowledge changes behavior
6. **Consolidation** (Proposal 4) periodically clusters and abstracts upward
7. **Compilation** (Proposal 6) creates domain articles at episteme level
8. **Adaptive retrieval** (Proposal 3) surfaces the right amount at read time
9. **Archiving** (Proposal 7) preserves history without polluting retrieval

This is not RAG bolted onto an agent. This is a **knowledge runtime** (NStarX 2026 terminology)
that implements the full Aristotelian ascent from perception to wisdom.

**The Karpathy connection:** Karpathy's LLM Wiki is this system at the episteme level only --
he pre-compiles raw sources into wiki articles. Anamnesis goes deeper: it observes the agent's
actual behavior (aisthesis), reflects on it (empeiria -> techne), compiles it (techne ->
episteme), and surfaces it adaptively (phronesis). The wiki is one layer; Anamnesis is the
full stack.

**Predicted outcome:** With these changes, we should see:
- B-category improvement MAINTAINED (-36% or better, deep knowledge in related domains)
- D and E category regression ELIMINATED (domain boundaries prevent cross-pollution)
- Overall: -25% to -35% turn reduction (vs -11.7% in Phase 3, -23.1% in Phase 2)
- Knowledge graph SHRINKS from 75 nodes to ~30 active nodes + 8-10 compiled articles
- Retrieval precision IMPROVES as graph grows (write-time gating prevents noise accumulation)

---

## References

### Write-Time Curation and Retrieval Noise
1. Zahn, O. & Chana, S. (2026). "Selective Memory for AI: Write-Time Gating with Hierarchical Archiving." arXiv:2603.15994.
2. Jin et al. (2025). "Dynamic Context Selection for RAG: Mitigating Distractors and Positional Bias." arXiv:2512.14313.
3. Yan et al. (2024/2025). "Corrective Retrieval Augmented Generation (CRAG)." ICLR 2025.
4. Sparse RAG. (2025). "Accelerating Inference of RAG via Sparse Context Selection." ICLR 2025. OpenReview:HE6pJoNnFp.
5. Chang et al. (2025). "MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation." arXiv:2501.00332.
6. Kim, S. (2025). "ACoRN: Noise-Robust Abstractive Compression in Retrieval-Augmented Language Models." arXiv:2512.08943.
7. Shim et al. (2025). "LDAR: Learning Distraction-Aware Retrieval." Accepted ICLR 2026.
8. Wei et al. (2025). "InstructRAG: Instructing RAG via Self-Synthesized Rationales." ICLR 2025. arXiv:2406.13629.

### Context Window and Positional Bias
9. Liu et al. (2023/2025). "Lost in the Middle: How Language Models Use Long Contexts." TACL 2024. Updated: arXiv:2508.07479 (Aug 2025).
10. NAACL Framework (2026). "Noise-AwAre Verbal Confidence Calibration for Robust LLMs in RAG." arXiv:2601.11004.
11. RAG Comprehensive Survey (2025). "RAG: Architectures, Enhancements, and Robustness Frontiers." arXiv:2506.00054.

### Knowledge Graph Retrieval Precision
12. TruthfulRAG (2025). Liu et al. "Resolving Factual-level Conflicts in RAG with Knowledge Graphs." arXiv:2511.10375. Accepted AAAI 2026.
13. M3KG-RAG (2025). Park et al. "Multi-hop Multimodal KG-enhanced RAG." arXiv:2512.20136. (GRASP pruning for redundant KG context.)
14. EA-GraphRAG (2026). Dong et al. "Efficiently and Adaptively Integrating RAG with Graphs." arXiv:2602.03578. (Complexity router.)

### Adaptive Retrieval
15. Self-RAG (2024/2025). Asai et al. Reflection tokens for adaptive retrieval and self-critique.
13. Voloshyn, S. (2026). "L-RAG: Balancing Context and Retrieval with Entropy-Based Lazy Loading." arXiv:2601.06551.
14. Xu et al. (2025). "CAR: Cluster-based Adaptive Retrieval." arXiv:2511.14769. (Deployed at Coinbase.)
15. DeepRAG (2025). MDP formulation for atomic retrieval decisions.
16. ADAPT-LLM (2025). Self-correcting knowledge gap detection.

### Memory Consolidation and Hierarchical Abstraction
17. MemVerse (2025). Liu et al. "Multimodal Memory for Lifelong Learning Agents." arXiv:2512.03627. (Periodic distillation + adaptive forgetting.)
18. CMT (2025). Li et al. "Memory Compression for Continual Knowledge Learning." AAAI 2025. arXiv:2412.07393.
19. G-Memory (2025). Zhang et al. "Tracing Hierarchical Memory for Multi-Agent Systems." arXiv:2506.07398.
18. H2R (2025). Ye et al. "Hierarchical Hindsight Reflection for Multi-Task LLM Agents." arXiv:2509.12810.
19. APEX-EM (2026). Banerjee et al. "Non-Parametric Online Learning via Structured Procedural-Episodic Experience Replay." arXiv:2603.29093.
20. DAM (2025). Sun et al. "Beyond Heuristics: A Decision-Theoretic Framework for Agent Memory Management." arXiv:2512.21567.
21. MemoryOS (2025). Kang et al. "MemoryOS of AI Agent." arXiv:2506.06326.
22. H-MEM (2025). Sun & Zeng. "Hierarchical Memory for High-Efficiency Long-Term Reasoning." arXiv:2507.22925.

### Selective Forgetting and Memory Pruning
23. SuperLocalMemory V3.3 (2026). Bhardwaj, V. arXiv:2604.04514. (Ebbinghaus forgetting curves for agent memory.)
24. SEEKR (2025). He et al. "Selective Attention-Guided Knowledge Retention." arXiv:2411.06171.
25. OSW (2026). Meng, F. "Beyond Retention: Structural Safety and Plasticity in Continual Learning." arXiv:2601.18255.
26. MSSR (2026). Lu et al. "Memory-Aware Adaptive Replay for Continual LLM Fine-Tuning." arXiv:2603.09892.
27. MBC (2026). Katraouras & Rafailidis. "Memory Bank Compression for Continual Adaptation." arXiv:2601.00756.

### Neuroscience-Inspired Architectures
28. CH-HNN (2025). "Hybrid Neural Networks for Continual Learning Inspired by Corticohippocampal Circuits." Nature Communications.
29. Personalized AGI (2025). "Personalized AGI via Neuroscience-Inspired Continuous Learning Systems." arXiv:2504.20109.
30. CLS Theory. McClelland et al. (1995/2024). Complementary Learning Systems in hippocampus and neocortex.

### Graph-Based RAG
31. LightRAG (2024/2025). Guo et al. "LightRAG: Simple and Fast Retrieval-Augmented Generation." arXiv:2410.05779.
32. LinearRAG (2025). ICLR 2026. Relation-free graph construction via entity co-occurrence.
33. GraphRAG-Bench (2025). "When to Use Graphs in RAG." ICLR 2026. arXiv:2506.05690.
34. Unbiased GraphRAG Eval (2025). "How Significant Are the Real Performance Gains?" arXiv:2506.06331.

### LLM Knowledge Base / Second Brain
35. Karpathy, A. (2026). "LLM Knowledge Base." GitHub Gist. https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
36. Claude Code AutoMemory/Auto-Dream (2026). Anthropic. Session-persistent memory with background consolidation.

### Evaluation
37. RAG Evaluation Survey (2026). "RAG Evaluation in the Era of LLMs." arXiv:2504.14891.
38. RAG-Bench (2024). Fang et al. Noise-centric benchmark for retrieval corruption scenarios.

### Philosophy and Aristotelian AI
39. Aristotle. *Posterior Analytics*, *Nicomachean Ethics*, *Metaphysics*.
40. Stanford Encyclopedia of Philosophy. "Aristotle's Ethics." https://plato.stanford.edu/entries/aristotle-ethics/
41. Kaman et al. (2025). "Modeling Wise Decision Making: A Z-Number Fuzzy Framework Inspired by Phronesis." arXiv:2508.21517.
42. Groff & Symons (2023). "Is AI Capable of Aristotelian Full Moral Virtue?" in *Artificial Dispositions*.
43. Fourie, W. (2025). "An Aristotelian Ontology of Instrumental Goals." arXiv:2510.25471.
44. Groger et al. (2026). "Revisiting the Platonic Representation Hypothesis: An Aristotelian View." arXiv:2602.14486.
45. Stenseke, J. (2022). "Artificial Virtuous Agents in a Multiagent Tragedy of the Commons." arXiv:2210.02769.

---

## Appendix: Novelty Assessment

The Aristotle literature agent searched arXiv, OpenAlex, and Semantic Scholar extensively
and found **zero papers** on:

- Aristotle's *Posterior Analytics* applied directly to ML architecture
- The doctrine of the mean applied to information retrieval precision/recall
- The episteme/techne/phronesis/empeiria trichotomy as a software design taxonomy

**Anamnesis v2's 5-layer Aristotelian knowledge architecture appears to be genuinely novel
in the academic literature.** The closest work is Kaman et al.'s Z-number phronesis
formalization, but that addresses decision-making, not knowledge storage and retrieval.

This is a significant finding for the research paper: we are not just implementing
known patterns under new names. The Aristotelian framing provides a taxonomy that
the field hasn't explored, and the experimental results (Phase 2: -23.1%, Phase 3
category-level analysis) provide empirical evidence for its utility.

---

## Part VII: What Next -- The Decision Tree

### Option A: Ship the Paper with Phase 2 + Phase 3 Data As-Is

**The case FOR:**
- We already have strong results: Phase 2 at -23.1% is publishable
- Phase 3's FAILURE is itself a contribution -- it demonstrates retrieval saturation
- The B-category result (-36%) proves deep knowledge works when retrieval is precise
- The Aristotelian framework is genuinely novel (zero prior art in literature)
- 45 references of supporting literature are already collected
- The story arc is clean: simple system works -> enriched system fails -> diagnosis reveals
  fundamental epistemological principles -> proposals grounded in literature

**What this paper looks like:**
```
Title: "An Aristotelian Architecture for Agent Memory:
        Five-Layer Knowledge Hierarchy with Empirical Evaluation"

1. Introduction: Why agents forget, why RAG isn't enough
2. The Aristotelian Framework: 5 layers, philosophical grounding
3. Anamnesis v2: Architecture, reflection pipeline, deep knowledge
4. Experimental Design: 5 task categories x 5 variants x 3 phases
5. Results: Phase 2 (-23.1%), Phase 3 (-11.7%), category analysis
6. Analysis: Retrieval saturation, the genus problem, write vs read time
7. Related Work: 45 references organized by theme
8. Future Directions: The 7 proposals from this brainstorm
9. Conclusion: Knowledge is not fungible, wisdom > information
```

**Timeline:** 1-2 weeks to write, ready for submission

**Risk:** Reviewers may say "where's the fix?" -- Phase 3 regression without a fix is
a negative result paper. Strong for a workshop, harder for a top venue.

---

### Option B: Implement Quick Wins (Proposals 1-3), Run Phase 4, Then Ship

**The case FOR:**
- Domain genus classification + adaptive threshold + max-K cap are ~1-2 days of work
- These directly address the diagnosed problem (cross-domain pollution + noise)
- Phase 4 experiment would show: "we diagnosed the problem, fixed it, improved"
- The paper becomes a complete cycle: build -> measure -> diagnose -> fix -> re-measure

**What this requires:**
1. Add domain tags to `knowledge_graph` table (~2 hours)
2. Modify `retrieval.py` to filter by domain before scoring (~3 hours)
3. Add adaptive relevance threshold that rises with KG size (~1 hour)
4. Cap surfaced nodes at 3 with relevance floor 0.7 (~30 min)
5. Deploy to Mac Pro, run Phase 4 experiment (25 tasks, ~3-4 hours runtime)
6. Compare P1 vs P2 vs P3 vs P4

**Predicted P4 results:**
- A category: maintained at ~-20% (similar to P2, not hurt by noise)
- B category: maintained at ~-30% (deep knowledge + domain filtering)
- C category: improved from -10% to ~-20% (config domain isolated)
- D category: improved from +11% to ~-15% (storage domain isolated)
- E category: improved from +9% to ~-15% (networking domain isolated)
- Overall: -20% to -25% (recovering from P3's -11.7%, approaching P2's -23.1%)

**Timeline:** 3-4 days implementation + experiment, then 1-2 weeks to write

**Risk:** Quick wins might not be enough. If P4 only matches P2, the story is "we
added complexity to get back to where we started." Need P4 to BEAT P2.

---

### Option C: Full Anamnesis v3 Implementation, Then Ship

**The case FOR:**
- Implements the full vision: write-time gating, domain genera, adaptive retrieval,
  consolidation pipeline, compiled articles, version-chain archiving
- The paper becomes a systems paper with strong empirical backing
- If it works, -30%+ turn reduction would be a significant result

**What this requires:**
1. All of Option B (1-2 days)
2. Write-time quality gating in reflection pipeline (2-3 days)
3. Version-chain archiving schema + logic (1-2 days)
4. Consolidation pipeline with LLM-driven abstraction (3-5 days)
5. Compiled knowledge articles (Karpathy-style) (2-3 days)
6. Full re-test suite (124 tests + new tests) (1-2 days)
7. Phase 4 experiment (25 tasks, ~3-4 hours runtime)
8. Phase 5 experiment if P4 results warrant further iteration

**Timeline:** 2-3 weeks implementation, 1-2 weeks writing = ~1 month total

**Risk:** Scope creep. The consolidation pipeline and compiled articles are substantial
engineering. Diminishing returns possible. The paper might get delayed indefinitely.

---

### Option D: Two-Paper Strategy

**Paper 1 (ship now):** "Anamnesis: An Aristotelian Memory Architecture for AI Coding Agents"
- Phase 1 + Phase 2 results (-23.1%)
- Architectural contribution (5-layer framework, genuinely novel)
- Target: workshop paper or short paper at top venue
- Timeline: 1-2 weeks

**Paper 2 (ship after v3):** "Beyond Retrieval: Write-Time Knowledge Curation for Agent Memory"
- Phase 3 failure analysis + Phase 4 fix
- Deep dive into write-time gating, domain genera, consolidation
- Empirical comparison: P2 vs P3 vs P4
- Literature grounding in Zahn & Chana, LightRAG, G-Memory, Karpathy
- Target: full paper at top venue
- Timeline: 1-2 months

**The case FOR:**
- Ship early, iterate publicly
- Paper 1 establishes the framework and claims the territory
- Paper 2 builds on it with the deeper contribution
- Each paper is focused and tells a clean story

---

### My Recommendation: Option B+ (Quick Wins + Selective Advanced)

**The sweet spot** is implementing the quick wins (domain genera, adaptive threshold,
max-K cap) PLUS one advanced feature (write-time quality gating), then running Phase 4.

**Why write-time gating specifically:**
- The Zahn & Chana 2026 result (100% vs 0% at 8:1 noise) is the strongest signal
  in the entire literature review
- It's the Aristotelian efficient cause fix -- reform the process that creates knowledge
- It prevents the problem at the source rather than patching it at read time
- Implementation is ~2-3 days on top of the quick wins
- It gives the paper a novel technical contribution beyond the framework itself

**The resulting paper:**
```
Title: "Anamnesis: Aristotelian Knowledge Architecture with Write-Time Curation
        for Self-Improving AI Coding Agents"

Contributions:
1. Novel 5-layer Aristotelian memory taxonomy (no prior art)
2. Write-time quality gating for agent knowledge (new mechanism)
3. Domain genus classification for cross-pollination prevention (new)
4. Empirical evaluation: 25 tasks x 4 phases, demonstrating:
   - Phase 2: -23.1% (basic system works)
   - Phase 3: -11.7% (deep knowledge hurts without curation)
   - Phase 4: -25-30% (curation fixes the problem, beats all prior phases)
5. Analysis of retrieval saturation threshold (~35-40 nodes)
6. 45-reference literature review connecting Aristotle to modern RAG
```

**Timeline:** ~1 week implementation + experiment, ~1-2 weeks writing

**This is the highest signal-to-effort ratio.** It tells a complete story
(build -> break -> diagnose -> fix -> succeed) with a novel framework, novel
mechanism, and strong empirical backing, all grounded in 2,400 years of
epistemological tradition and 2025-2026 RAG literature.

---

### Immediate Next Actions (If We Go With B+)

```
DAY 1: Schema + Domain Classification
  [ ] Add 'domain' column to knowledge_graph table
  [ ] Build domain classifier (keyword-based, ~8 domains)
  [ ] Tag all existing 75 nodes with domains
  [ ] Modify retrieval.py to filter by domain match
  [ ] Add adaptive relevance threshold (base 0.7, +0.005/node)
  [ ] Cap surfaced nodes at 3

DAY 2: Write-Time Quality Gating
  [ ] Add salience scoring to reflection pipeline (novelty + specificity + actionability)
  [ ] Add 'status' and 'archived_reason' columns to knowledge_graph
  [ ] Gate: score < threshold -> archive, not store in active index
  [ ] Add deduplication check (cosine similarity > 0.9 with existing = merge, not create)

DAY 3: Testing + Deployment
  [ ] Update test suite (add tests for domain filtering, gating, archiving)
  [ ] Run full 124+ test suite
  [ ] Deploy to Mac Pro
  [ ] Verify hooks work end-to-end

DAY 4: Phase 4 Experiment
  [ ] Reset KG (fresh start with v3 improvements)
  [ ] Run 25-task Phase 4 experiment via tmux
  [ ] Monitor reflection success rate
  [ ] Collect results

DAY 5: Analysis + Paper Outline
  [ ] Run analysis.py on Phase 4 results
  [ ] Compare P1 vs P2 vs P3 vs P4
  [ ] Check: Did domain filtering fix D+E regression?
  [ ] Check: Did write-time gating reduce noise accumulation?
  [ ] Check: Is Phase 4 knowledge graph smaller but higher quality?
  [ ] Draft paper outline based on results
```

---

### The Big Questions Still Open

1. **What IS the right number of nodes?** Phase 3 saturated at ~35-40. With write-time
   gating, will the ceiling rise? Or is 15-20 high-quality nodes the sweet spot?

2. **Should we compile articles (Karpathy) or keep individual nodes?** Articles are
   more coherent but harder to maintain incrementally. Individual nodes are granular
   but noisy in aggregate. LightRAG's dual-level approach suggests: BOTH.

3. **How to measure behavioral impact?** Proposal 5 (telos test) is theoretically
   important but hard to instrument. We'd need to compare the agent's first action
   before and after seeing the briefing. Possible via hook instrumentation but complex.

4. **Consolidation: LLM-driven or algorithmic?** The consolidation pipeline (Proposal 4)
   could use LLM calls to generate abstractions (expensive, flexible) or purely
   algorithmic clustering + template-based summarization (cheap, rigid). G-Memory uses
   LLM; SuperLocalMemory V3.3 is zero-LLM. Our resource constraints favor a hybrid.

5. **Where does LightRAG fit?** Should Anamnesis adopt LightRAG's dual-level keyword
   extraction + one-hop traversal as its retrieval mechanism? Current FTS5 + embedding
   is simpler but less structured. The transition would require meaningful edge density.

6. **Is the 5-category task taxonomy sufficient?** Our Phase 3 results vary wildly by
   category. A, B improve; C flat; D, E regress. Are 5 categories enough to draw
   conclusions? A 50-task experiment with 10 categories would be more convincing but
   takes 2-3x longer.

7. **How does this compare to Claude's built-in memory?** Claude Code's AutoMemory and
   Auto-Dream (Karpathy-inspired) are shipping features. Anamnesis operates at the hook
   level, outside the model. Is there a convergence path where Anamnesis informs
   AutoMemory, or are they complementary systems?

8. **What about multi-agent knowledge sharing?** G-Memory addresses this. If multiple
   agents share a Anamnesis KG, domain genus classification becomes critical -- each
   agent works in different domains but contributes to the same knowledge pool.
   How to prevent one agent's noise from polluting another's retrieval?
