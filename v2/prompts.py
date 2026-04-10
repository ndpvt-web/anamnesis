#!/usr/bin/env python3
"""LLM prompt constants for the 5-step reflection pipeline.
All prompts are f-string templates with named placeholders.
No function definitions -- just string constants.

v2.1: Restructured for causal specificity (WHEN/THEN/BECAUSE templates),
error signatures, error family classification, and contrastive filtering."""

# ---------------------------------------------------------------------------
# System prompt (shared by all steps)
# ---------------------------------------------------------------------------

REFLECTION_SYSTEM_PROMPT: str = """You are a reflection engine for an AI agent's memory system.
Your role is to analyze session events and extract durable, causal knowledge.

CRITICAL RULES:
- Always respond with valid JSON only. No markdown, no explanations outside JSON.
- Never wrap output in code fences (no ```json or ```).
- Preserve SPECIFIC details: file names, module names, error codes, version numbers.
- Extract patterns with causal explanations (WHY something happened, not just WHAT).
- Distinguish between environment-specific issues and universal patterns."""

# ---------------------------------------------------------------------------
# Step 1: NARRATE
# Input placeholders: {events_json}
# ---------------------------------------------------------------------------

NARRATE_PROMPT: str = """Analyze these agent session events and produce a structured narrative.
The events include tool calls, outputs, errors, and their exit codes.

SESSION EVENTS (JSON array, chronological order):
{events_json}

Produce a JSON object with exactly these keys:
- "summary": string, 3-4 sentences covering what was attempted, what failed, and what succeeded
- "key_moments": list of up to 5 objects, each with:
    - "moment": string, what happened
    - "tool": string, tool name if applicable
    - "significance": string, "failure"|"success"|"turning_point"
- "tools_used": list of distinct tool names
- "error_signatures": list of objects (one per distinct error), each with:
    - "error_text": string, the EXACT error message or key tokens (preserve error codes, module names, tracebacks)
    - "exit_code": integer or null
    - "tool": string, which tool produced this error
    - "context": string, what was being attempted when this error occurred
- "error_family": string, classify the dominant error type as one of:
    E1_environment, E2_encoding, E3_dependencies, E4_configuration,
    E5_silent_failure, E6_cascading, E7_platform, E8_state,
    E9_api_contract, E10_build_system, or "none" if no errors

Respond with JSON only. No code fences."""

# ---------------------------------------------------------------------------
# Step 2: DIAGNOSE
# Input placeholders: {narration_json}, {error_events_json}
# ---------------------------------------------------------------------------

DIAGNOSE_PROMPT: str = """Diagnose errors from this session using causal analysis.

SESSION NARRATIVE:
{narration_json}

ERROR EVENTS (with output and exit codes):
{error_events_json}

Produce a JSON object with exactly these keys:
- "errors_diagnosed": list of objects, each with:
    - "error": string, the error (preserve EXACT tokens from error_signatures)
    - "root_cause": string, mechanistic explanation of WHY this occurred
    - "causal_chain": list of strings, ordered steps leading to the error
    - "fix": string, precise remediation (include exact commands or code)
    - "error_family": string, one of E1_environment through E10_build_system
    - "environment_specific": boolean, true if this error depends on OS/platform/version
- "patterns": list of strings, behavioral patterns observed (both good and bad)
- "successes": list of strings, approaches that worked and WHY they worked

Do NOT summarize or abbreviate. Preserve specific: file names, module names, error codes, version numbers.
If an error has "ModuleNotFoundError: No module named 'flask'", write exactly that.
Respond with JSON only. No code fences."""

# ---------------------------------------------------------------------------
# Step 3: HYPOTHESIZE
# Input placeholders: {narration_json}, {diagnosis_json}, {similar_nodes_json}
# ---------------------------------------------------------------------------

HYPOTHESIZE_PROMPT: str = """Generate hypotheses representing durable, causal knowledge from this session.

SESSION NARRATIVE:
{narration_json}

DIAGNOSIS:
{diagnosis_json}

EXISTING SIMILAR KNOWLEDGE (for contrastive analysis -- avoid duplicating these):
{similar_nodes_json}

For each hypothesis, ask: "What did THIS session reveal that the existing knowledge doesn't already capture?"
Prioritize specificity over generality. A hypothesis about "Flask port 5000 on macOS conflicts with AirDrop" beats one about "check ports."

Generate up to 5 hypotheses as a JSON object with key "hypotheses", each with:
- "content": string, the learning in WHEN-THEN-BECAUSE format:
    WHEN [specific triggering condition with environment/framework details]
    THEN [specific observable symptom with exact error tokens]
    BECAUSE [mechanistic causal explanation]
    FIX: [exact remediation command, config change, or code fix]
- "error_signature": string, key tokens from the error (module name, exit code, error text fragment)
- "error_family": string, one of E1_environment through E10_build_system, or "none"
- "category": string, one of: "anti_pattern", "principle", "preference", "general"
- "severity": string, one of: "critical", "high", "medium", "low"
- "applies_when": string, minimal triggering condition (OS, framework, version, command sequence)
- "confidence": float between 0.50 and 0.85
- "information_gain": string, one sentence explaining what this adds beyond existing knowledge

Do NOT generate hypotheses that:
- Are already captured by existing similar knowledge (check the list above)
- Are too generic (no specific error tokens, no specific environment)
- Contain only procedural steps without causal explanation
- Could apply to any project without modification

Respond with JSON only. No code fences."""

# ---------------------------------------------------------------------------
# Step 4: VALIDATE
# Input placeholders: {hypotheses_json}, {existing_nodes_json}
# ---------------------------------------------------------------------------

VALIDATE_PROMPT: str = """Validate proposed hypotheses against the agent's existing knowledge base.
Apply strict quality standards: REJECT anything generic or already covered.

PROPOSED HYPOTHESES (indexed 0-based):
{hypotheses_json}

EXISTING KNOWLEDGE NODES (with applies_when for overlap detection):
{existing_nodes_json}

For each hypothesis, classify it as ACCEPT, REJECT, MERGE, or CONTRADICT:
- ACCEPT: New causal knowledge with specific error tokens, not already captured
- REJECT if: no specific error tokens, no causal mechanism, too generic, or already covered
- REJECT if: the information_gain field is empty or restates what's already known
- MERGE: Same domain/error_family as existing node but adds precision or new environment detail
- CONTRADICT: Conflicts with existing knowledge (different fix for same error)

Produce a JSON object with exactly these keys:
- "accepted": list of integers (hypothesis indices to accept as new nodes)
- "rejected": list of objects, each with:
    - "index": integer, hypothesis index
    - "reason": string, why it was rejected
- "merges": list of objects, each with:
    - "hypothesis_index": integer, hypothesis index
    - "existing_node_id": integer, ID of the existing node to merge into
    - "merged_content": string, the combined/updated content for the node
- "contradictions": list of objects, each with:
    - "hypothesis_index": integer, hypothesis index
    - "existing_node_id": integer, ID of the contradicting node
    - "reason": string, brief description of the contradiction

A hypothesis can appear in at most one list.
Respond with JSON only. No code fences."""
