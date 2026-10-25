#!/usr/bin/env python3
"""5-step LLM reflection pipeline: Narrate->Diagnose->Hypothesize->Validate->Store.
Triggered at session end by agent_learn.py.
LLM provider is resolved via anamnesis_config (any OpenAI-compatible
endpoint or native Anthropic); the pipeline degrades to 'failed' cleanly
when no API key is configured.
Import: from v2 import db as v2db, prompts, knowledge_graph as kg"""

import sqlite3
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from v2 import db as v2db
from v2 import prompts
from v2 import knowledge_graph as kg
from v2 import retrieval as v2ret
from v2 import llm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_EVENTS = 30
TIMEOUT_SECONDS = 120  # Budget for all 4 LLM calls (structured prompts need ~14s each)
PER_CALL_MAX = 30      # Max per LLM call (WHEN/THEN/BECAUSE responses need ~14s)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_reflection_pipeline(agent_id: str, session_id: str) -> dict:
    """Run the complete 5-step reflection pipeline.
    Opens its own DB connection.
    Returns: {"status": "completed"|"failed"|"skipped",
              "nodes_created": int, "edges_created": int,
              "run_id": int, "duration_ms": int, "error": str}"""
    start_ms = int(time.time() * 1000)

    db = v2db.open_db(agent_id)
    run_id = v2db.start_reflection_run(db, agent_id, session_id)

    def _fail(error_msg: str) -> dict:
        duration = int(time.time() * 1000) - start_ms
        v2db.complete_reflection_run(
            db, run_id, status="failed",
            duration_ms=duration, error_text=error_msg
        )
        db.close()
        print(f"[anamnesis] reflection failed: {error_msg}", file=sys.stderr)
        return {
            "status": "failed", "nodes_created": 0, "edges_created": 0,
            "run_id": run_id, "duration_ms": duration, "error": error_msg
        }

    try:
        # Step 1: Fetch session events (DB returns DESC, reverse to chronological)
        events = v2db.get_session_events(db, agent_id, session_id, limit=MAX_EVENTS)
        events.reverse()  # Chronological order for narration
        if len(events) < 3:
            duration = int(time.time() * 1000) - start_ms
            v2db.complete_reflection_run(
                db, run_id, status="skipped",
                events_processed=len(events), duration_ms=duration
            )
            db.close()
            print(f"[anamnesis] reflection skipped: only {len(events)} events", file=sys.stderr)
            return {
                "status": "skipped", "nodes_created": 0, "edges_created": 0,
                "run_id": run_id, "duration_ms": duration, "error": ""
            }

        event_ids = [e["id"] for e in events]
        slim_events = [_slim_event(e) for e in events]
        error_events = [_slim_event(e) for e in events
                        if e.get("event_type") == "tool_error" or e.get("error_text", "")]

        # Global timeout budget: shrinking deadline across all LLM calls
        deadline = time.time() + TIMEOUT_SECONDS

        def _remaining():
            return max(0.5, min(PER_CALL_MAX, deadline - time.time()))

        # Step 2: Narrate
        narration = _step_narrate(slim_events, timeout=_remaining())
        if narration is None:
            return _fail("narrate step returned None")
        if time.time() >= deadline:
            return _fail("timeout after narrate")

        # Step 3: Diagnose
        diagnosis = _step_diagnose(narration, error_events, timeout=_remaining())
        if diagnosis is None:
            return _fail("diagnose step returned None")
        if time.time() >= deadline:
            return _fail("timeout after diagnose")

        # Fetch similar nodes BEFORE hypothesize for contrastive analysis
        # Build query from error signatures if available
        error_sigs = narration.get("error_signatures", [])
        if error_sigs:
            sig_query = " ".join(
                s.get("error_text", "")[:50] for s in error_sigs[:3]
            )
        else:
            sig_query = narration.get("summary", "")[:150]
        similar_nodes = v2db.fts5_search(db, sig_query, agent_id, top_k=10)
        error_family = narration.get("error_family", "none")

        # Point 2: Tinyfish reflection enrichment (didaxis)
        # Search web for error signatures to provide external context
        web_context = ""
        try:
            from v2.tinyfish import enrich_with_web_context
            web_context = enrich_with_web_context(
                error_sigs, error_family=error_family
            )
        except Exception:
            pass  # Tinyfish failure must never block reflection

        # Step 4: Hypothesize (with contrastive context + web enrichment)
        hypotheses = _step_hypothesize(narration, diagnosis,
                                        similar_nodes=similar_nodes,
                                        web_context=web_context,
                                        timeout=_remaining())
        if hypotheses is None:
            return _fail("hypothesize step returned None")
        if not isinstance(hypotheses, list):
            hypotheses = hypotheses.get("hypotheses", []) if isinstance(hypotheses, dict) else []

        if not hypotheses:
            duration = int(time.time() * 1000) - start_ms
            v2db.complete_reflection_run(
                db, run_id, status="completed",
                events_processed=len(events), duration_ms=duration
            )
            db.close()
            return {
                "status": "completed", "nodes_created": 0, "edges_created": 0,
                "run_id": run_id, "duration_ms": duration, "error": ""
            }

        if time.time() >= deadline:
            return _fail("timeout after hypothesize")

        # Step 5: Validate (reuse similar_nodes, no extra DB query)
        validation = _step_validate(hypotheses, similar_nodes, timeout=_remaining())
        if validation is None:
            return _fail("validate step returned None")

        # Step 6: Store (with error_family for edge creation)
        store_result = _step_store(
            db, agent_id, session_id,
            hypotheses, validation, event_ids, run_id,
            error_family=error_family
        )

        duration = int(time.time() * 1000) - start_ms
        nodes_created = store_result.get("nodes_created", 0)
        edges_created = store_result.get("edges_created", 0)

        v2db.complete_reflection_run(
            db, run_id, status="completed",
            events_processed=len(events),
            nodes_created=nodes_created,
            edges_created=edges_created,
            duration_ms=duration,
            steps_json={
                "narration_summary": narration.get("summary", ""),
                "hypotheses_count": len(hypotheses),
                "accepted_count": len(validation.get("accepted", [])),
                "merges_count": len(validation.get("merges", [])),
                "contradictions_count": len(validation.get("contradictions", [])),
            }
        )
        db.close()

        print(
            f"[anamnesis] reflection completed: +{nodes_created} nodes, "
            f"+{edges_created} edges, {duration}ms",
            file=sys.stderr
        )
        return {
            "status": "completed",
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "run_id": run_id,
            "duration_ms": duration,
            "error": ""
        }

    except Exception as e:
        return _fail(str(e))


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _slim_event(e: dict) -> dict:
    """Reduce event to essential fields for reflection, cutting token count ~40%."""
    return {
        "type": e.get("event_type", ""),
        "tool": e.get("tool_name", ""),
        "cmd": (e.get("command", "") or "")[:200],
        "out": (e.get("tool_output", "") or "")[:300],
        "err": (e.get("error_text", "") or "")[:400],
        "exit": e.get("exit_code"),
        "ms": e.get("duration_ms"),
        "file": e.get("file_path", ""),
    }


SALIENCE_THRESHOLD = 0.4  # Minimum salience to store a hypothesis


def _compute_salience(hyp: dict) -> float:
    """Score hypothesis quality on 3 dimensions (0.0-1.0 each).
    Novelty: has specific error signature? Specificity: named errors, ports, paths?
    Actionability: has BECAUSE or FIX? Returns average score."""
    import re as _re
    content = hyp.get("content", "")
    # Novelty: error_signature present
    novelty = 1.0 if hyp.get("error_signature") else 0.3
    # Specificity: count technical tokens (ErrorTypes, ports, versions, file paths)
    specifics = len(_re.findall(
        r'[A-Z][a-z]+Error|port \d+|v\d+\.\d+|/[\w/]+\.py|0x[0-9a-f]+|\d{3,5}',
        content
    ))
    specificity = min(specifics / 3.0, 1.0)
    # Actionability: has causal structure
    actionability = 1.0 if ("FIX:" in content or "BECAUSE" in content) else 0.3
    return (novelty + specificity + actionability) / 3.0


def _step_narrate(events: list, timeout: float = 2.0) -> dict:
    """Step 1: Narrate the session from events.
    Returns: {summary, key_moments, tools_used, error_signatures, error_family}
    or None on failure."""
    events_json = json.dumps(events, default=str)
    prompt = prompts.NARRATE_PROMPT.format(events_json=events_json)
    result = _call_llm(prompts.REFLECTION_SYSTEM_PROMPT, prompt, timeout=timeout)
    if result is None:
        return None
    return {
        "summary": result.get("summary", ""),
        "key_moments": result.get("key_moments", []),
        "tools_used": result.get("tools_used", []),
        "error_signatures": result.get("error_signatures", []),
        "error_family": result.get("error_family", "none"),
    }


def _step_diagnose(narration: dict, error_events: list, timeout: float = 2.0) -> dict:
    """Step 2: Diagnose errors and patterns.
    Returns: {errors_diagnosed, patterns, successes}
    or None on failure."""
    narration_json = json.dumps(narration, default=str)
    error_events_json = json.dumps(error_events, default=str)
    prompt = prompts.DIAGNOSE_PROMPT.format(
        narration_json=narration_json,
        error_events_json=error_events_json
    )
    print(f"[anamnesis] diagnose: narration keys={list(narration.keys())}, "
          f"error_events={len(error_events)}, timeout={timeout:.1f}s", file=sys.stderr)
    result = _call_llm(prompts.REFLECTION_SYSTEM_PROMPT, prompt, timeout=timeout)
    if result is None:
        print(f"[anamnesis] diagnose: _call_llm returned None", file=sys.stderr)
        return None
    return {
        "errors_diagnosed": result.get("errors_diagnosed", []),
        "patterns": result.get("patterns", []),
        "successes": result.get("successes", []),
    }


def _step_hypothesize(narration: dict, diagnosis: dict,
                      similar_nodes: list = None,
                      web_context: str = "",
                      timeout: float = 2.0) -> list:
    """Step 3: Generate hypotheses (max 5) with contrastive analysis.
    Returns: list of {content, category, severity, applies_when, confidence,
             error_signature, error_family, information_gain}
    or None on failure."""
    narration_json = json.dumps(narration, default=str)
    diagnosis_json = json.dumps(diagnosis, default=str)
    # Slim similar nodes for contrastive context
    slim_similar = [
        {"id": n.get("id"), "content": n.get("content", "")[:200],
         "applies_when": n.get("applies_when", "")[:100],
         "layer": n.get("layer"), "confidence": n.get("confidence")}
        for n in (similar_nodes or [])[:10]
    ]
    similar_json = json.dumps(slim_similar, default=str)
    prompt = prompts.HYPOTHESIZE_PROMPT.format(
        narration_json=narration_json,
        diagnosis_json=diagnosis_json,
        similar_nodes_json=similar_json
    )
    # Append web context from Tinyfish (didaxis enrichment)
    if web_context:
        prompt += f"\n\n{web_context}\n\nUse the web search results above as additional evidence when forming hypotheses. Cross-reference with session observations."
    result = _call_llm(prompts.REFLECTION_SYSTEM_PROMPT, prompt, timeout=timeout)
    if result is None:
        return None
    hypotheses = result.get("hypotheses", result if isinstance(result, list) else [])
    return hypotheses[:5]


def _step_validate(hypotheses: list, existing_nodes: list, timeout: float = 2.0) -> dict:
    """Step 4: Validate hypotheses against existing knowledge.
    Returns: {accepted, rejected, merges, contradictions}
    or None on failure."""
    hypotheses_json = json.dumps(hypotheses, default=str)
    # Slim down existing nodes -- include applies_when for overlap detection
    slim_nodes = [
        {"id": n.get("id"), "content": n.get("content", "")[:300],
         "applies_when": n.get("applies_when", "")[:100],
         "layer": n.get("layer"), "confidence": n.get("confidence"),
         "category": n.get("category"), "error_family": n.get("error_family", "")}
        for n in existing_nodes
    ]
    existing_nodes_json = json.dumps(slim_nodes, default=str)
    prompt = prompts.VALIDATE_PROMPT.format(
        hypotheses_json=hypotheses_json,
        existing_nodes_json=existing_nodes_json
    )
    result = _call_llm(prompts.REFLECTION_SYSTEM_PROMPT, prompt, timeout=timeout)
    if result is None:
        return None
    return {
        "accepted": result.get("accepted", []),
        "rejected": result.get("rejected", []),
        "merges": result.get("merges", []),
        "contradictions": result.get("contradictions", []),
    }


def _step_store(db: sqlite3.Connection, agent_id: str, session_id: str,
                hypotheses: list, validation: dict,
                event_ids: list, run_id: int,
                error_family: str = "") -> dict:
    """Step 5: Write accepted hypotheses as kg_nodes + create edges.
    No LLM call -- pure DB writes.
    Edge creation mechanisms:
    1. Intra-session SUPPORTS between co-accepted hypotheses (weight=0.3)
    2. Error-family SUPPORTS to existing nodes in same family (weight=0.5)
    3. CONTRADICTS edges for contradictions
    4. DERIVED_FROM edges for merges
    Returns: {"nodes_created": int, "edges_created": int}"""
    nodes_created = 0
    edges_created = 0
    run_id_str = str(run_id)

    accepted_indices = set(validation.get("accepted", []))
    merges = {m["hypothesis_index"]: m for m in validation.get("merges", [])}
    contradictions_list = validation.get("contradictions", [])

    # Map hypothesis_index -> new node_id
    hypothesis_node_ids = {}

    # Handle accepted hypotheses -> create new nodes (with quality gate)
    for idx in accepted_indices:
        if idx >= len(hypotheses):
            continue
        h = hypotheses[idx]
        content = h.get("content", "").strip()
        if not content:
            continue

        # v3 Quality Gate: compute salience and reject platitudes
        salience = _compute_salience(h)
        if salience < SALIENCE_THRESHOLD:
            print(f"[anamnesis] quality gate REJECTED (salience={salience:.2f}): "
                  f"{content[:60]}", file=sys.stderr)
            continue

        # Use per-hypothesis error_family if available, fall back to session-level
        h_family = h.get("error_family", "") or error_family
        if h_family == "none":
            h_family = ""

        # v3 Domain classification at write time
        h_domain = v2ret.classify_domain(content)

        node_id = v2db.add_node(
            db,
            agent_id=agent_id,
            content=content,
            node_type=v2db.NODE_REFLECTION,
            layer=v2db.LAYER_TECHNE,
            summary=h.get("applies_when", "")[:200],
            applies_when=h.get("applies_when", ""),
            category=h.get("category", "general"),
            severity=h.get("severity", "medium"),
            confidence=float(h.get("confidence", 0.70)),
            source_session=session_id,
            origin_agent_id=agent_id,
            derived_from=event_ids[:10],
            reflection_run_id=run_id_str,
            maturity="candidate",
            error_family=h_family,
            domain=h_domain,
        )
        if node_id > 0:
            nodes_created += 1
            hypothesis_node_ids[idx] = node_id
            print(
                f"[anamnesis] stored node {node_id} (salience={salience:.2f}, "
                f"domain={h_domain}): {content[:60]}",
                file=sys.stderr
            )

    # Handle merges -> update existing node content + DERIVED_FROM edge
    for idx, merge in merges.items():
        if idx >= len(hypotheses):
            continue
        existing_node_id = merge.get("existing_node_id")
        merged_content = merge.get("merged_content", "").strip()
        if not existing_node_id or not merged_content:
            continue
        try:
            h = hypotheses[idx]
            h_family = h.get("error_family", "") or error_family
            if h_family == "none":
                h_family = ""
            db.execute(
                "UPDATE kg_nodes SET content = ?, last_seen = datetime('now'), "
                "error_family = CASE WHEN error_family = '' THEN ? ELSE error_family END "
                "WHERE id = ? AND agent_id = ?",
                (merged_content, h_family, existing_node_id, agent_id)
            )
            v2db.update_node_confidence(db, existing_node_id, v2db.CONFIDENCE_BOOST)
            db.execute(
                "UPDATE kg_nodes SET helpful_count = helpful_count + 1 WHERE id = ?",
                (existing_node_id,)
            )
            db.commit()
            hypothesis_node_ids[idx] = existing_node_id
            print(f"[anamnesis] merged into node {existing_node_id}: {merged_content[:60]}", file=sys.stderr)

            # DERIVED_FROM edge: trace merge lineage
            # Find the most recent node from this session (if any)
            recent = db.execute(
                "SELECT id FROM kg_nodes WHERE source_session = ? AND agent_id = ? "
                "AND id != ? ORDER BY id DESC LIMIT 1",
                (session_id, agent_id, existing_node_id)
            ).fetchone()
            if recent:
                eid = v2db.add_edge(db, agent_id, existing_node_id, recent["id"],
                                     v2db.EDGE_DERIVED_FROM, weight=1.0,
                                     metadata={"source": "merge", "run_id": run_id_str})
                if eid > 0:
                    edges_created += 1
        except Exception as e:
            print(f"[anamnesis] merge update failed: {e}", file=sys.stderr)

    # --- Intra-session SUPPORTS edges (co-accepted hypotheses share evidence) ---
    accepted_node_ids = [nid for nid in hypothesis_node_ids.values() if nid > 0]
    if len(accepted_node_ids) >= 2:
        anchor = accepted_node_ids[0]
        for other_id in accepted_node_ids[1:]:
            if anchor == other_id:
                continue
            eid = v2db.add_edge(db, agent_id, anchor, other_id,
                                v2db.EDGE_SUPPORTS, weight=0.3,
                                metadata={"source": "co_session", "run_id": run_id_str})
            if eid > 0:
                edges_created += 1

    # --- Error-family SUPPORTS edges (cross-session, same error domain) ---
    for idx, node_id in hypothesis_node_ids.items():
        if node_id <= 0 or idx >= len(hypotheses):
            continue
        h = hypotheses[idx]
        h_family = h.get("error_family", "") or error_family
        if h_family and h_family != "none":
            family_nodes = v2db.find_nodes_by_error_family(
                db, agent_id, h_family,
                exclude_ids=[node_id] + accepted_node_ids,
                limit=3
            )
            for fn in family_nodes:
                eid = v2db.add_edge(db, agent_id, node_id, fn["id"],
                                    v2db.EDGE_SUPPORTS, weight=0.5,
                                    metadata={"source": "error_family", "family": h_family})
                if eid > 0:
                    edges_created += 1

    # Handle contradictions -> add CONTRADICTS edges
    for contradiction in contradictions_list:
        hyp_idx = contradiction.get("hypothesis_index")
        existing_node_id = contradiction.get("existing_node_id")
        if hyp_idx is None or not existing_node_id:
            continue

        if hyp_idx not in hypothesis_node_ids:
            if hyp_idx < len(hypotheses):
                h = hypotheses[hyp_idx]
                content = h.get("content", "").strip()
                if content:
                    h_family = h.get("error_family", "") or error_family
                    if h_family == "none":
                        h_family = ""
                    node_id = v2db.add_node(
                        db, agent_id=agent_id, content=content,
                        node_type=v2db.NODE_REFLECTION, layer=v2db.LAYER_TECHNE,
                        category=h.get("category", "general"),
                        severity=h.get("severity", "medium"),
                        confidence=float(h.get("confidence", 0.70)),
                        source_session=session_id, origin_agent_id=agent_id,
                        derived_from=event_ids[:10], reflection_run_id=run_id_str,
                        maturity="candidate", error_family=h_family,
                    )
                    if node_id > 0:
                        nodes_created += 1
                        hypothesis_node_ids[hyp_idx] = node_id

        if hyp_idx in hypothesis_node_ids:
            new_node_id = hypothesis_node_ids[hyp_idx]
            edge_id = v2db.add_edge(
                db, agent_id, from_node_id=new_node_id, to_node_id=existing_node_id,
                edge_type=v2db.EDGE_CONTRADICTS, weight=1.0,
                metadata={"reason": contradiction.get("reason", "")}
            )
            if edge_id > 0:
                edges_created += 1

    return {"nodes_created": nodes_created, "edges_created": edges_created}


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_prompt: str,
              timeout: float = 30.0) -> dict:
    """Call the configured LLM. Returns parsed JSON dict or None.

    Provider, model, and credentials come from anamnesis_config via
    v2.llm; see docs/INTEGRATIONS.md for supported providers."""
    return llm.chat_json(system_prompt, user_prompt,
                         purpose="reflection", timeout=timeout,
                         max_tokens=4096)
