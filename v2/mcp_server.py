#!/usr/bin/env python3
"""Anamnesis MCP Server (stdio, JSON-RPC 2.0).
Provides 5 tools for active agent cognition.
Works with any MCP client; see docs/INTEGRATIONS.md for configuration.
Import: from v2 import db, retrieval, feedback, knowledge_graph as kg"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2 import db as v2db
from v2 import retrieval
from v2 import feedback
from v2 import knowledge_graph as kg
from learnings_db import detect_active_agent

# MCP Protocol constants
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "anamnesis"
SERVER_VERSION = "3.1.0"

# 5 MCP Tools
TOOLS = [
    {
        "name": "anamnesis_recall",
        "description": "Search agent's knowledge base. Returns relevant learnings for a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                "layer": {
                    "type": "string",
                    "description": "Filter by layer: sophia|episteme|techne",
                    "enum": ["sophia", "episteme", "techne"]
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "anamnesis_record",
        "description": "Immediately record a learning without waiting for session-end reflection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The learning to record"},
                "category": {
                    "type": "string",
                    "description": "Category: anti_pattern|preference|principle|general",
                    "default": "general"
                },
                "severity": {
                    "type": "string",
                    "description": "Severity: critical|high|medium|low",
                    "default": "medium"
                },
                "applies_when": {
                    "type": "string",
                    "description": "When this learning applies",
                    "default": ""
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "anamnesis_feedback",
        "description": "Provide explicit feedback on a knowledge node (confirm, refute, or mark irrelevant).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "integer", "description": "The node ID to give feedback on"},
                "feedback": {
                    "type": "string",
                    "description": "Feedback type",
                    "enum": ["confirmed", "refuted", "irrelevant"]
                }
            },
            "required": ["node_id", "feedback"]
        }
    },
    {
        "name": "anamnesis_status",
        "description": "Get memory statistics: node counts by layer, feedback stats, health metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "anamnesis_graph",
        "description": "Explore the knowledge graph from a specific node. Returns related nodes and edges.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "integer", "description": "Starting node ID"},
                "max_depth": {
                    "type": "integer",
                    "description": "Max traversal depth (default 2)",
                    "default": 2
                }
            },
            "required": ["node_id"]
        }
    }
]


def _make_response(request_id, result: dict) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result
    }


def _make_error(request_id, code: int, message: str) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    }


def _make_tool_result(text: str, is_error: bool = False) -> dict:
    """Build a tools/call content result."""
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error
    }


def _handle_initialize(request_id) -> dict:
    """Handle MCP initialize. Returns capabilities with tools."""
    result = {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}
    }
    return _make_response(request_id, result)


def _handle_tools_list(request_id) -> dict:
    """Return list of 5 tools."""
    return _make_response(request_id, {"tools": TOOLS})


def _tool_anamnesis_recall(args: dict) -> dict:
    """Execute anamnesis_recall tool."""
    query = args.get("query", "").strip()
    if not query:
        return _make_tool_result("Error: query is required.", is_error=True)

    top_k = int(args.get("top_k", 5))
    layer_filter = args.get("layer")

    agent_id = detect_active_agent()
    if not agent_id:
        return _make_tool_result("Error: no active agent detected.", is_error=True)

    db = None
    try:
        db = v2db.open_db(agent_id)
        results = retrieval.dual_retrieve(db, agent_id, query, top_k=top_k)
        # Record surfaced nodes for feedback tracking
        if results:
            v2db.record_surfaced(db, agent_id, "", [r["id"] for r in results])
    except Exception as e:
        print(f"[anamnesis] anamnesis_recall error: {e}", file=sys.stderr)
        return _make_tool_result(f"Error: {e}", is_error=True)
    finally:
        if db:
            db.close()

    # Apply optional layer filter post-retrieval
    if layer_filter:
        results = [r for r in results if r.get("layer") == layer_filter]

    if not results:
        return _make_tool_result(f"No results found for query: {query}")

    lines = [f"## Anamnesis Recall: {query} ({agent_id})", ""]
    for i, r in enumerate(results, 1):
        layer = r.get("layer", "?").upper()
        conf = r.get("confidence", 0.0)
        cat = r.get("category", "general")
        content = r.get("content", "")
        node_id = r.get("id", "?")
        source = r.get("source", "fts5")
        severity = r.get("severity", "")
        maturity = r.get("maturity", "")

        header = f"{i}. [ID:{node_id}] [{layer}] [{conf:.2f}] [{cat}]"
        if severity and severity != "medium":
            header += f" [{severity.upper()}]"
        if maturity == "crystallized":
            header += " [CRYSTALLIZED]"
        header += f" (src:{source})"
        lines.append(header)
        lines.append(f"   {content}")

        applies_when = r.get("applies_when", "")
        if applies_when:
            lines.append(f"   Applies when: {applies_when}")

        graph_context = r.get("graph_context", "")
        if graph_context:
            lines.append(f"   Graph context: {graph_context}")

        lines.append("")

    return _make_tool_result("\n".join(lines))


def _tool_anamnesis_record(args: dict) -> dict:
    """Execute anamnesis_record tool."""
    content = args.get("content", "").strip()
    if not content:
        return _make_tool_result("Error: content is required.", is_error=True)

    category = args.get("category", "general")
    severity = args.get("severity", "medium")
    applies_when = args.get("applies_when", "")

    agent_id = detect_active_agent()
    if not agent_id:
        return _make_tool_result("Error: no active agent detected.", is_error=True)

    db = None
    try:
        db = v2db.open_db(agent_id)
        node_id = v2db.add_node(
            db,
            agent_id=agent_id,
            content=content,
            node_type=v2db.NODE_FACT,
            layer=v2db.LAYER_TECHNE,
            applies_when=applies_when,
            category=category,
            severity=severity,
        )
    except Exception as e:
        print(f"[anamnesis] anamnesis_record error: {e}", file=sys.stderr)
        return _make_tool_result(f"Error: {e}", is_error=True)
    finally:
        if db:
            db.close()

    if node_id == -1:
        return _make_tool_result(
            f"Learning already exists (duplicate detected). No new node created.\n"
            f"Content: {content}"
        )

    return _make_tool_result(
        f"Recorded learning.\n"
        f"Node ID: {node_id}\n"
        f"Layer: techne\n"
        f"Category: {category}\n"
        f"Severity: {severity}\n"
        f"Content: {content}"
    )


def _tool_anamnesis_feedback(args: dict) -> dict:
    """Execute anamnesis_feedback tool."""
    node_id = args.get("node_id")
    feedback_val = args.get("feedback", "").strip()

    if node_id is None:
        return _make_tool_result("Error: node_id is required.", is_error=True)
    if feedback_val not in ("confirmed", "refuted", "irrelevant"):
        return _make_tool_result(
            "Error: feedback must be 'confirmed', 'refuted', or 'irrelevant'.",
            is_error=True
        )

    agent_id = detect_active_agent()
    if not agent_id:
        return _make_tool_result("Error: no active agent detected.", is_error=True)

    db = None
    try:
        db = v2db.open_db(agent_id)
        result = feedback.apply_explicit_feedback(db, int(node_id), feedback_val)
    except Exception as e:
        print(f"[anamnesis] anamnesis_feedback error: {e}", file=sys.stderr)
        return _make_tool_result(f"Error: {e}", is_error=True)
    finally:
        if db:
            db.close()

    new_conf = result.get("new_confidence", 0.0)
    if new_conf == 0.0 and result.get("node_id") == node_id:
        # Likely node not found
        return _make_tool_result(
            f"Node {node_id} not found or feedback had no effect.",
            is_error=True
        )

    return _make_tool_result(
        f"Feedback applied.\n"
        f"Node ID: {result['node_id']}\n"
        f"Feedback: {result['feedback']}\n"
        f"New confidence: {new_conf:.3f}"
    )


def _tool_anamnesis_status(args: dict) -> dict:
    """Execute anamnesis_status tool."""
    agent_id = detect_active_agent()
    if not agent_id:
        return _make_tool_result("Error: no active agent detected.", is_error=True)

    db = None
    try:
        db = v2db.open_db(agent_id)
        stats = v2db.get_stats(db, agent_id)
    except Exception as e:
        print(f"[anamnesis] anamnesis_status error: {e}", file=sys.stderr)
        return _make_tool_result(f"Error: {e}", is_error=True)
    finally:
        if db:
            db.close()

    nodes_by_layer = stats.get("nodes_by_layer", {})
    maturity = stats.get("maturity", {})

    lines = [
        f"## Anamnesis Status: {agent_id}",
        "",
        "### Node Counts by Layer",
        f"  Sophia (axioms):    {nodes_by_layer.get('sophia', 0)}",
        f"  Episteme:           {nodes_by_layer.get('episteme', 0)}",
        f"  Techne:             {nodes_by_layer.get('techne', 0)}",
        f"  Total active:       {stats.get('total_nodes', 0)}",
        "",
        "### Maturity Breakdown",
    ]
    for mat_key, mat_cnt in sorted(maturity.items()):
        lines.append(f"  {mat_key}: {mat_cnt}")

    lines += [
        "",
        "### Activity",
        f"  Total events:       {stats.get('total_events', 0)}",
        f"  Total edges:        {stats.get('total_edges', 0)}",
        f"  Axioms (sophia):    {stats.get('axioms', 0)}",
        f"  Reflection runs:    {stats.get('reflection_runs', 0)}",
    ]

    return _make_tool_result("\n".join(lines))


def _tool_anamnesis_graph(args: dict) -> dict:
    """Execute anamnesis_graph tool."""
    node_id = args.get("node_id")
    if node_id is None:
        return _make_tool_result("Error: node_id is required.", is_error=True)

    max_depth = int(args.get("max_depth", 2))

    agent_id = detect_active_agent()
    if not agent_id:
        return _make_tool_result("Error: no active agent detected.", is_error=True)

    db = None
    try:
        db = v2db.open_db(agent_id)
        # Fetch the seed node itself
        seed_node = v2db.get_node(db, int(node_id))
        if not seed_node:
            return _make_tool_result(f"Node {node_id} not found.", is_error=True)
        related = kg.find_related_nodes(
            db, agent_id, int(node_id),
            max_depth=max_depth,
            max_nodes=20
        )
    except Exception as e:
        print(f"[anamnesis] anamnesis_graph error: {e}", file=sys.stderr)
        return _make_tool_result(f"Error: {e}", is_error=True)
    finally:
        if db:
            db.close()

    lines = [
        f"## Anamnesis Graph: Node {node_id} ({agent_id})",
        "",
        "### Seed Node",
        f"  ID:         {seed_node.get('id')}",
        f"  Layer:      {seed_node.get('layer', '?')}",
        f"  Category:   {seed_node.get('category', '?')}",
        f"  Confidence: {seed_node.get('confidence', 0.0):.3f}",
        f"  Maturity:   {seed_node.get('maturity', '?')}",
        f"  Content:    {seed_node.get('content', '')}",
        "",
    ]

    if not related:
        lines.append("No related nodes found.")
    else:
        lines.append(f"### Related Nodes ({len(related)} found, max_depth={max_depth})")
        lines.append("")
        for r in related:
            depth = r.get("depth", "?")
            edge_type = r.get("edge_type", "?")
            rid = r.get("id", "?")
            rlayer = r.get("layer", "?").upper()
            rconf = r.get("confidence", 0.0)
            rcontent = r.get("content", "")
            lines.append(
                f"  [depth:{depth}] [{edge_type}] [ID:{rid}] [{rlayer}] "
                f"[{rconf:.2f}] {rcontent}"
            )

    return _make_tool_result("\n".join(lines))


def _handle_tools_call(request_id, tool_name: str, arguments: dict) -> dict:
    """Dispatch to appropriate tool handler. Returns JSON-RPC result."""
    dispatch = {
        "anamnesis_recall": _tool_anamnesis_recall,
        "anamnesis_record": _tool_anamnesis_record,
        "anamnesis_feedback": _tool_anamnesis_feedback,
        "anamnesis_status": _tool_anamnesis_status,
        "anamnesis_graph": _tool_anamnesis_graph,
    }

    handler = dispatch.get(tool_name)
    if handler is None:
        result = _make_tool_result(f"Unknown tool: {tool_name}", is_error=True)
    else:
        try:
            result = handler(arguments or {})
        except Exception as e:
            print(f"[anamnesis] tool {tool_name} unhandled error: {e}", file=sys.stderr)
            result = _make_tool_result(f"Internal error: {e}", is_error=True)

    return _make_response(request_id, result)


def main():
    """stdio MCP server main loop.
    Reads JSON-RPC 2.0 messages from stdin, writes responses to stdout.
    Handles: initialize, tools/list, tools/call.
    Agent identity resolves via anamnesis_config (env, config, or
    provisioned default). Opens DB connection per request (stateless)."""
    print(f"[anamnesis] MCP server starting (pid={__import__('os').getpid()})", file=sys.stderr)

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError as e:
            print(f"[anamnesis] JSON parse error: {e}", file=sys.stderr)
            # Cannot determine request id; skip
            continue

        request_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}

        print(f"[anamnesis] method={method} id={request_id}", file=sys.stderr)

        if method == "initialize":
            response = _handle_initialize(request_id)

        elif method == "tools/list":
            response = _handle_tools_list(request_id)

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments") or {}
            response = _handle_tools_call(request_id, tool_name, arguments)

        elif method == "notifications/initialized":
            # Notification -- no response needed
            continue

        else:
            # Unknown method
            response = _make_error(request_id, -32601, f"Method not found: {method}")

        try:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"[anamnesis] write error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
