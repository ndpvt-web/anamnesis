#!/usr/bin/env python3
"""Anamnesis command-line interface.

Shell entry points for harnesses that integrate by running commands, and
for humans inspecting their agent's memory:

  python3 anamnesis_cli.py briefing [--query TEXT] [--agent ID]
  python3 anamnesis_cli.py record "content" [--category C] [--severity S]
  python3 anamnesis_cli.py status
  python3 anamnesis_cli.py doctor

Identity resolution follows anamnesis_config.resolve_agent_id; pass --agent
or --cwd to influence it explicitly.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anamnesis_config


def _resolve(args) -> str:
    hook_input = {}
    if getattr(args, "cwd", None):
        hook_input["cwd"] = args.cwd
    return getattr(args, "agent", None) or anamnesis_config.resolve_agent_id(hook_input)


def cmd_briefing(args) -> int:
    from v2 import db as v2db
    from v2 import retrieval
    agent_id = _resolve(args)
    db = v2db.open_db(agent_id)
    try:
        if args.query:
            from v2.retrieval import classify_error_family
            results = retrieval.dual_retrieve(
                db, agent_id, args.query, top_k=args.top_k,
                error_family=classify_error_family(args.query))
            text = retrieval.format_query_based_briefing(db, agent_id, results)
        else:
            text = retrieval.format_session_briefing(db, agent_id)
    finally:
        db.close()
    print(text or f"(no knowledge yet for agent '{agent_id}')")
    return 0


def cmd_record(args) -> int:
    from v2 import db as v2db
    from v2 import retrieval as v2ret
    agent_id = _resolve(args)
    db = v2db.open_db(agent_id)
    try:
        node_id = v2db.add_node(
            db, agent_id=agent_id, content=args.content,
            node_type=v2db.NODE_FACT, layer=v2db.LAYER_TECHNE,
            category=args.category, severity=args.severity,
            applies_when=args.applies_when,
            domain=v2ret.classify_domain(args.content))
    finally:
        db.close()
    if node_id == -1:
        print("Duplicate: an equivalent learning already exists.")
    else:
        print(f"Recorded node {node_id} for agent '{agent_id}'.")
    return 0


def cmd_status(args) -> int:
    from v2 import db as v2db
    agent_id = _resolve(args)
    db = v2db.open_db(agent_id)
    try:
        stats = v2db.get_stats(db, agent_id)
    finally:
        db.close()
    print(json.dumps({"agent_id": agent_id,
                      "data_home": str(anamnesis_config.data_home()),
                      **stats}, indent=2, default=str))
    return 0


def cmd_doctor(args) -> int:
    import install as install_mod
    return install_mod.doctor(agent=getattr(args, "agent", None),
                              cwd=getattr(args, "cwd", None))


def main() -> int:
    parser = argparse.ArgumentParser(description="Anamnesis CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_identity(p):
        p.add_argument("--agent", help="Agent id override")
        p.add_argument("--cwd", help="Project directory for identity resolution")

    p = sub.add_parser("briefing", help="Print a knowledge briefing")
    p.add_argument("--query", help="Query-based briefing instead of layer dump")
    p.add_argument("--top-k", type=int, default=5)
    add_identity(p)
    p.set_defaults(func=cmd_briefing)

    p = sub.add_parser("record", help="Record a learning immediately")
    p.add_argument("content")
    p.add_argument("--category", default="general")
    p.add_argument("--severity", default="medium")
    p.add_argument("--applies-when", default="")
    add_identity(p)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("status", help="Memory statistics")
    add_identity(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", help="Run installation diagnostics")
    add_identity(p)
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
