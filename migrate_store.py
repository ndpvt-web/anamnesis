#!/usr/bin/env python3
"""Anamnesis store importer.

Imports agent knowledge stores from a previous installation layout into the
current Anamnesis data home. Generic: the source root is given explicitly;
any directory containing per-agent subdirectories with a
`learnings/learnings.db` or `hermes/hermes.db` store is recognized.

Safety:
  - Never modifies the source.
  - Never runs automatically; you invoke it with explicit paths.
  - Refuses to overwrite an existing target store unless --force is given.

Usage:
  python3 migrate_store.py --from-dir /path/to/old/agents [--to-dir PATH]
                           [--agent ID] [--dry-run] [--force]

Defaults: --to-dir is the current data home's agents directory
($ANAMNESIS_HOME or ~/.anamnesis). Without --agent, all agents found under
--from-dir are listed and imported.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anamnesis_config

# Recognized store layouts inside an agent directory:
#   (relative path, kind)
_STORE_LAYOUTS = [
    (Path("learnings") / "learnings.db", "v1"),
    (Path("hermes") / "hermes.db", "v2-legacy"),
    (Path("anamnesis.db"), "v2"),
]


def find_source_agents(from_dir: Path) -> dict:
    """Map agent id -> list of (relative store path, kind) found."""
    found = {}
    if not from_dir.is_dir():
        return found
    for agent_dir in sorted(from_dir.iterdir()):
        if not agent_dir.is_dir():
            continue
        stores = [(rel, kind) for rel, kind in _STORE_LAYOUTS
                  if (agent_dir / rel).is_file()]
        if stores:
            found[agent_dir.name] = stores
    return found


def _target_for(to_dir: Path, agent_id: str, kind: str) -> Path:
    if kind == "v1":
        return to_dir / agent_id / "learnings" / "learnings.db"
    return to_dir / agent_id / "anamnesis.db"


def import_agent(from_dir: Path, to_dir: Path, agent_id: str,
                 dry_run: bool = False, force: bool = False) -> dict:
    """Import one agent's stores. Returns a per-store result list."""
    results = []
    stores = find_source_agents(from_dir).get(agent_id, [])
    if not stores:
        return [{"agent": agent_id, "error": "no recognized stores found"}]
    for rel, kind in stores:
        src = from_dir / agent_id / rel
        dst = _target_for(to_dir, agent_id, kind)
        entry = {"agent": agent_id, "kind": kind,
                 "from": str(src), "to": str(dst)}
        if dst.exists() and not force:
            entry["status"] = "refused: target exists (use --force to overwrite)"
            results.append(entry)
            continue
        entry["status"] = "would copy" if dry_run else "copied"
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + ".importing")
            shutil.copyfile(str(src), str(tmp))
            tmp.replace(dst)
        results.append(entry)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-dir", required=True,
                        help="Root containing per-agent store directories")
    parser.add_argument("--to-dir",
                        help="Target agents directory "
                             "(default: current data home)")
    parser.add_argument("--agent", help="Import only this agent id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing target stores")
    args = parser.parse_args()

    from_dir = Path(args.from_dir).expanduser().resolve()
    to_dir = (Path(args.to_dir).expanduser().resolve() if args.to_dir
              else anamnesis_config.agents_dir())

    found = find_source_agents(from_dir)
    if not found:
        print(f"No recognized agent stores under {from_dir}")
        sys.exit(1)

    agent_ids = [args.agent] if args.agent else sorted(found)
    failures = 0
    for agent_id in agent_ids:
        for entry in import_agent(from_dir, to_dir, agent_id,
                                  dry_run=args.dry_run, force=args.force):
            if "error" in entry:
                print(f"ERROR {entry['agent']}: {entry['error']}")
                failures += 1
            else:
                print(f"{entry['status']}: [{entry['kind']}] "
                      f"{entry['from']} -> {entry['to']}")
                if entry["status"].startswith("refused"):
                    failures += 1
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
