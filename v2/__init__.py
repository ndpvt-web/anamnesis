"""Hermes v2: Aristotelian Memory Architecture for Agent Intelligence.

Five-layer memory stack:
  L1 AISTHESIS  - Session-scoped working memory (ephemeral, 30-min TTL)
  L2 EMPEIRIA   - Append-only event log (never modified, source of truth)
  L3 TECHNE     - Proven patterns from reflection (confidence 0.50-0.84)
  L4 EPISTEME   - Validated KG nodes (confidence >= 0.85, 2+ SUPPORTS, no CONTRADICTS)
  L5 SOPHIA     - Human-approved axioms (read-only, written once)

Zero external dependencies. stdlib only."""

__version__ = "2.0.0"
