"""AgentCore Platform v1.0"""

# HCR-C2-011 — src.graph package exports.
#
# The manifest (config/agent.yaml) points at this agent with the dotted path
#   class: "src.graph.graph.ClinicalGuidelinesQAAgent"
# and the agent registry also resolves package-level lookups of the form
#   getattr(import_module("src.graph"), "ClinicalGuidelinesQAAgent").
# Re-exporting the outer graph class here (together with the `Graph` alias that
# src/api/server.py imports) keeps both forms of resolution working.

from src.graph.graph import ClinicalGuidelinesQAAgent, Graph

__all__ = ["ClinicalGuidelinesQAAgent", "Graph"]
