"""AgentCore Platform v1.0"""

# src/graph/context_bridge.py — carries the caller's input_context across the
# outer -> inner graph boundary.
#
# Why this exists: GraphNode.execute() invokes the inner graph as
# `subgraph.invoke(user_input, session_id=..., ctx=...)` and does NOT forward
# the outer state's input_context. Without a bridge, inner-node reads of
# state["input_context"] (patient_context, top_k, score_threshold, channel)
# would always see {} through the full nested graph. The sanctioned subclass
# hooks carry it across:
#
#   ClinicalGuidelinesGraphNode.extract_input(state)  [runs BEFORE subgraph.invoke]
#       -> set_caller_input_context(state["input_context"])
#   DomainWorkflowGraph._extra_initial_state()        [runs INSIDE subgraph.invoke]
#       -> returns {"input_context": get_caller_input_context()}
#
# A ContextVar keeps the hand-off correct per thread/task, so concurrent
# invocations in one process cannot see each other's caller context.

from contextvars import ContextVar
from typing import Any, Optional

_CALLER_INPUT_CONTEXT: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "hcr_c2_011_caller_input_context", default=None
)


def set_caller_input_context(input_context: Optional[dict[str, Any]]) -> None:
    """Stash the outer graph's input_context for the imminent inner-graph invoke."""
    _CALLER_INPUT_CONTEXT.set(dict(input_context) if input_context else {})


def get_caller_input_context() -> dict[str, Any]:
    """Read (without consuming) the stashed input_context; {} when none was set."""
    return _CALLER_INPUT_CONTEXT.get() or {}
