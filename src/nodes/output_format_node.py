"""AgentCore Platform v1.0"""

# HCR-C2-011 — OutputFormatNode (inner domain node 5, last in DomainWorkflowGraph)
# Assembles the final answer document from generated_answer + citations.
# This is the last inner node — it produces the `answer` string that the outer
# PostProcessNode passes through the output gate.
#
# Life-safety: attaches the advisory-only disclaimer if an upstream path
# omitted it. Its presence on the caller-facing response is separately ENFORCED
# by PostProcessNode, which blocks a response that lacks it.
#
# Inner node — ANONYMOUS trust.
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.schemas.state import from_json

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 72
_SUBSEP = "-" * 72

# Kept verbatim in step with GenerateAnswerNode.DISCLAIMER and
# PostProcessNode.DISCLAIMER (life-safety guarantee).
DISCLAIMER = "This output is advisory only. Clinical judgment of the attending " "physician takes precedence."


def _assemble(answer_body: str, citations: List[Dict[str, Any]]) -> str:
    """Assemble the final clinical Q&A answer with a citations block."""
    lines: List[str] = [
        _SEPARATOR,
        "CLINICAL GUIDELINES Q&A — ADVISORY RESPONSE",
        _SEPARATOR,
        "",
        answer_body,
        "",
        "Citations",
        _SUBSEP,
    ]
    if citations:
        for c in citations:
            lines.append(f"  - [{c.get('id', 'N/A')}] {c.get('title', '')} — {c.get('source', '')}")
    else:
        lines.append("  (no grounded citations — see advisory above)")

    body = "\n".join(lines)

    # Life-safety guarantee: ensure the advisory disclaimer is present.
    if DISCLAIMER not in body:
        body = f"{body}\n\n{DISCLAIMER}"
    return body


class OutputFormatNode(FunctionNode):
    """Assemble the final advisory answer document (inner domain node).

    Reads generated_answer and citations from State, renders the final
    answer text with a citations block and the advisory disclaimer, and
    writes it to `answer` (and result) for the outer PostProcessNode.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        generated_answer: str  — grounded advisory answer text
        citations:        str  — JSON-serialised citation list

    Output state keys (partial dict):
        answer: str
        result: str  (same as answer — backbone convention)
        status: str
        error_log: list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        generated_answer = state.get("generated_answer") or ""
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), [])

        # ── Fallback for empty upstream answer ────────────────────────────────
        if not generated_answer.strip():
            logger.warning("OutputFormatNode: generated_answer empty — using advisory fallback")
            emit_trace_event(
                "output_format_fallback",
                {"reason": "empty_generated_answer"},
                state,
            )
            fallback = "No answer could be generated for this clinical question. " f"\n\n{DISCLAIMER}"
            return {
                "answer": fallback,
                "result": fallback,
                "status": AgentStatus.SUCCESS.value,
            }

        answer = _assemble(generated_answer, citations)

        logger.info(
            "OutputFormatNode: answer_chars=%d citations=%d",
            len(answer),
            len(citations),
        )
        emit_trace_event(
            "output_format_complete",
            {"answer_length": len(answer), "citation_count": len(citations)},
            state,
        )

        return {
            "answer": answer,
            "result": answer,
            "status": AgentStatus.SUCCESS.value,
        }
