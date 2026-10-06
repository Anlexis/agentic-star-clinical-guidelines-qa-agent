"""AgentCore Platform v1.0"""

# HCR-C2-011 — GenerateAnswerNode
# Inner domain node 4: generate a grounded advisory answer from the filtered
# guideline passages.
#
# This implementation performs DETERMINISTIC, extractive grounding — it makes
# no language-model call. The answer is synthesised ONLY from the passages that
# survived the score_threshold filter, so there is no ungrounded generation.
# The declared language-model settings (config/config.yaml `llm` —
# system_prompt_template, temperature, max_tokens; prompts/hcr_qa.j2) describe
# the wiring point for a real model at temperature 0.0 and are deliberately not
# exercised by this code path. A model call is never faked here.
#
# Life-safety guarantees:
#   - Grounded ONLY in filtered_passages (never fabricate clinical guidance).
#   - If no passage survives the filter, return an explicit "insufficient
#     evidence" advisory rather than an unsupported answer.
#   - The advisory-only disclaimer is always attached.
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

from src.schemas.state import from_json, to_json

logger = logging.getLogger(__name__)

# Mandatory advisory-only disclaimer. Its presence on every caller-facing
# response is ENFORCED by PostProcessNode (the output gate).
DISCLAIMER = "This output is advisory only. Clinical judgment of the attending " "physician takes precedence."

_INSUFFICIENT_EVIDENCE = (
    "Insufficient high-confidence evidence was retrieved from the clinical "
    "guidelines knowledge base to answer this question reliably. Please "
    "consult the primary guideline sources or a specialist before acting."
)


def _compose_answer(question: str, passages: List[Dict[str, Any]]) -> str:
    """Deterministically compose a grounded answer from filtered passages."""
    lines: List[str] = [
        f"Question: {question}",
        "",
        f"Based on {len(passages)} matching clinical guideline passage(s):",
    ]
    for p in passages:
        lines.append(f"  - [{p.get('id', 'N/A')}] {p.get('title', '')}: {p.get('text', '')}")
    return "\n".join(lines)


class GenerateAnswerNode(FunctionNode):
    """Generate a grounded advisory answer for the clinical question.

    Deterministic, extractive grounding over filtered_passages — no
    language-model call. The declared `llm` settings are the wiring point for a
    real model at temperature 0.0 and are not exercised here.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        clinical_query:    str  — JSON-serialised query dict
        filtered_passages: str  — JSON-serialised passages >= threshold

    Output state keys (partial dict):
        generated_answer: str  — grounded advisory answer text (with disclaimer)
        citations:        str  — JSON-serialised list of citation dicts
        status:           str
        error_log:        list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        query: Dict[str, Any] = from_json(state.get("clinical_query"), {})
        passages: List[Dict[str, Any]] = from_json(state.get("filtered_passages"), [])
        question = str(query.get("question", "")) or "(question unavailable)"

        # Deterministic, extractive grounding — no language-model call. The
        # declared llm settings (system_prompt_template / temperature /
        # max_tokens) are the wiring point for a real model and are deliberately
        # not consumed here. The answer below is composed ONLY from
        # filtered_passages, so it can never rest on ungrounded generation.

        # ── No grounded evidence → explicit insufficient-evidence advisory ────
        if not passages:
            answer = f"{_INSUFFICIENT_EVIDENCE}\n\n{DISCLAIMER}"
            logger.info("GenerateAnswerNode: no filtered passages — insufficient-evidence advisory")
            emit_trace_event(
                "generate_answer_insufficient_evidence",
                {"question_length": len(question)},
                state,
            )
            return {
                "generated_answer": answer,
                "citations": to_json([]),
                "status": AgentStatus.SUCCESS.value,
            }

        # ── Grounded answer ───────────────────────────────────────────────────
        body = _compose_answer(question, passages)
        answer = f"{body}\n\n{DISCLAIMER}"

        citations = [{"id": p.get("id"), "title": p.get("title"), "source": p.get("source")} for p in passages]

        logger.info(
            "GenerateAnswerNode: grounded answer over %d passage(s), answer_chars=%d",
            len(passages),
            len(answer),
        )
        emit_trace_event(
            "generate_answer_complete",
            {"passage_count": len(passages), "answer_length": len(answer)},
            state,
        )

        return {
            "generated_answer": answer,
            "citations": to_json(citations),
            "status": AgentStatus.SUCCESS.value,
        }
