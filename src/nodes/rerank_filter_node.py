"""AgentCore Platform v1.0"""

# HCR-C2-011 — RerankFilterNode
# Inner domain node 3: rerank retrieved passages and filter by the RAG
# score_threshold.
#
# Life-safety note: the score_threshold is deliberately high (0.75 by
# default) — for a clinical question-answering agent it is safer to drop
# low-confidence evidence and return an "insufficient evidence" answer than to
# ground an advisory reply on weak retrieval.
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

# Life-safety default; raised (never lowered) via the state key
# retrieval_score_threshold.
_DEFAULT_SCORE_THRESHOLD = 0.75


class RerankFilterNode(FunctionNode):
    """Rerank candidate passages and filter by score_threshold.

    Reranks by the retrieval score (descending, stable by id) and keeps only
    passages at or above the score_threshold. A deployment may swap in a
    cross-encoder reranker; the filter contract is unchanged.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        retrieved_passages:        str   — JSON-serialised scored candidate list
        retrieval_score_threshold: float — confidence floor; seeded from
                                           configuration and optionally raised
                                           by the caller

    Output state keys (partial dict):
        filtered_passages: str   — JSON-serialised passages >= threshold
        status:            str
        error_log:         list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_passages"), [])

        # Runtime precedence: the value in State wins, then the declared
        # life-safety default of 0.75. ClinicalGuidelinesGraphNode
        # ._parent_config() forwards config/config.yaml's
        # retrieval.score_threshold, DomainWorkflowGraph._extra_initial_state()
        # seeds it, and InputValidateNode may raise it to a validated caller
        # value. State is the only route in — execute() takes state alone.
        raw_threshold = state.get("retrieval_score_threshold", _DEFAULT_SCORE_THRESHOLD)
        if raw_threshold is None:
            raw_threshold = _DEFAULT_SCORE_THRESHOLD
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            threshold = _DEFAULT_SCORE_THRESHOLD

        if not candidates:
            logger.warning("RerankFilterNode: no retrieved passages to rerank")
            emit_trace_event(
                "rerank_filter_skipped",
                {"reason": "no_candidates", "score_threshold": threshold},
                state,
            )
            return {
                "filtered_passages": to_json([]),
                "status": AgentStatus.SUCCESS.value,
            }

        # Rerank (score desc, stable by id) on a local copy, then threshold-filter.
        reranked = sorted(
            candidates,
            key=lambda p: (-float(p.get("score", 0.0)), str(p.get("id", ""))),
        )
        kept = [p for p in reranked if float(p.get("score", 0.0)) >= threshold]

        logger.info(
            "RerankFilterNode: candidates=%d kept=%d threshold=%.2f",
            len(candidates),
            len(kept),
            threshold,
        )
        emit_trace_event(
            "rerank_filter_complete",
            {
                "candidate_count": len(candidates),
                "kept_count": len(kept),
                "score_threshold": threshold,
            },
            state,
        )

        return {
            "filtered_passages": to_json(kept),
            "status": AgentStatus.SUCCESS.value,
        }
