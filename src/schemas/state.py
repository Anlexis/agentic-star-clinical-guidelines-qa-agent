"""AgentCore Platform v1.0"""

# State must be a flat TypedDict — never a Pydantic BaseModel. Checkpoints use
# msgpack serialization, and Pydantic objects cause silent corruption. Extend
# AgentState with agent-specific fields only. Do NOT add credentials, secrets,
# or Pydantic models.
#
# HCR-C2-011 — Clinical Guidelines Q&A Agent (two-layer nested graph).
# The outer backbone (AgentBaseGraph) and the inner domain workflow (BaseGraph)
# share this one schema; the fields below cover both layers.
#
# Serialization contract: all dict/list-valued fields are stored as
# JSON-serialized Optional[str]. Use the to_json() / from_json() helpers below
# at every producer and every consumer — one contract end-to-end. Never type a
# dict/list field as a bare dict/list; that causes msgpack serialization
# failures.
#
# Life-safety note: this template answers over a SYNTHETIC clinical-guidelines
# knowledge base only — no patient records and no personal health information
# are stored in State. Caller free text is screened for direct-identifier
# shapes by PreProcessNode (and, on the context channel, by InputValidateNode)
# before it is written to any state field.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a value to a JSON string for State storage."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON string from State storage."""
    if value is None:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for HCR-C2-011 clinical guidelines question answering.

    All shared fields (user_input, input_context, status, session_id,
    node_history, error_log, hitl_*, etc.) are inherited from AgentState.

    dict/list fields are stored as JSON-serialized Optional[str].
    formatted_output is NOT re-declared here — it is inherited from AgentState.
    """

    # ------------------------------------------------------------------
    # Outer layer — set by PreProcessNode (pre_process backbone slot)
    # ------------------------------------------------------------------

    # Screened, normalised clinical question produced by PreProcessNode;
    # consumed by the inner InputValidateNode.
    validated_input: NotRequired[Optional[str]]

    # JSON-serialised channel/request metadata dict (stored as str).
    # Shape: {"source": str, "channel": str}
    enriched_context: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Retrieval parameters. Seeded from config/config.yaml's `retrieval`
    # block by DomainWorkflowGraph._extra_initial_state(), then optionally
    # narrowed to a validated caller value by InputValidateNode, so the
    # declared settings are observed on the real .invoke() path. Nodes read
    # these seeded values first and fall back to their declared defaults.
    # Scalar int/float — not dict/list, so the JSON-string rule above does
    # not apply.
    # ------------------------------------------------------------------

    retrieval_top_k: NotRequired[Optional[int]]
    retrieval_score_threshold: NotRequired[Optional[float]]

    # ------------------------------------------------------------------
    # Inner layer — domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # JSON-serialised normalised query dict (stored as str).
    # Shape: {"question": str, "length": int, "terms": list[str], "context_chars": int}
    clinical_query: NotRequired[Optional[str]]

    # JSON-serialised list of retrieved candidate passages (stored as str).
    # Each item: {"id": str, "title": str, "source": str, "text": str, "score": float}
    retrieved_passages: NotRequired[Optional[str]]

    # JSON-serialised list of passages surviving the score_threshold filter
    # (stored as str). Same item shape as retrieved_passages.
    filtered_passages: NotRequired[Optional[str]]

    # Grounded advisory answer text produced by GenerateAnswerNode.
    generated_answer: NotRequired[Optional[str]]

    # JSON-serialised list of citation dicts (stored as str).
    # Each item: {"id": str, "title": str, "source": str}
    citations: NotRequired[Optional[str]]

    # Final formatted answer document (grounded answer + citations + advisory
    # disclaimer). Assembled by the inner OutputFormatNode.
    answer: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Outer layer — set by PostProcessNode (post_process backbone slot)
    # ------------------------------------------------------------------

    # Primary result surfaced to the caller. Set to the gated answer content
    # once the output gate has passed; formatted_output (from AgentState) is
    # set by PostProcessNode alongside it.
    result: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit — framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: NotRequired[Optional[str]]
    correlation_id: NotRequired[Optional[str]]
    error_code: Optional[str]
    # node_history inherited from AgentState
