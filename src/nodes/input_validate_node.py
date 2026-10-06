"""AgentCore Platform v1.0"""

# HCR-C2-011 — InputValidateNode
# Inner domain node 1: the caller-data contract for this template.
#
# Distinct from PreProcessNode (trust gate + free-text screening): this node
# applies the DOMAIN rules — minimum meaningful length, query-term extraction
# for lexical matching against the guidelines knowledge base, and field-by-field
# validation of the structured caller parameters.
#
# Caller data arrives on TWO channels, both validated here:
#
#   input_context (structured invocation parameters, bridged into inner state —
#   see src/graph/context_bridge.py):
#     patient_context  -> de-identified free text describing the case; its terms
#                         widen retrieval. Redacted with the SAME direct-identifier
#                         screen PreProcessNode applies, because input_context does
#                         not pass through PreProcessNode and the framework's own
#                         input mask is scoped to user_input/validated_input.
#     top_k            -> number of candidate passages to retrieve (1..20)
#     score_threshold  -> retrieval-confidence floor; the caller may only make it
#                         STRICTER than the configured value, never looser
#
#   the string payload (validated_input, already screened by PreProcessNode):
#     the clinical question itself.
#
# Validation is fail-CLOSED. Every caller-supplied number goes through
# _finite_in_range(): bools, non-numerics, NaN, +/-Infinity and out-of-range
# magnitudes are all rejected. This matters because NaN parses cleanly through
# float() and arrives intact through raw JSON, while every IEEE comparison
# against NaN is False — a NaN confidence floor would silently admit every
# passage, which is a fail-OPEN on exactly the decision this template exists to
# make. Rejections name the FIELD and never echo the rejected value.
#
# Absent caller data is not an error: the pipeline degrades to the configured
# defaults rather than fabricating anything.
#
# Inner node — ANONYMOUS trust: the outer PreProcessNode (VERIFIED_EXTERNAL)
# already enforced caller trust, and the inner nodes must be ANONYMOUS so the
# outer invocation context passes through the subgraph boundary without being
# rejected.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import math
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import INPUT_REJECTED
from src.services.progress import emit_progress

from src.nodes.pre_process_node import strip_direct_identifiers
from src.schemas.state import to_json

logger = logging.getLogger(__name__)

# Minimum characters for a meaningful clinical question.
_MIN_QUESTION_CHARS = 3

# Structural caps on the caller-supplied case description.
_MAX_CONTEXT_CHARS = 1000

# Bounds on the caller-tunable retrieval parameters.
_MIN_TOP_K = 1
_MAX_TOP_K = 20
_MAX_SCORE_THRESHOLD = 1.0

# Fallback confidence floor when nothing was seeded from configuration. Kept in
# step with RerankFilterNode's own default.
_DEFAULT_SCORE_THRESHOLD = 0.75

# Very common words dropped before lexical matching (kept small and generic).
# "redacted" is in the set because the identifier screen replaces a matched
# token with [REDACTED]: without this, a question that mentions a record number
# would carry a meaningless extra term into the match and depress every score,
# quietly turning a well-formed question into an abstention.
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "for",
        "to",
        "in",
        "on",
        "and",
        "or",
        "is",
        "are",
        "what",
        "which",
        "how",
        "should",
        "with",
        "at",
        "be",
        "do",
        "does",
        "can",
        "redacted",
    }
)

_WORD_RE = re.compile(r"[a-z0-9\-]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _extract_terms(text: str) -> List[str]:
    """Lowercase, tokenise, drop stopwords, dedupe — deterministic order."""
    terms: List[str] = []
    seen: set[str] = set()
    for tok in _WORD_RE.findall(text.lower()):
        if tok in _STOPWORDS or len(tok) < 2:
            continue
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
    return terms


def _reject(field: str, reason: str, code: str = "INVALID_REQUEST") -> Dict[str, Any]:
    """Fail-closed rejection naming the offending FIELD only (no value echo)."""
    if code:
        # A value the caller can correct: the run COMPLETES carrying the
        # reason so the request can be sent again on the same conversation.
        emit_progress(INPUT_REJECTED)
        return {
            "status": AgentStatus.SUCCESS.value,
            "error_code": code,
            "error_log": [f"InputValidateNode: {field} {reason}"],
        }
    return {
        "status": AgentStatus.ERROR.value,
        "error_log": [f"InputValidateNode: {field} {reason}"],
    }


def _finite_in_range(
    raw: Any,
    field: str,
    low: float,
    high: float,
    integral: bool = False,
) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Parse a caller-supplied number, failing CLOSED on anything unbounded.

    Returns ``(value, None)`` on success or ``(None, error_dict)`` on rejection.
    ``None`` input is not an error — it means "not supplied on this channel".

    Rejected: booleans (``isinstance(True, int)`` is True in Python, so a bare
    isinstance check would let ``true`` through as 1), non-numeric types and
    strings, NaN, +/-Infinity, and any magnitude outside ``[low, high]``.
    """
    if raw is None:
        return None, None
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None, _reject(field, "must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, _reject(field, "must be a number")
    if not math.isfinite(value):
        return None, _reject(field, "must be a finite number")
    if value < low or value > high:
        return None, _reject(field, f"must be between {low} and {high}")
    if integral:
        if value != int(value):
            return None, _reject(field, "must be a whole number")
        return float(int(value)), None
    return value, None


class InputValidateNode(FunctionNode):
    """Validate the caller request and normalise it into a retrievable query.

    Input state keys:
        validated_input:           str  — screened question from PreProcessNode
                                          (falls back to user_input for direct
                                          node-level use)
        input_context:             dict — caller invocation parameters (read-only)
        retrieval_top_k:           int  — configured default, seeded by the graph
        retrieval_score_threshold: float — configured confidence floor, seeded by
                                          the graph; the caller may only tighten it

    Output state keys (partial dict):
        clinical_query:            str  — JSON: {question, length, terms, context_chars}
        retrieval_top_k:           int  — only when the caller supplied one
        retrieval_score_threshold: float — only when the caller supplied one
        status:                    str
        error_log:                 list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        question = raw.strip() if isinstance(raw, str) else ""
        input_context = state.get("input_context", {}) or {}
        if not isinstance(input_context, dict):
            return _reject("input_context", "must be an object")

        # ── Emptiness / length ────────────────────────────────────────────────
        if len(question) < _MIN_QUESTION_CHARS:
            logger.error("InputValidateNode: question too short or empty")
            emit_trace_event(
                "input_validate_failed",
                {"reason": "question_too_short", "field": "user_input"},
                state,
            )
            return _reject("user_input", "is empty or shorter than the minimum question length")

        # ── Caller case description (free text — same identifier screen) ──────
        raw_context = input_context.get("patient_context")
        case_text = ""
        if raw_context is not None:
            if not isinstance(raw_context, str):
                return _reject("input_context.patient_context", "must be a string")
            case_text = _WHITESPACE_RE.sub(" ", strip_direct_identifiers(raw_context)).strip()
            if len(case_text) > _MAX_CONTEXT_CHARS:
                return _reject(
                    "input_context.patient_context",
                    f"must be at most {_MAX_CONTEXT_CHARS} characters",
                )

        # ── Caller retrieval parameters (finite + bounded, fail CLOSED) ───────
        top_k, err = _finite_in_range(
            input_context.get("top_k"), "input_context.top_k", _MIN_TOP_K, _MAX_TOP_K, integral=True
        )
        if err:
            return err

        # The configured floor is the strictest value the deployment declared.
        # A caller may raise it (fewer, better-matched passages) but never lower
        # it, so a request can never widen the evidence base this template is
        # willing to ground an advisory answer on.
        seeded_floor = state.get("retrieval_score_threshold")
        try:
            floor = float(seeded_floor) if seeded_floor is not None else _DEFAULT_SCORE_THRESHOLD
        except (TypeError, ValueError):
            floor = _DEFAULT_SCORE_THRESHOLD
        if not math.isfinite(floor):
            floor = _DEFAULT_SCORE_THRESHOLD

        score_threshold, err = _finite_in_range(
            input_context.get("score_threshold"),
            "input_context.score_threshold",
            floor,
            _MAX_SCORE_THRESHOLD,
        )
        if err:
            return err

        # ── Query terms (question + the caller's case description) ────────────
        searchable = f"{question} {case_text}".strip()
        terms = _extract_terms(searchable)
        if not terms:
            logger.error("InputValidateNode: no searchable terms in the request")
            emit_trace_event(
                "input_validate_failed",
                {"reason": "no_searchable_terms", "field": "user_input"},
                state,
            )
            return _reject("user_input", "contains no searchable terms")

        clinical_query: Dict[str, Any] = {
            "question": question,
            "length": len(question),
            "terms": terms,
            "context_chars": len(case_text),
        }

        logger.info(
            "InputValidateNode: question_len=%d context_chars=%d terms=%d",
            len(question),
            len(case_text),
            len(terms),
        )
        emit_trace_event(
            "input_validate_complete",
            {
                "question_length": len(question),
                "context_chars": len(case_text),
                "term_count": len(terms),
                "top_k_overridden": top_k is not None,
                "threshold_overridden": score_threshold is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "clinical_query": to_json(clinical_query),
            "status": AgentStatus.SUCCESS.value,
        }
        # Only write a key the caller actually supplied — otherwise the value
        # seeded from configuration stays in force.
        if top_k is not None:
            out["retrieval_top_k"] = int(top_k)
        if score_threshold is not None:
            out["retrieval_score_threshold"] = score_threshold
        return out
