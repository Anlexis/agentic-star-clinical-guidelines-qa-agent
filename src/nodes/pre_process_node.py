"""AgentCore Platform v1.0"""

# HCR-C2-011 — PreProcessNode
# Outer backbone pre_process slot: caller trust gate + question screening.
#
# Responsibilities:
#   - Enforce VERIFIED_EXTERNAL caller trust (required_trust_level)
#   - Reject empty / oversized clinical questions early (fail-fast)
#   - Refuse prompt-injection-shaped payloads IN THIS NODE, so the refusal holds
#     even where no framework input gate runs in front of execute()
#   - Redact direct-identifier / protected-health-information-shaped tokens from
#     the free-text payload BEFORE validated_input is written, so raw
#     identifiers never reach the domain workflow or the checkpoint store
#   - Lock the caller-supplied channel label to an inert identifier
#   - Normalise the question and write validated_input + enriched_context
#   - Emit an audit event for every validation decision
#
# The clinical question is FREE TEXT (not JSON) — this is a question-answering
# template. Returns ONLY the state keys this node writes (partial-dict contract).
#
# The screens below are the template's OWN guarantees, asserted by calling
# execute() directly (no framework wrapper in front). Rejections are
# fail-CLOSED and name the offending FIELD — never the rejected value.

import logging
import re
from typing import Any, ClassVar, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import INPUT_REJECTED
from src.services.progress import emit_progress

from src.schemas.state import to_json

logger = logging.getLogger(__name__)

# Upper bound on a single clinical question (defensive; blocks payload-stuffing).
_MAX_QUERY_CHARS = 4000

# Caller-supplied labels that travel with the request are locked to an inert
# identifier shape: free text on a field that is stored and later rendered is
# caller-controlled output injection.
_INERT_LABEL_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# Direct-identifier patterns redacted before validated_input is written. The
# downstream nodes only ever operate on the normalised clinical question and on
# knowledge-base passage text, never on raw identifiers. Deliberately limited to
# HIGH-CONFIDENCE structural shapes — never freeform name guessing, which is out
# of scope and error-prone (see docs/02_design.md, "Safety Boundary").
_IDENTIFIER_PATTERNS: List[re.Pattern[str]] = [
    # Medical-record-number-style token (kept in sync with the output-gate
    # pattern in post_process_node.py — deliberately the same shape).
    re.compile(r"\bMRN[-:\s]?\d{6,10}\b", re.IGNORECASE),
    # National-identifier-shaped sequence (NNN-NN-NNNN).
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Telephone numbers (loose international/US form).
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    # Calendar-date-shaped tokens: a specific date of birth is a far stronger
    # direct-identifier signal than a bare age like "68yo", which is left
    # untouched because the domain nodes need age and criteria phrasing intact.
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]
_IDENTIFIER_REPLACEMENT = "[REDACTED]"

# Prompt-injection shapes refused outright. Each pattern targets an instruction
# to abandon the configured behaviour or to disclose the operating instructions
# — none of them is a way a clinician phrases a guidelines question. Ordinary
# clinical wording that merely contains "instructions", "system" or "ignore"
# does NOT match, and that direction is pinned by tests.
_INJECTION_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    (
        "override_instructions",
        re.compile(
            r"\b(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|the\s+)*"
            r"(?:previous|prior|above|preceding|earlier)\s+"
            r"(?:instructions?|prompts?|rules?|directions?|guidelines?)",
            re.IGNORECASE,
        ),
    ),
    (
        "disclose_instructions",
        re.compile(
            r"\b(?:reveal|repeat|print|show|output|display|expose)\s+"
            r"(?:me\s+|us\s+)?(?:your|the)\s+"
            r"(?:system\s+prompt|initial\s+prompt|hidden\s+prompt|"
            r"system\s+instructions?|configuration)",
            re.IGNORECASE,
        ),
    ),
    (
        "persona_override",
        re.compile(
            r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|"
            r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:a|an|the)\s+\w+\s+(?:with\s+)?no\s+restrictions|"
            r"pretend\s+(?:to\s+be|you\s+are)|enter\s+developer\s+mode|"
            r"developer\s+mode\s+enabled)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_tag_injection",
        re.compile(r"<\s*/?\s*(?:system|assistant|user)\s*>|\b(?:BEGIN|END)\s+SYSTEM\s+PROMPT\b", re.IGNORECASE),
    ),
]


def strip_direct_identifiers(text: str) -> str:
    """Redact direct-identifier-shaped tokens from caller free text."""
    for pattern in _IDENTIFIER_PATTERNS:
        text = pattern.sub(_IDENTIFIER_REPLACEMENT, text)
    return text


def detect_prompt_injection(text: str) -> Optional[str]:
    """Return the name of the first prompt-injection shape found, else None."""
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return name
    return None


def is_inert_label(value: Any) -> bool:
    """True when *value* is a lowercase identifier safe to store and render."""
    return isinstance(value, str) and bool(_INERT_LABEL_RE.match(value))


class PreProcessNode(FunctionNode):
    """Caller trust gate and question screening for HCR-C2-011.

    Screens the caller-supplied clinical question before the domain workflow
    runs. This is the outer backbone's pre_process slot — the only node
    requiring VERIFIED_EXTERNAL trust, so unauthenticated or anonymous callers
    are rejected here (fail-fast; the inner domain nodes carry ANONYMOUS trust
    and never see untrusted input directly).

    Input state keys:
        user_input:    str  — caller-supplied clinical question (free text)
        input_context: dict — caller invocation parameters (read-only); only
                              ``channel`` is consumed here, and it must be an
                              inert identifier

    Output state keys (partial dict):
        validated_input:  str        — screened, normalised question string
        enriched_context: str        — JSON-serialised channel metadata
        status:           str        — AgentStatus.SUCCESS.value or ERROR
        error_log:        list[str]  — set only on ERROR
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _reject(
        self, state: AgentState, reason: str, field: str, message: str, code: str = "INVALID_REQUEST"
    ) -> dict[str, Any]:
        """Fail-closed rejection: audit the reason, name the field, echo nothing."""
        logger.warning("PreProcessNode: rejected request — %s (%s)", reason, field)
        emit_trace_event(
            "pre_process_validation_failed",
            {"reason": reason, "field": field},
            state,
        )
        if code:
            # A value the caller can correct: the run COMPLETES carrying the
            # reason so the request can be sent again on the same conversation.
            emit_progress(INPUT_REJECTED)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": code,
                "error_log": [f"PreProcessNode: {message}"],
            }
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"PreProcessNode: {message}"],
        }

    def execute(self, state: AgentState) -> dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {}) or {}

        # ── Emptiness check ───────────────────────────────────────────────────
        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            return self._reject(state, "empty_input", "user_input", "user_input is empty or missing", "EMPTY_INPUT")

        question = user_input.strip()

        # ── Size check ────────────────────────────────────────────────────────
        if len(question) > _MAX_QUERY_CHARS:
            return self._reject(
                state,
                "query_too_long",
                "user_input",
                f"question exceeds {_MAX_QUERY_CHARS} chars",
                "QUESTION_TOO_LONG",
            )

        # ── Prompt-injection screen (owned here, not delegated) ───────────────
        # Enforced inside execute() so the refusal holds on any path that
        # reaches this node, including one with no framework input gate in
        # front of it. Nothing from the payload is carried forward.
        injection = detect_prompt_injection(question)
        if injection:
            return self._reject(
                state,
                f"prompt_injection:{injection}",
                "user_input",
                # Not a value the caller can correct: the content itself is
                # refused, so the run terminates rather than inviting a resend.
                "question contains an instruction-override pattern and was refused",
                "",
            )

        # ── Channel label must be inert ───────────────────────────────────────
        raw_channel = input_context.get("channel")
        if raw_channel is None:
            channel = "unknown"
        elif is_inert_label(raw_channel):
            channel = raw_channel
        else:
            return self._reject(
                state,
                "channel_not_inert",
                "input_context.channel",
                "input_context.channel must match [a-z0-9_]{1,32}",
            )

        # ── Direct-identifier redaction ───────────────────────────────────────
        screened = strip_direct_identifiers(question)
        redacted = screened != question

        # ── Success ───────────────────────────────────────────────────────────
        logger.info("PreProcessNode: accepted clinical question len=%d", len(screened))
        emit_trace_event(
            "pre_process_validated",
            {"question_length": len(screened), "identifiers_redacted": redacted},
            state,
        )

        return {
            "validated_input": screened,
            "enriched_context": to_json(
                {
                    "source": "ClinicalGuidelinesQAAgent",
                    "channel": channel,
                }
            ),
            "status": AgentStatus.SUCCESS.value,
        }
