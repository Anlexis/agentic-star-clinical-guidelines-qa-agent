"""AgentCore Platform v1.0"""

# HCR-C2-011 — PostProcessNode
# Outer backbone post_process slot: the caller-facing output boundary.
#
# This node is the ONE place every response passes through, so both output
# invariants this template states are enforced HERE rather than in a domain node
# a later change could accidentally skip:
#
#   Layer 1 — disallowed-content block. The answer is scanned for credential
#             shapes (API keys, JWTs, bearer tokens, credential assignments) and
#             for direct-identifier shapes (medical-record numbers,
#             national-identifier sequences, e-mail addresses, telephone
#             numbers). A hit returns the closed-set error envelope with
#             status=ERROR (see _contain / error_envelope below).
#
#             Credential recognition is the union of the domain patterns below
#             and the FRAMEWORK's own detect_credentials(). The framework scans
#             every node result with that detector and RAISES on a hit, and a
#             raise makes BaseNode.__call__ discard the node's whole return —
#             including the containment this gate applies. A shape the framework
#             catches and this gate missed is therefore a containment BYPASS,
#             not merely a narrower gate: the response is refused, but by an
#             uncontained path that surfaces a traceback instead of the stub.
#             Consulting the framework detector here keeps the two from drifting
#             apart by construction (measured: sk_live_/AKIA/conn-string shapes
#             were all missed by the domain set alone).
#
#   Layer 2 — mandatory advisory disclaimer. Every SUCCESSFUL response carries
#             the advisory-only disclaimer verbatim. It is attached if absent and
#             then VERIFIED; a response that still lacks it is blocked rather
#             than shipped, so a future edit cannot silently emit
#             disclaimer-less clinical text.
#
# The two layers are independent and each emits its own audit event.
#
# ORDER NOTE: the pattern scan runs on the answer text exactly as produced, and
# this gate performs NO numeric rewriting of the output. That matters: a gate
# that rewrites numbers before scanning can destroy the very shape an identifier
# pattern matches (turning "123-45-6789" into something the scan no longer
# recognises) and ship the result. This template renders no monetary aggregates
# and applies no rounding grid, so no such rewrite exists — and the domain
# identifiers it does render (guideline citation keys such as GL-HTN-001) pass
# through byte-identical. Both directions are pinned by tests.
#
# The output gate is a MODULE-LEVEL function (_security_gate_output) called from
# inside execute() — not an instance method on the node class, which the
# framework would auto-wrap into the graph chain.
#
# Returns ONLY the state keys this node writes (partial-dict contract).

import logging
import re
from typing import Any, ClassVar, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import TrustLevel
from framework.security.credential_detector import detect_credentials
from shared.utils.audit_logger import emit_trace_event
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, TOO_LONG

logger = logging.getLogger(__name__)

# Content that must never appear in a caller-facing answer.
# Order matters — most specific first.
_DISALLOWED_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    # API key shapes: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # Signed token: three base64url segments separated by dots.
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")),
    # Bearer token in an authorization-like context.
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{8,}", re.IGNORECASE)),
    # Credential assignment.
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
    # Direct-identifier shapes. The knowledge base is a curated guideline corpus
    # that carries none of these, so a match can only have come from caller text
    # echoed back — the reason this is a defensive floor rather than the primary
    # control (PreProcessNode redacts the same shapes on the way in).
    ("mrn_like_id", re.compile(r"\bMRN[-:\s]?\d{6,10}\b", re.IGNORECASE)),
    ("national_id_like", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("email_address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("phone_like", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
]

_SANITISED_STUB = (
    "[ANSWER BLOCKED by the output gate — disallowed content was detected. "
    "Retry without credential-like or direct-identifier-like strings.]"
)

# CONTAINMENT INVENTORY — every state field that carries answer text or a
# payload, with the content-free value written over it when the gate blocks.
#
# Blocking is not a status flip. AgentBaseGraph.get_output() resolves the
# caller-facing value as ``formatted_output or result`` with NO status check, so
# a gate that returns ERROR while leaving these fields populated still ships the
# refused answer inside the error envelope. Both halves are needed and both are
# here: the outer get_output() withholds these on any non-success outcome, and
# this mapping empties them at the source so a single edit to either one cannot
# release clinical text on its own.
#
# ``answer`` is replaced with the (deliberately TRUTHY) stub rather than "" or
# None: a falsy replacement does not suppress the envelope's ``or result``
# fallback, it ACTIVATES it. The JSON payload fields carry no notice — they are
# emptied outright.
#
# Guarded by an inventory test that cross-checks these keys against the domain
# content keys ClinicalGuidelinesGraphNode.merge_output() writes, so a future
# domain field cannot quietly join the state without joining this mapping.
_CLEARED_ON_BLOCK: "dict[str, Any]" = {
    "answer": _SANITISED_STUB,
    "generated_answer": None,
    "citations": None,
    "filtered_passages": None,
}

# ── Caller-visible ERROR envelope — closed-set labels only ────────────────────
#
# On every non-success path the caller receives values this module chose: one
# constant reason code, nothing else. Nothing read from ``error_log``, from the
# gate's own violation message, or from any other node-authored string is
# projected. ``error_log`` is free text — the framework writes ``[Node]
# <message>`` plus a full traceback into it whenever a node raises, and a node's
# own entry can quote whatever it was handed (an identifier, a name, an e-mail,
# an upstream response body) — so truncating, path-masking or credential-only
# redaction of it is not a closed set; not publishing it is. ``error_log``
# itself is untouched: it stays the internal channel the state reducer appends
# to and the audit trail reads.
#
# The envelope always carries its constant key, so it is always truthy:
# AgentBaseGraph.get_output() selects ``formatted_output or result`` with no
# status check, and a falsy envelope would re-open that fallback.
_REASON_WORKFLOW_FAILED = "workflow_failed"  # the workflow reported status=error before the gate ran
_REASON_OUTPUT_WITHHELD = "output_withheld"  # this node's output gate refused the response
ERROR_REASONS = frozenset({_REASON_WORKFLOW_FAILED, _REASON_OUTPUT_WITHHELD})


def error_envelope(reason: str) -> dict[str, Any]:
    """The caller-visible ERROR envelope: a constant reason code and nothing else."""
    if reason not in ERROR_REASONS:
        raise ValueError("error envelope reason must be one of ERROR_REASONS")
    return {"reason": reason}


def _contain(reason: str, new_errors: Optional[List[str]] = None) -> dict[str, Any]:
    """Fail-closed ERROR result: the closed-set envelope, every answer field cleared.

    ``formatted_output`` becomes the (truthy) envelope, ``result`` is cleared —
    the base envelope falls back to it for ``output`` — and every field in
    _CLEARED_ON_BLOCK is overwritten content-free. ``new_errors`` (this node's
    own gate message, which names a violation TYPE and never a value) goes to
    ``error_log``, the internal channel, never into the envelope. Entries
    already in ``error_log`` are not re-emitted: the state reducer appends, so
    they would be duplicated.
    """
    contained: dict[str, Any] = {
        "formatted_output": error_envelope(reason),
        "result": None,
        # Contain, do not merely refuse — see _CLEARED_ON_BLOCK.
        **_CLEARED_ON_BLOCK,
        "status": AgentStatus.ERROR.value,
    }
    if new_errors:
        contained["error_log"] = list(new_errors)
    return contained


# The mandatory advisory-only disclaimer. Kept verbatim in step with
# GenerateAnswerNode.DISCLAIMER and OutputFormatNode.DISCLAIMER; this node is
# where its presence is ENFORCED.
DISCLAIMER = "This output is advisory only. Clinical judgment of the attending " "physician takes precedence."


def _attach_disclaimer(text: str) -> str:
    """Return *text* with the advisory disclaimer appended if it is not already there.

    Separated from the verification step below on purpose: attaching and
    verifying are two different responsibilities, and keeping them apart is what
    makes the fail-closed branch reachable and testable rather than a comment
    claiming it would work.
    """
    if DISCLAIMER in text:
        return text
    return f"{text}\n\n{DISCLAIMER}"


def _security_gate_output(content: Any) -> Optional[str]:
    """Scan caller-facing output for disallowed content, RECURSIVELY.

    ``content`` may be a string, or a dict/list/tuple nesting strings at any
    depth — every string leaf is scanned, so a violation buried inside a
    structured field is caught exactly like a top-level string and a future
    structured output shape needs no gate rewrite.

    Returns the name of the first violation found, or None when clean.
    """
    if isinstance(content, str):
        for name, pattern in _DISALLOWED_PATTERNS:
            if pattern.search(content):
                return name
        # The framework's own recognizer, consulted second so the domain
        # patterns keep naming the domain violation. Anything only IT knows
        # (Stripe sk_live_/sk_test_ keys, AWS AKIA access-key ids, database
        # connection strings, single-segment JWTs) is blocked HERE — where the
        # containment below applies — instead of reaching the framework's
        # output scan, which raises and discards that containment.
        findings = detect_credentials(content)
        if findings:
            # The credential TYPE, never the matched value: echoing the value
            # would put it back into this node's own result, where the
            # framework scan raises and throws the containment away with it.
            return f"framework_credential:{findings[0]['type']}"
        return None
    if isinstance(content, dict):
        for value in content.values():
            violation = _security_gate_output(value)
            if violation:
                return violation
        return None
    if isinstance(content, (list, tuple)):
        for item in content:
            violation = _security_gate_output(item)
            if violation:
                return violation
        return None
    # Non-string scalars (int/float/bool/None/...) carry no disallowed text.
    return None


# Reason code -> the sentence the caller reads. A code with no entry falls
# back to the generic one rather than leaking the code itself.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Apply the output gate and expose the final advisory answer.

    Outer backbone post_process slot. Declared ANONYMOUS — caller trust was
    already enforced at PreProcessNode (VERIFIED_EXTERNAL).

    Input state keys:
        answer: str   — formatted advisory answer from the inner OutputFormatNode
        result: str   — fallback source when answer is absent

    Output state keys (partial dict):
        formatted_output: str   — the gated answer on SUCCESS;
                                  the closed-set envelope {"reason": <code>} on ERROR
        result:           str   — the gated answer on SUCCESS; None on ERROR
        status:           str
        error_log:        list[str]  (only on a gate block — the violation type)
        answer / generated_answer / citations / filtered_passages
                          — overwritten content-free on every ERROR return
                            (see _CLEARED_ON_BLOCK)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def _block(self, state: AgentState, event: str, detail: dict[str, Any], message: str) -> dict[str, Any]:
        """Fail-closed block: audit it, ship the closed-set envelope, never the raw content."""
        logger.error("PostProcessNode: output blocked — %s", message)
        emit_trace_event(event, detail, state)
        # The message names the violation TYPE and never the matched value
        # (see _security_gate_output): a value echoed here would be scanned by
        # the framework on this very return, and a raise there discards
        # everything above. It goes to error_log only — the internal channel;
        # the caller receives the reason code.
        return _contain(_REASON_OUTPUT_WITHHELD, [f"PostProcessNode: {message}"])

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A run declined upstream has nothing to format. Render the reason as
        # the caller-facing body and carry the marker onward.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
                "result": message,
                "formatted_output": message,
            }
        # ── Already-errored state: contain, never fabricate a success ─────────
        # The backbone routes an errored run straight to finalize, so on the
        # compiled graph this node never sees one; a direct call still returns
        # the same closed-set envelope as every other error path instead of
        # gating an empty answer into a SUCCESS. The entries already in
        # error_log stay internal and are not re-emitted.
        if state.get("status") == AgentStatus.ERROR.value:
            return _contain(_REASON_WORKFLOW_FAILED)

        answer: str = state.get("answer") or state.get("result") or ""

        # ── Fallback for an empty answer ──────────────────────────────────────
        if not answer.strip():
            logger.warning("PostProcessNode: answer is empty — using fallback message")
            answer = (
                "No answer content was generated for this clinical question. " "Check error_log for upstream failures."
            )

        # ── Layer 1: disallowed-content scan (runs on the text as produced) ───
        violation = _security_gate_output(answer)
        if violation:
            return self._block(
                state,
                "post_process_output_blocked",
                {"violation": violation},
                f"a disallowed pattern was detected in the answer ({violation})",
            )

        # ── Layer 2: mandatory advisory disclaimer (attach, then VERIFY) ──────
        # Fail-closed by construction: if a future edit truncates or reshapes the
        # answer so the disclaimer is not present after attachment, the response
        # is blocked instead of shipping clinical text without it.
        output = answer
        if DISCLAIMER not in output:
            emit_trace_event("post_process_disclaimer_attached", {}, state)
            output = _attach_disclaimer(output)
        if DISCLAIMER not in output:
            return self._block(
                state,
                "post_process_disclaimer_missing",
                {},
                "the mandatory advisory disclaimer could not be verified in the answer",
            )

        logger.info("PostProcessNode: output gate passed — length=%d", len(output))
        emit_trace_event(
            "post_process_complete",
            {"output_length": len(output)},
            state,
        )

        return {
            "formatted_output": output,
            "result": output,
            "status": AgentStatus.SUCCESS.value,
        }
