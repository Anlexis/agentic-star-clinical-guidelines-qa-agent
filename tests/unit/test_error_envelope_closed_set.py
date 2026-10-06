# HCR-C2-011 — The caller-visible ERROR envelope carries closed-set labels only.
#
# On any non-success path the caller must receive values this template chose
# from a closed set — a constant reason code — and nothing read from error_log,
# from the output gate's violation message, or from any other node-authored
# string. error_log is free text: the framework writes "[Node] <message>" plus a
# full traceback into it whenever a node raises, and a node's own entry can
# quote whatever the failing node was handed (an identifier, a name, an e-mail,
# an upstream response body). Truncating, path-masking or credential-only
# redaction of such text is not a closed-set contract; not publishing it is.
#
# Two surfaces are held, each parameterised over every non-success path it has:
#   - PostProcessNode's `formatted_output` (the field the framework envelope
#     selects as `output`), driven through the framework pipeline (node(state))
#     and directly (execute());
#   - ClinicalGuidelinesQAAgent.get_output() — the invoke envelope every
#     deployment shape returns — including states a compiled run cannot be
#     coaxed into.
# The end-to-end consequence on the real ASGI /invoke surface is held in
# tests/proof_of_boundary/test_pb_error_envelope_closed_set.py.
#
# The sentinel is deliberately NOT credential-shaped and carries no trace
# fragment: a redaction-based surface passes it straight through, which is the
# defect these tests must fail on — not one a redaction happened to catch.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import ClinicalGuidelinesQAAgent
from src.nodes.post_process_node import (
    _CLEARED_ON_BLOCK,
    _REASON_OUTPUT_WITHHELD,
    _REASON_WORKFLOW_FAILED,
    DISCLAIMER,
    ERROR_REASONS,
    PostProcessNode,
    error_envelope,
)

# Assembled at runtime (never a committed literal): a name, an e-mail and a
# token-shaped fragment — what an upstream response body can carry.
_FRAGMENTS = ("A. Tanaka", "a.tanaka@example.com", "sk-" + "live-xxx")
_SENTINEL = (
    "boom: upstream said {'patient':'"
    + _FRAGMENTS[0]
    + "','email':'"
    + _FRAGMENTS[1]
    + "','token':'"
    + _FRAGMENTS[2]
    + "'}"
)
_MARKERS = (_SENTINEL, "upstream said", *_FRAGMENTS)
# A framework-authored entry (the trust gate's own wording) — internal too.
_FRAMEWORK_ENTRY = "[PreProcessNode] S-1 trust gate denied: required=verified_external, caller=anonymous"
# A framework traceback entry, as BaseNode.__call__ writes it when a node raises.
_TRACEBACK_ENTRY = (
    "[RetrieveNode] knowledge base unavailable\n"
    "Traceback (most recent call last):\n"
    '  File "/srv/app/src/nodes/retrieve_node.py", line 1, in execute\n'
)
_INTERNAL_TEXT = ("upstream said", "trust gate denied", "knowledge base unavailable", "Traceback", "/srv/app", ".py")

# What the inner workflow produced before the output gate ran.
_PRE_GATE_ANSWER = (
    "CLINICAL GUIDELINES Q&A — ADVISORY RESPONSE\n\n"
    "Based on 1 matching clinical guideline passage(s):\n"
    "  - [GL-HTN-001] Adult hypertension: lifestyle measures are first-line."
)
_PRE_GATE_CITATIONS = json.dumps([{"id": "GL-HTN-001", "title": "Adult hypertension"}])
_PRE_GATE_PASSAGES = json.dumps([{"id": "GL-HTN-001", "text": "lifestyle"}])


@pytest.fixture(autouse=True)
def _mute_audit(monkeypatch):
    monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)


def _strings(value):
    """Every string reachable in value: dict keys and values, list/tuple items,
    and the repr of anything else that is not a plain scalar."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _strings(key)
            yield from _strings(child)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for child in value:
            yield from _strings(child)
    elif value is not None and not isinstance(value, (bool, int, float)):
        yield repr(value)


def _found(mapping, *needles) -> list:
    """The needles reachable anywhere inside mapping (nested keys and values)."""
    texts = list(_strings(mapping))
    return [needle for needle in needles if any(needle in text for text in texts)]


def _assert_nothing_internal(mapping) -> None:
    assert _found(mapping, *_MARKERS) == [], mapping
    assert _found(mapping, *_INTERNAL_TEXT) == [], mapping
    rendered = json.dumps(mapping, default=str)
    for marker in (*_MARKERS, *_INTERNAL_TEXT):
        assert marker not in rendered


# ── PostProcessNode: formatted_output on every non-success path ───────────────


def _post_state(**overrides) -> dict:
    state = {
        "status": AgentStatus.SUCCESS.value,
        "answer": f"{_PRE_GATE_ANSWER}\n\n{DISCLAIMER}",
        "result": _PRE_GATE_ANSWER,
        "generated_answer": "lifestyle measures are first-line.",
        "citations": _PRE_GATE_CITATIONS,
        "filtered_passages": _PRE_GATE_PASSAGES,
        # Every internal channel is seeded: the log carries the sentinel, a
        # framework entry and a traceback entry.
        "error_log": [_SENTINEL, _FRAMEWORK_ENTRY, _TRACEBACK_ENTRY],
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "correlation_id": "closed-set-test",
        "node_history": [],
        "execution_time": {},
    }
    state.update(overrides)
    return state


def _drive(state: dict, entry: str, monkeypatch, disclaimer_broken: bool) -> dict:
    if disclaimer_broken:
        # A future edit that breaks attachment: the body comes back unchanged,
        # so the verify step refuses the response.
        monkeypatch.setattr("src.nodes.post_process_node._attach_disclaimer", lambda text: text)
    node = PostProcessNode()
    return node.execute(state) if entry == "execute" else node(state)


# (state overrides, expected reason, entry point, break the disclaimer attach).
# The already-errored state is reached only by a direct execute(): the backbone
# routes an errored run straight to finalize, and the framework pipeline
# short-circuits before execute() on an errored incoming state (returning the
# state itself — a graph-internal partial update, not the caller's envelope).
# That branch must contain rather than gate an empty answer into a SUCCESS.
# Every gate arm is driven through the framework pipeline and directly.
_POST_PROCESS_ERROR_PATHS = [
    pytest.param(
        {"status": AgentStatus.ERROR.value}, _REASON_WORKFLOW_FAILED, "execute", False, id="errored-state/execute"
    ),
    pytest.param(
        {"status": AgentStatus.ERROR.value, "answer": None},
        _REASON_WORKFLOW_FAILED,
        "execute",
        False,
        id="errored-state-answer-only-in-result/execute",
    ),
    pytest.param(
        {"answer": f"Reference MRN 1234567 for details.\n\n{DISCLAIMER}"},
        _REASON_OUTPUT_WITHHELD,
        "call",
        False,
        id="gate-direct-identifier/call",
    ),
    pytest.param(
        {"answer": f"Reference MRN 1234567 for details.\n\n{DISCLAIMER}"},
        _REASON_OUTPUT_WITHHELD,
        "execute",
        False,
        id="gate-direct-identifier/execute",
    ),
    pytest.param(
        {"answer": "Use " + "sk_live_" + "a" * 20 + " to fetch it."},
        _REASON_OUTPUT_WITHHELD,
        "call",
        False,
        id="gate-framework-credential/call",
    ),
    pytest.param(
        {"answer": "Contact dr.smith@example.org for the record."},
        _REASON_OUTPUT_WITHHELD,
        "execute",
        False,
        id="gate-email/execute",
    ),
    pytest.param(
        {"answer": "An answer body with no disclaimer."},
        _REASON_OUTPUT_WITHHELD,
        "execute",
        True,
        id="gate-disclaimer-unverifiable/execute",
    ),
]
_GATE_PATHS = [p for p in _POST_PROCESS_ERROR_PATHS if p.values[1] == _REASON_OUTPUT_WITHHELD]
_ERRORED_PATHS = [p for p in _POST_PROCESS_ERROR_PATHS if p.values[1] == _REASON_WORKFLOW_FAILED]


class TestPostProcessErrorEnvelope:
    @pytest.mark.parametrize("overrides, reason, entry, broken", _POST_PROCESS_ERROR_PATHS)
    def test_envelope_values_are_drawn_from_the_declared_constants(self, overrides, reason, entry, broken, monkeypatch):
        result = _drive(_post_state(**overrides), entry, monkeypatch, broken)
        assert result["status"] == AgentStatus.ERROR.value
        envelope = result["formatted_output"]
        assert set(envelope) == {"reason"}, envelope
        assert envelope["reason"] in ERROR_REASONS
        assert envelope == error_envelope(reason)
        assert result["result"] is None

    @pytest.mark.parametrize("overrides, reason, entry, broken", _POST_PROCESS_ERROR_PATHS)
    def test_envelope_is_truthy_so_the_result_fallback_stays_closed(
        self, overrides, reason, entry, broken, monkeypatch
    ):
        # AgentBaseGraph.get_output() selects `formatted_output or result`; a
        # falsy envelope would re-open the fallback onto `result`.
        result = _drive(_post_state(**overrides), entry, monkeypatch, broken)
        assert result["formatted_output"]
        assert (result["formatted_output"] or result["result"]) == error_envelope(reason)

    @pytest.mark.parametrize("overrides, reason, entry, broken", _POST_PROCESS_ERROR_PATHS)
    def test_every_answer_bearing_field_is_cleared(self, overrides, reason, entry, broken, monkeypatch):
        result = _drive(_post_state(**overrides), entry, monkeypatch, broken)
        for field, cleared in _CLEARED_ON_BLOCK.items():
            assert result[field] == cleared, field
        rendered = json.dumps(result, default=str)
        assert "GL-HTN-001" not in rendered
        assert "lifestyle" not in rendered

    @pytest.mark.parametrize("overrides, reason, entry, broken", _POST_PROCESS_ERROR_PATHS)
    def test_seeded_error_text_appears_nowhere_in_the_returned_mapping(
        self, overrides, reason, entry, broken, monkeypatch
    ):
        result = _drive(_post_state(**overrides), entry, monkeypatch, broken)
        _assert_nothing_internal(result)

    @pytest.mark.parametrize("overrides, reason, entry, broken", _GATE_PATHS)
    def test_gate_messages_stay_in_the_internal_channel(self, overrides, reason, entry, broken, monkeypatch):
        result = _drive(_post_state(**overrides), entry, monkeypatch, broken)
        # The audit trail keeps the violation (its TYPE, never a value); the
        # caller-visible field does not carry it.
        assert any(line.startswith("PostProcessNode: ") for line in result["error_log"])
        assert not any("PostProcessNode" in text for text in _strings(result["formatted_output"]))
        assert "MRN 1234567" not in json.dumps(result, default=str)

    @pytest.mark.parametrize("overrides, reason, entry, broken", _ERRORED_PATHS)
    def test_errored_state_entries_are_not_re_emitted(self, overrides, reason, entry, broken, monkeypatch):
        # The state reducer appends error_log; re-emitting the incoming entries
        # would duplicate every line — and the caller never sees them anyway.
        result = _drive(_post_state(**overrides), entry, monkeypatch, broken)
        assert "error_log" not in result

    def test_success_path_is_unchanged(self):
        result = PostProcessNode()(_post_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == result["result"]
        assert "GL-HTN-001" in result["formatted_output"]
        assert "reason" not in result["formatted_output"]


# ── The invoke envelope: get_output() on every non-success state ──────────────


def _invoke_state(**overrides) -> dict:
    state = {
        "status": AgentStatus.ERROR.value,
        "formatted_output": None,
        "result": _PRE_GATE_ANSWER,
        "answer": _PRE_GATE_ANSWER,
        "generated_answer": "lifestyle measures are first-line.",
        "citations": _PRE_GATE_CITATIONS,
        "filtered_passages": _PRE_GATE_PASSAGES,
        "error_log": [_SENTINEL, _FRAMEWORK_ENTRY, _TRACEBACK_ENTRY],
        "trace_id": "tr",
        "correlation_id": "co",
        "node_history": ["InitializeNode", "FinalizeNode"],
    }
    state.update(overrides)
    return state


_INVOKE_ERROR_PATHS = [
    pytest.param({}, _REASON_WORKFLOW_FAILED, id="inner-error-routed-to-finalize"),
    pytest.param({"status": AgentStatus.TIMEOUT.value}, _REASON_WORKFLOW_FAILED, id="timeout-status"),
    pytest.param({"status": AgentStatus.CANCELLED.value}, _REASON_WORKFLOW_FAILED, id="cancelled-status"),
    pytest.param({"status": AgentStatus.RETRY.value}, _REASON_WORKFLOW_FAILED, id="retry-status"),
    pytest.param({"status": AgentStatus.PENDING.value}, _REASON_WORKFLOW_FAILED, id="pending-status"),
    pytest.param({"status": AgentStatus.AWAITING_HUMAN.value}, _REASON_WORKFLOW_FAILED, id="awaiting-human-status"),
    pytest.param(
        {"formatted_output": error_envelope(_REASON_OUTPUT_WITHHELD), "result": None, **_CLEARED_ON_BLOCK},
        _REASON_OUTPUT_WITHHELD,
        id="post-process-gate-block",
    ),
    pytest.param({"formatted_output": _SENTINEL}, _REASON_WORKFLOW_FAILED, id="non-dict-formatted-output"),
    pytest.param(
        {"formatted_output": {"reason": _SENTINEL}}, _REASON_WORKFLOW_FAILED, id="reason-outside-the-closed-set"
    ),
    pytest.param(
        {"formatted_output": {"reason": ["not", "a", "str"]}}, _REASON_WORKFLOW_FAILED, id="reason-not-a-string"
    ),
    pytest.param({"formatted_output": {}}, _REASON_WORKFLOW_FAILED, id="empty-dict-formatted-output"),
]
_WITHHELD_KEYS = ("result", "answer", "generated_answer", "citations", "filtered_passages")


class TestInvokeErrorEnvelope:
    @pytest.mark.parametrize("overrides, reason", _INVOKE_ERROR_PATHS)
    def test_error_values_are_drawn_from_the_declared_constants(self, overrides, reason):
        out = ClinicalGuidelinesQAAgent().get_output(_invoke_state(**overrides))
        assert out["status"] != AgentStatus.SUCCESS.value
        assert set(out["error"]) == {"reason"}
        assert out["error"]["reason"] in ERROR_REASONS
        assert out["error"] == error_envelope(reason)

    @pytest.mark.parametrize("overrides, reason", _INVOKE_ERROR_PATHS)
    def test_output_is_null_and_every_answer_slot_is_withheld(self, overrides, reason):
        out = ClinicalGuidelinesQAAgent().get_output(_invoke_state(**overrides))
        assert out["output"] is None
        assert out["formatted_output"] is None
        for key in _WITHHELD_KEYS:
            assert out[key] is None, key
        # The base envelope keys survive the override.
        for key in ("status", "trace_id", "correlation_id", "node_history"):
            assert key in out

    @pytest.mark.parametrize("overrides, reason", _INVOKE_ERROR_PATHS)
    def test_error_log_is_not_projected_and_the_sentinel_appears_nowhere(self, overrides, reason):
        out = ClinicalGuidelinesQAAgent().get_output(_invoke_state(**overrides))
        assert "error_log" not in out
        _assert_nothing_internal(out)
        rendered = json.dumps(out, default=str)
        assert "GL-HTN-001" not in rendered
        assert "lifestyle" not in rendered

    def test_success_envelope_is_unchanged(self):
        gated = f"{_PRE_GATE_ANSWER}\n\n{DISCLAIMER}"
        out = ClinicalGuidelinesQAAgent().get_output(
            _invoke_state(status=AgentStatus.SUCCESS.value, formatted_output=gated, result=gated, error_log=[])
        )
        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["output"] == gated
        assert out["answer"] == gated
        assert json.loads(out["citations"])[0]["id"] == "GL-HTN-001"
        assert "error" not in out
        assert "error_log" not in out


# ── The envelope builder itself ───────────────────────────────────────────────


class TestErrorEnvelopeBuilder:
    def test_every_declared_reason_builds_a_truthy_single_key_envelope(self):
        for reason in ERROR_REASONS:
            envelope = error_envelope(reason)
            assert envelope
            assert envelope == {"reason": reason}

    def test_a_reason_outside_the_closed_set_is_refused_and_not_echoed(self):
        with pytest.raises(ValueError) as info:
            error_envelope(_SENTINEL)
        assert _SENTINEL not in str(info.value)


# ── Verify the verifier ───────────────────────────────────────────────────────


class TestTheProbeFindsTheSentinelWhereItLives:
    def test_walk_finds_the_sentinel_in_a_seeded_mapping(self):
        # The same walk DOES find every marker in the state it was seeded into,
        # so the "nowhere" assertions above are not vacuous.
        assert set(_found(_post_state(), *_MARKERS)) == set(_MARKERS)
        assert set(_found(_invoke_state(), *_INTERNAL_TEXT)) == set(_INTERNAL_TEXT)
        assert _found({"nested": [{"deep": {"key " + _SENTINEL: 1}}]}, _SENTINEL) == [_SENTINEL]
