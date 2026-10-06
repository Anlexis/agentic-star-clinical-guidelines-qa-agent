# HCR-C2-011 — Unit tests: the outer invoke() envelope
# (ClinicalGuidelinesQAAgent.get_output) and the output gate's containment.
#
# The envelope is the last thing between the graph state and the caller, and it
# is the second half of the output-gate contract. The framework base resolves
# its `output` key as `formatted_output or result` WITHOUT consulting status,
# and this template's override additionally surfaced `result` unconditionally —
# so an envelope that forwards state verbatim hands back the very clinical
# advisory the output gate refused, inside an error envelope.
#
# These call get_output() directly on a state dict so the resolution rule itself
# is pinned, independent of which node happened to produce that state; the gate
# half is pinned beside it, and the end-to-end consequence on the real /invoke
# surface is pinned in tests/proof_of_boundary/test_invoke_e2e.py. The
# closed-set property of the error envelope (nothing node-authored is ever
# projected) is held path-by-path in tests/unit/test_error_envelope_closed_set.py.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import ClinicalGuidelinesGraphNode, ClinicalGuidelinesQAAgent
from src.nodes.post_process_node import (
    _CLEARED_ON_BLOCK,
    _REASON_OUTPUT_WITHHELD,
    _REASON_WORKFLOW_FAILED,
    _SANITISED_STUB,
    PostProcessNode,
    error_envelope,
)

# What the inner workflow produced before the output gate ran.
_PRE_GATE_ANSWER = (
    "CLINICAL GUIDELINES Q&A — ADVISORY RESPONSE\n\n"
    "Based on 1 matching clinical guideline passage(s):\n"
    "  - [GL-HTN-001] Adult hypertension: lifestyle measures are first-line."
)
_PRE_GATE_CITATIONS = json.dumps([{"id": "GL-HTN-001", "title": "Adult hypertension"}])


def _state(**overrides) -> dict:
    state = {
        "status": AgentStatus.SUCCESS.value,
        "formatted_output": _PRE_GATE_ANSWER,
        "result": _PRE_GATE_ANSWER,
        "answer": _PRE_GATE_ANSWER,
        "generated_answer": "lifestyle measures are first-line.",
        "citations": _PRE_GATE_CITATIONS,
        "filtered_passages": json.dumps([{"id": "GL-HTN-001", "text": "lifestyle"}]),
        "error_log": [],
        "trace_id": "envelope-test",
        "correlation_id": "envelope-test",
        "node_history": [],
    }
    state.update(overrides)
    return state


def _envelope(**overrides) -> dict:
    return ClinicalGuidelinesQAAgent().get_output(_state(**overrides))


class TestSuccessEnvelope:
    """The control: on the gated success path the envelope really does carry the
    advisory answer and its grounding. Without it every containment assertion
    below would pass just as well on an envelope that returns nothing at all —
    and for a clinical template a refuse-everything gate is the worse outcome.
    """

    def test_success_surfaces_the_grounded_answer(self):
        envelope = _envelope()
        assert envelope["status"] == AgentStatus.SUCCESS.value
        assert envelope["output"] == _PRE_GATE_ANSWER
        assert envelope["result"] == _PRE_GATE_ANSWER
        assert envelope["answer"] == _PRE_GATE_ANSWER
        assert envelope["generated_answer"]
        assert json.loads(envelope["citations"])[0]["id"] == "GL-HTN-001"
        assert envelope["filtered_passages"]
        # The base envelope keys survive the extension.
        for key in ("trace_id", "correlation_id", "node_history"):
            assert key in envelope
        # A success carries no error surface at all.
        assert "error" not in envelope
        assert "error_log" not in envelope


class TestErrorEnvelopeContainment:
    def test_pre_gate_result_is_never_surfaced_on_an_error(self):
        """`result` is the PRE-gate value. On any non-success outcome it is
        withheld, together with every structured domain field."""
        envelope = _envelope(status=AgentStatus.ERROR.value)
        assert envelope["result"] is None
        for key in ("answer", "generated_answer", "citations", "filtered_passages"):
            assert envelope[key] is None

    def test_a_gate_blocked_state_releases_nothing(self):
        """The realistic shape: post_process blocked, so `formatted_output` is
        the gate's closed-set envelope and every other content slot was emptied
        at the gate. The caller receives the reason code and nothing else."""
        envelope = _envelope(
            status=AgentStatus.ERROR.value,
            formatted_output=error_envelope(_REASON_OUTPUT_WITHHELD),
            result=None,
            answer=_SANITISED_STUB,
            generated_answer=None,
            citations=None,
            filtered_passages=None,
            error_log=["PostProcessNode: a disallowed pattern was detected in the answer (mrn_like_id)"],
        )
        blob = json.dumps(envelope)
        assert "GL-HTN-001" not in blob
        assert "lifestyle" not in blob
        assert envelope["result"] is None
        assert envelope["output"] is None
        assert envelope["formatted_output"] is None
        assert envelope["error"] == {"reason": _REASON_OUTPUT_WITHHELD}
        assert "error_log" not in envelope
        assert "disallowed pattern" not in blob

    def test_the_or_result_fallback_is_dead_on_an_error(self):
        """The hole in the base envelope: with no formatted_output, `output`
        falls through to the pre-gate `result`. On a non-success outcome an
        absent gate output stays absent, never becomes the inner advisory."""
        envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=None)
        assert envelope["output"] is None
        assert "lifestyle" not in json.dumps(envelope)
        assert envelope["error"] == {"reason": _REASON_WORKFLOW_FAILED}

    def test_error_surfaces_the_closed_set_reason_and_nothing_else(self):
        """The reason code PostProcessNode chose is the caller's whole error
        surface; the gate's envelope itself is not echoed on any other key."""
        envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=error_envelope(_REASON_OUTPUT_WITHHELD))
        assert envelope["output"] is None
        assert envelope["formatted_output"] is None
        assert envelope["result"] is None
        assert envelope["error"] == error_envelope(_REASON_OUTPUT_WITHHELD)

    def test_structured_domain_fields_are_withheld_on_an_error(self):
        envelope = _envelope(status=AgentStatus.ERROR.value)
        for key in ("answer", "generated_answer", "citations", "filtered_passages"):
            assert envelope[key] is None

    def test_containment_holds_for_every_non_success_status(self):
        """Not an ERROR special case: any status that is not SUCCESS means the
        output gate did not pass the response, including the terminal statuses
        that route straight to finalize without post_process running at all."""
        for status in (
            AgentStatus.TIMEOUT.value,
            AgentStatus.CANCELLED.value,
            AgentStatus.RETRY.value,
            AgentStatus.PENDING.value,
            AgentStatus.AWAITING_HUMAN.value,
        ):
            envelope = _envelope(status=status, formatted_output=None)
            assert envelope["result"] is None, status
            assert envelope["output"] is None, status
            assert envelope["error"] == {"reason": _REASON_WORKFLOW_FAILED}, status
            assert "GL-HTN-001" not in json.dumps(envelope), status


class TestErrorSurfaceIsClosedSet:
    """The framework writes `[Node] <message>\\n<traceback>` into error_log on any
    node exception, and that reached /invoke verbatim with absolute deployment
    source paths in it. A summarised form still carried whatever the failing
    node had quoted — so error_log is not projected in any form."""

    _RAW_ENTRY = (
        "[RetrieveNode] output gate: credential pattern 'aws_key' detected in "
        "result['retrieved_passages']. Output blocked.\n"
        "Traceback (most recent call last):\n"
        '  File "/srv/app/framework/nodes/base_node.py", line 197, in __call__\n'
        "    result = self._security_gate_output(result)\n"
        "RuntimeError: output gate refused the result\n"
    )

    def test_no_entry_is_projected_in_any_form(self):
        envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=None, error_log=[self._RAW_ENTRY])
        blob = json.dumps(envelope)
        assert "error_log" not in envelope
        assert "aws_key" not in blob
        assert "Traceback" not in blob
        assert ".py" not in blob
        assert "/srv/app" not in blob
        assert "output gate" not in blob

    def test_the_caller_still_learns_the_request_failed(self):
        envelope = _envelope(status=AgentStatus.ERROR.value, formatted_output=None, error_log=[self._RAW_ENTRY])
        assert envelope["status"] == AgentStatus.ERROR.value
        assert envelope["error"] == {"reason": _REASON_WORKFLOW_FAILED}

    def test_an_empty_error_log_makes_no_difference_to_the_surface(self):
        with_entries = _envelope(status=AgentStatus.ERROR.value, formatted_output=None, error_log=[self._RAW_ENTRY])
        without = _envelope(status=AgentStatus.ERROR.value, formatted_output=None, error_log=[])
        assert with_entries == without


class TestOutputGateContainment:
    """Blocking a response must also CONTAIN it — at the source, not only at the
    envelope. The two halves are deliberately both present: neither the gate's
    clearing nor the envelope's withholding should be the single thing standing
    between a refused clinical advisory and the caller.
    """

    @pytest.fixture(autouse=True)
    def patch_emit(self, monkeypatch):
        monkeypatch.setattr("src.nodes.post_process_node.emit_trace_event", lambda *a, **k: None)

    def _blocked(self) -> dict:
        node = PostProcessNode()
        return node(
            {
                "answer": f"Advisory text. Reference MRN-1234567.\n\n{_PRE_GATE_ANSWER}",
                "generated_answer": "lifestyle measures are first-line.",
                "citations": _PRE_GATE_CITATIONS,
                "filtered_passages": json.dumps([{"id": "GL-HTN-001", "text": "lifestyle"}]),
                "node_history": [],
                "error_log": [],
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )

    def test_block_clears_every_output_bearing_field(self):
        result = self._blocked()
        assert result["status"] == AgentStatus.ERROR.value
        assert result["answer"] == _SANITISED_STUB
        for field in ("generated_answer", "citations", "filtered_passages"):
            assert result[field] is None, f"{field} still carries the pre-gate value"
        assert result["result"] is None

    def test_the_replacement_defeats_the_envelope_fallback(self):
        """`formatted_output or result` must resolve to the gate's own envelope.

        An empty replacement ("" or {}) is falsy and hands the resolution
        straight back to `result` — the exact hole the clearing closes. The
        closed-set envelope always carries its constant key, so it is truthy.
        """
        result = self._blocked()
        resolved = result["formatted_output"] or result["result"]
        assert resolved == error_envelope(_REASON_OUTPUT_WITHHELD)
        assert resolved

    def test_block_releases_no_clinical_content(self):
        blob = json.dumps(self._blocked())
        assert "MRN-1234567" not in blob
        assert "GL-HTN-001" not in blob
        assert "lifestyle" not in blob

    def test_clearing_survives_the_framework_output_scan(self):
        """The violation names the violation TYPE only, never the matched value.

        Echoing it would put the refused string back into this node's own
        result, where the framework's output-side credential scan raises — and a
        raise DISCARDS the whole return, so the clearing would never be applied
        and the pre-gate advisory would stay in state. Driven through
        node(state) so that scan really runs.
        """
        secret = "sk-" + "A" * 24
        result = PostProcessNode()(
            {
                "answer": f"Advisory text with {secret} in it.",
                "generated_answer": "lifestyle measures are first-line.",
                "node_history": [],
                "error_log": [],
                "caller_trust_level": TrustLevel.ANONYMOUS.value,
            }
        )
        blob = json.dumps(result)
        assert result["status"] == AgentStatus.ERROR.value
        assert secret not in blob
        assert "Traceback" not in blob, "the framework scan discarded the clearing return"
        assert result["result"] is None
        assert result["formatted_output"] == error_envelope(_REASON_OUTPUT_WITHHELD)
        assert result["generated_answer"] is None

    def test_cleared_inventory_covers_every_domain_content_field(self):
        """Inventory guard: the fields the gate empties must stay in step with
        the domain content fields merge_output() writes into outer state.

        A future domain field added to merge_output() would otherwise join the
        outer state carrying answer text and quietly survive a block — the
        original defect, one field at a time.
        """
        merged = ClinicalGuidelinesGraphNode().merge_output(
            {}, {"answer": "a", "generated_answer": "b", "citations": "c", "filtered_passages": "d", "status": "s"}
        )
        # error_code is the reason marker, not domain content: it must survive a
        # block, or the caller is left with an empty body and no reason.
        content_fields = set(merged) - {"status", "error_code"}
        assert content_fields == set(_CLEARED_ON_BLOCK), (
            "merge_output() and _CLEARED_ON_BLOCK have drifted: " f"{content_fields ^ set(_CLEARED_ON_BLOCK)}"
        )

    def test_every_cleared_value_is_content_free(self):
        """Inert-by-inspection: nothing in the replacement carries domain text."""
        for field, value in _CLEARED_ON_BLOCK.items():
            assert value is None or value == _SANITISED_STUB, field


class TestDetectorParity:
    """The gate's recogniser must not be narrower than the framework's.

    The framework scans every node result with detect_credentials() and RAISES
    on a hit — and BaseNode.__call__ replaces a raising node's return with a
    bare ERROR partial, discarding the gate's clearing along with it. So a shape
    the framework catches and this gate misses is not merely a narrower gate: it
    is a containment BYPASS that swaps the contained stub for an uncontained
    error. Measured on the shipped pattern set: sk_live_ keys, AWS AKIA ids and
    database connection strings were all missed.
    """

    @pytest.mark.parametrize(
        "shape",
        [
            "sk_live_" + "a" * 20,  # Stripe secret key — underscore, not hyphen
            "sk-" + "b" * 24,  # generic/OpenAI API key
            "eyJ" + "c" * 20,  # JWT (single segment — no dots)
            "AKIA" + "D" * 16,  # AWS access key id
            "Bearer " + "e" * 24,  # bearer token
            "postgresql://" + "user:pw@host:5432/db",  # connection string
        ],
    )
    def test_gate_blocks_every_shape_the_framework_would_refuse(self, shape):
        from framework.security.credential_detector import detect_credentials

        from src.nodes.post_process_node import _security_gate_output

        # Keeps the parametrization honest: each probe really is one the
        # framework refuses, so a miss below is a genuine parity gap.
        assert detect_credentials(shape), "probe shape is not a credential the framework refuses"
        violation = _security_gate_output(f"Advisory text mentioning {shape} inline.")
        assert violation is not None, f"gate missed a shape the framework refuses: {shape[:12]}..."
        assert shape not in violation, "the violation must name the type, never the value"

    @pytest.mark.parametrize(
        "clinical",
        [
            "For newly diagnosed adult hypertension, lifestyle measures are first-line.",
            "See [GL-SEP-002] for the sepsis resuscitation bundle.",
            "Reassess within 90d; give 5000 units subcutaneously.",
        ],
    )
    def test_ordinary_clinical_text_is_not_flagged(self, clinical):
        """The other direction — a refuse-everything gate would be worse than
        the leak it closes."""
        from src.nodes.post_process_node import _security_gate_output

        assert _security_gate_output(clinical) is None
