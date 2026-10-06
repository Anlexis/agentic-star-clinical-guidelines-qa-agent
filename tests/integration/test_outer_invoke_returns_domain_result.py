# HCR-C2-011 — Integration: the compiled OUTER graph's success path must
# return the domain advisory answer to the caller.
#
# AgentBaseGraph.get_output() surfaces only the {output, status, ...} envelope,
# so without the override in ClinicalGuidelinesQAAgent a successful
# agent.invoke() would drop the domain result (formatted_output / result /
# answer / generated_answer / citations / filtered_passages) even though
# PostProcessNode and the inner DomainWorkflowGraph populate them on the
# internal state. A test that only inspects ["output"] would never notice.
#
# These tests invoke the COMPILED OUTER graph (Graph().compile().invoke(...)) —
# the same construction the PB-6 backbone test uses — with a valid
# VERIFIED_EXTERNAL clinical question, and assert the domain result is
# surfaced. A second case proves the output gate is NOT weakened: a credential
# echoed into the (grounded) answer blocks the invoke and the structured fields
# are withheld.

import json

from framework.schemas.agent_status import AgentStatus


# A clinical question whose non-stopword content terms (adult / hypertension /
# blood / pressure / management / lifestyle / modification / pharmacotherapy)
# all match the synthetic GL-HTN-001 guideline passage (lexical score 1.0),
# well above the 0.75 score_threshold, so RerankFilterNode keeps it and
# GenerateAnswerNode returns a grounded (non-abstention) advisory answer.
_VALID_QUESTION = "adult hypertension blood pressure management lifestyle modification " "pharmacotherapy"

# The same grounded question with a credential token appended. It stays a
# single lexical token (score 8/9 ≈ 0.89 >= 0.75, still grounded) and is echoed
# into the composed answer ("Question: ..."), so the output gate
# (_security_gate_output) must block it: status -> ERROR, sanitised stub.
_CREDENTIAL_SECRET = "sk-ABCDEFGHIJKLMNOP1234"
_CREDENTIAL_QUESTION = f"{_VALID_QUESTION} {_CREDENTIAL_SECRET}"


def _patch_domain_emit(monkeypatch):
    """Patch emit_trace_event in every node module (avoids audit-backend calls)."""
    for mod_suffix in (
        "pre_process_node",
        "input_validate_node",
        "retrieve_node",
        "rerank_filter_node",
        "generate_answer_node",
        "output_format_node",
        "post_process_node",
    ):
        try:
            monkeypatch.setattr(
                f"src.nodes.{mod_suffix}.emit_trace_event",
                lambda *a, **k: None,
            )
        except AttributeError:
            pass  # module not imported / no emit symbol; fine


def _invoke(monkeypatch, question):
    """Compile and invoke the OUTER graph as a real VERIFIED_EXTERNAL caller."""
    _patch_domain_emit(monkeypatch)
    from framework.schemas.invocation_context import InvocationContext, TrustLevel
    from src.graph.graph import Graph

    from src.graph.graph import load_runtime_config

    agent = Graph(config=load_runtime_config())
    agent.compile()
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
    return agent.invoke(question, ctx=ctx)


class TestOuterInvokeReturnsDomainResult:
    """A successful outer invoke must surface the domain advisory answer."""

    def test_success_invoke_surfaces_domain_result(self, monkeypatch):
        result = _invoke(monkeypatch, _VALID_QUESTION)

        assert result.get("status") == AgentStatus.SUCCESS.value, (
            f"expected SUCCESS, got {result.get('status')!r}; " f"error_log={result.get('error_log')}"
        )

        # The regression: these were ALL None before the get_output() override.
        formatted_output = result.get("formatted_output")
        answer = result.get("answer")
        assert formatted_output is not None, "formatted_output must be surfaced on a successful invoke"
        assert answer is not None, "answer must be surfaced on a successful invoke"
        assert result.get("result") is not None, "result must be surfaced on a successful invoke"

        # The surfaced answer must actually contain the grounded advisory result.
        for needle in ("CLINICAL GUIDELINES Q&A", "GL-HTN-001", "advisory only"):
            assert needle.lower() in formatted_output.lower(), f"{needle!r} missing from formatted_output"
            assert needle.lower() in answer.lower(), f"{needle!r} missing from answer"

        # Structured domain result is surfaced and carries the grounded citation.
        generated_answer = result.get("generated_answer")
        citations = result.get("citations")
        assert generated_answer is not None, "generated_answer must be surfaced on a successful invoke"
        assert citations is not None, "citations must be surfaced on a successful invoke"
        assert "advisory only" in generated_answer.lower()
        parsed_citations = json.loads(citations)
        assert any(c.get("id") == "GL-HTN-001" for c in parsed_citations)

        # The framework envelope is preserved (backward compatible).
        assert result.get("output") is not None

    def test_credential_block_withholds_structured_fields(self, monkeypatch):
        """The output gate is not weakened: a credential echoed in the grounded
        answer blocks the invoke and the structured domain fields are withheld
        (fail-closed)."""
        result = _invoke(monkeypatch, _CREDENTIAL_QUESTION)

        assert (
            result.get("status") == AgentStatus.ERROR.value
        ), f"expected the output gate to block, got status={result.get('status')!r}"
        # The credential-bearing structured result is NOT surfaced.
        assert result.get("answer") is None
        assert result.get("generated_answer") is None
        assert result.get("citations") is None
        assert result.get("filtered_passages") is None
        # The caller-facing output carries only the sanitised stub — never the raw secret.
        assert _CREDENTIAL_SECRET not in (result.get("formatted_output") or "")
        assert _CREDENTIAL_SECRET not in (result.get("result") or "")
        assert _CREDENTIAL_SECRET not in (result.get("output") or "")
