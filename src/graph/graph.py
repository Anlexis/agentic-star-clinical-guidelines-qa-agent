"""AgentCore Platform v1.0"""

# HCR-C2-011 — Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed — do NOT override add_edges()):
#     START → initialize → pre_process → main → {route} → post_process → finalize → END
#                                             ↓ (RETRY, max_retry)
#                                          pre_process
#
#   The `main` slot is a GraphNode subclass (ClinicalGuidelinesGraphNode) that
#   delegates the full retrieval-and-answer workflow to DomainWorkflowGraph
#   (the inner BaseGraph).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 ← outer graph (this file)
#   src/graph/domain_workflow_graph.py ← inner graph (multi-step retrieval topology)
#   src/graph/context_bridge.py        ← caller input_context outer→inner hand-off
#
# Rules enforced:
#   ✅ ClinicalGuidelinesQAAgent inherits AgentBaseGraph (the framework base class)
#   ✅ super().register_nodes() called first (fills initialize + finalize)
#   ✅ ClinicalGuidelinesGraphNode assigned to self._nodes["main"]
#   ✅ PreProcessNode (VERIFIED_EXTERNAL) in the pre_process slot (input trust gate)
#   ✅ PostProcessNode in the post_process slot (output gate)
#   ✅ merge_output() returns only changed keys
#   ✅ get_output() surfaces the domain advisory answer
#   ✅ class name matches config/agent.yaml `class:` exactly
#   ❌ add_edges() NOT overridden on the outer graph
#   ❌ No platform-internal SDK imports

import os
from typing import Any, ClassVar

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import ERROR_REASONS, PostProcessNode, _REASON_WORKFLOW_FAILED, error_envelope
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

# Runtime parameters live in config/config.yaml (the repo root is three levels
# up from this file: src/graph/graph.py → src/graph → src → <repo root>).
# config/agent.yaml is the STATIC manifest — a flat document with no `agent:`
# block and no runtime values in it — so runtime settings are never read from
# there.
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "config.yaml",
)


def load_runtime_config() -> dict[str, Any]:
    """Return the runtime parameters declared in config/config.yaml.

    Shape (see that file): ``max_retry`` and ``timeout_s`` at the top level,
    plus the ``retrieval`` / ``llm`` / ``security`` blocks. The top-level keys
    are the ones AgentBaseGraph itself validates and consumes, so this dict is
    passed straight into the graph constructor by src/api/server.py.

    Best-effort: a missing or unparseable file yields ``{}`` so graph
    construction never breaks (every consumer falls back to its declared
    default). PyYAML is loaded lazily — it is a framework runtime dependency,
    so importing it on demand avoids a hard module-load coupling.
    """
    try:
        import yaml

        with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        return config if isinstance(config, dict) else {}
    except Exception:
        return {}


# ── Error surface ─────────────────────────────────────────────────────────────
#
# error_log is node-authored text and stays INTERNAL. There are two ways a raw
# framework traceback arrives in it:
#
#   * an OUTER node raises — BaseNode.__call__ catches it and writes
#     "[Node] <message>\n<traceback.format_exc()>" straight into outer state;
#   * an INNER node raises — the same string travels the inner error_log into
#     SubgraphError and out through on_subgraph_error() below.
#
# Both were measured reaching /invoke verbatim, carrying absolute source paths
# of the deployment filesystem — and a summarised form still carried whatever
# the failing node had quoted. get_output() below therefore projects none of
# it: the caller-visible error is the closed-set envelope PostProcessNode
# declares (error_envelope / ERROR_REASONS), and error_log remains the channel
# the state reducer appends to and the audit trail reads.


class ClinicalGuidelinesGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of ClinicalGuidelinesQAAgent.

    Wraps DomainWorkflowGraph (the inner retrieval-and-answer BaseGraph).
    Called by the AgentBaseGraph backbone after pre_process and before
    post_process.

    Contracts:
      get_subgraph()  — instantiate and return DomainWorkflowGraph
      extract_input() — pull validated_input from outer state; bridge input_context
      merge_output()  — map sub_result fields into the outer state delta (changed keys only)
      on_subgraph_error() — turn an inner failure into a clean outer error

    error_strategy is "capture" rather than "propagate". Both are fail-closed —
    the backbone routes an error status straight to finalize, so post_process
    never runs and no answer is emitted either way. The difference is what the
    caller receives: "propagate" re-raises, and the framework's node wrapper
    turns the exception into an error_log entry containing a full traceback with
    internal file paths, while the domain message that names the offending field
    is buried inside it. Capturing lets on_subgraph_error() below return the
    field-naming message on its own.
    """

    error_strategy: ClassVar[str] = "capture"
    propagate_hitl: ClassVar[bool] = False

    def get_subgraph(self) -> Any:
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to act on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates, screens and normalises the raw user_input and
        writes the result to validated_input. Prefer that; fall back to
        user_input if validated_input is absent (e.g. in unit tests).

        This is also where the caller's input_context is bridged to the inner
        graph: GraphNode.execute() does not forward input_context on
        subgraph.invoke(), and extract_input is the last hook of ours that sees
        the outer state before the inner invoke — see src/graph/context_bridge.py.
        """
        set_caller_input_context(state.get("input_context") or {})
        return str(state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map the inner graph's sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys — never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  → "answer", "generated_answer", "citations",
                                       "filtered_passages", "status"
          This merge_output() reads → sub_result.get(...) for each of these keys.

        PostProcessNode (outer post_process) reads `answer` from state to apply
        the output gate and set formatted_output.
        """
        return {
            # Outer reason wins: a reason settled before the inner run is the real
            # one, and a plain sub_result.get() would erase it.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
            "answer": sub_result.get("answer"),
            "generated_answer": sub_result.get("generated_answer"),
            "citations": sub_result.get("citations"),
            "filtered_passages": sub_result.get("filtered_passages"),
            "status": sub_result.get("status"),
        }

    def on_subgraph_error(self, state: AgentState, error: Exception) -> dict[str, Any]:
        """Return a clean outer error delta for an inner-graph failure.

        The inner error_log — a validation rejection's field-naming message, or
        the framework's traceback entry for a raising inner node — is carried
        onto the OUTER error_log, the internal channel the audit trail reads;
        an empty one degrades to a generic line. None of it is projected to the
        caller: get_output() publishes the closed-set envelope only.
        """
        messages = [str(m) for m in getattr(error, "error_log", []) or [] if m]
        if not messages:
            messages = ["The request could not be completed."]
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": messages,
        }

    def _parent_config(self) -> dict[str, Any]:
        """Forward the declared runtime settings to the inner graph.

        Node ``execute()`` takes ``state`` alone, so a graph's ``self.config``
        never reaches a node by itself — State is the only route in. This method
        reads config/config.yaml and exposes the declared ``retrieval`` settings
        under the LangGraph ``configurable`` key;
        DomainWorkflowGraph._extra_initial_state() copies them into the inner
        state so RetrieveNode / RerankFilterNode actually observe the configured
        ``top_k`` / ``score_threshold`` on the real ``.invoke()`` path. Only keys
        the file actually declares are forwarded; absent keys fall back to the
        node defaults.

        The declared ``llm`` settings (system_prompt_template / temperature /
        max_tokens) are forwarded for parity but are NOT exercised by this
        version — the answer path is deterministic, extractive grounding (see
        GenerateAnswerNode); they are the wiring point for a real language
        model, not a live code path.
        """
        cfg = load_runtime_config()
        retrieval = cfg.get("retrieval", {}) or {}
        llm = cfg.get("llm", {}) or {}
        declared = {
            "top_k": retrieval.get("top_k"),
            "score_threshold": retrieval.get("score_threshold"),
            "system_prompt_template": llm.get("system_prompt_template"),
            "temperature": llm.get("temperature"),
            "max_tokens": llm.get("max_tokens"),
        }
        return {"configurable": {k: v for k, v in declared.items() if v is not None}}


class ClinicalGuidelinesQAAgent(AgentBaseGraph):
    """Outer graph for HCR-C2-011 — clinical guidelines question answering.

    Inherits AgentBaseGraph directly (the framework base class). Domain logic is
    fully encapsulated in ClinicalGuidelinesGraphNode (main slot), which
    delegates to DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed):
        START → initialize → pre_process → main → post_process → finalize → END

    register_nodes() and get_output() are the only overrides:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode   (VERIFIED_EXTERNAL — caller trust gate)
      - main:         ClinicalGuidelinesGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode  (output gate)
      - get_output(): surfaces the domain advisory answer

    add_edges() is NOT overridden — backbone wiring belongs to the framework.

    The class name MUST match the config/agent.yaml `class:` field exactly.
    src/api/server.py imports this as `Graph` via the alias below.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the agent registry."""
        return "ClinicalGuidelinesQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = ClinicalGuidelinesGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Surface the domain advisory answer on the outer invoke() return.

        AgentBaseGraph.get_output() returns only the minimal ``{output, status,
        trace_id, correlation_id, node_history}`` envelope, which on the
        compiled success path drops the domain result — the fields the inner
        DomainWorkflowGraph produces (merged into outer state by
        ClinicalGuidelinesGraphNode.merge_output(): answer, generated_answer,
        citations, filtered_passages) and the gated PostProcessNode outputs
        (formatted_output, result). This override extends the base envelope so a
        successful invocation actually returns the advisory answer.

        On any non-success status the envelope is fail-closed AND closed-set:
          * ``output`` is None. The framework base resolves it as
            ``formatted_output or result`` WITHOUT consulting status, so it is
            re-resolved here: the pre-gate inner answer never rides out under an
            error, and neither does anything else — the caller reads the reason
            from ``error`` instead. Two properties make this load-bearing rather
            than theoretical — a FALSY ``formatted_output`` (``""``/``{}``/
            absent) does not suppress the fallback but ACTIVATES it, and the
            framework's node wrapper turns a raise into a bare ERROR partial
            that clears nothing at all.
          * ``result``, ``answer`` and the structured domain result
            (generated_answer / citations / filtered_passages) are withheld
            (None). PostProcessNode ALSO empties these at the source when it
            blocks (see _CLEARED_ON_BLOCK), so neither this withholding nor that
            clearing is the single thing standing between a refused answer and
            the caller.
          * ``error`` is ``{"reason": <code>}`` with the code drawn from
            PostProcessNode's ERROR_REASONS: the gate's own reason when it ran
            (``output_withheld``), ``workflow_failed`` otherwise — an errored
            main routes straight to finalize, so the gate never ran. A value in
            that slot outside the closed set is replaced, never echoed.
          * ``error_log`` is NOT projected. It is node-authored text — the
            framework writes ``[Node] <message>`` plus a full traceback with
            absolute source paths there whenever a node raises, and an entry can
            quote whatever the failing node was handed — so summarising or
            redacting it is not a closed-set contract; not publishing it is. It
            stays the internal channel (state reducer, audit trail).
        """
        output: dict[str, Any] = super().get_output(state)  # {output, status, trace_id, correlation_id, node_history}
        # A run that completed WITHOUT carrying out the request holds the
        # sentence saying what to correct, not a product: none of the
        # structured fields below were produced, so none is released.
        if state.get("error_code"):
            return output
        succeeded = state.get("status") == AgentStatus.SUCCESS.value
        formatted_output = state.get("formatted_output")

        if not succeeded:
            reason = _REASON_WORKFLOW_FAILED
            if isinstance(formatted_output, dict):
                candidate = formatted_output.get("reason")
                if isinstance(candidate, str) and candidate in ERROR_REASONS:
                    reason = candidate
            output["output"] = None
            output["formatted_output"] = None
            output["result"] = None
            output["answer"] = None
            output["generated_answer"] = None
            output["citations"] = None
            output["filtered_passages"] = None
            output["error"] = error_envelope(reason)
            return output

        # Caller-facing, already-gated value, plus the structured domain result
        # — surfaced only on the gated success path.
        output["formatted_output"] = formatted_output
        output["result"] = state.get("result")
        output["answer"] = formatted_output or state.get("result")
        output["generated_answer"] = state.get("generated_answer")
        output["citations"] = state.get("citations")
        output["filtered_passages"] = state.get("filtered_passages")
        return output

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.


# Alias for backward compat (src/api/server.py imports Graph).
# The class name ClinicalGuidelinesQAAgent matches the config/agent.yaml `class:` field.
Graph = ClinicalGuidelinesQAAgent
