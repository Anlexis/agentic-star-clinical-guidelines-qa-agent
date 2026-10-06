"""AgentCore Platform v1.0"""

# HCR-C2-011 — DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph of the Cat 2 two-layer nested architecture.
# It encapsulates the clinical-guidelines question-answering pipeline:
#
#   START
#     → input_validate   (InputValidateNode)
#     → retrieve         (RetrieveNode)
#     → rerank_filter    (RerankFilterNode)
#     → generate_answer  (GenerateAnswerNode)
#     → output_format    (OutputFormatNode)
#     → END
#
# Called by ClinicalGuidelinesGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   ✅ Inherits BaseGraph (fully custom topology — no forced backbone)
#   ✅ Implements all 7 BaseGraph abstract methods
#   ✅ register_nodes() does NOT call super() (abstract in BaseGraph)
#   ✅ Does NOT register initialize / finalize (outer backbone concerns)
#   ✅ All inner nodes declare required_trust_level = TrustLevel.ANONYMOUS
#   ✅ get_output() designed together with ClinicalGuidelinesGraphNode.merge_output()
#   ✅ All inner node constructors are empty-parens (framework nodes take no arguments)
#   ❌ No platform-internal SDK imports

from langgraph.graph import END, START

from typing import Any

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for HCR-C2-011 (retrieve → answer pipeline).

    Inherits BaseGraph directly for a fully custom node topology.
    Called by ClinicalGuidelinesGraphNode.get_subgraph() in graph.py.

    Pipeline (linear):
        START
          → input_validate   (InputValidateNode)
          → retrieve         (RetrieveNode)
          → rerank_filter    (RerankFilterNode)
          → generate_answer  (GenerateAnswerNode)
          → output_format    (OutputFormatNode)
          → END

    All nodes are FunctionNode subclasses with ANONYMOUS trust_level.
    initialize / finalize are outer backbone concerns — not registered here.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "hcr_c2_011_clinical_guidelines_qa_workflow"

    @property
    def state_schema(self) -> type:
        return State

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """No mandatory config keys for the deterministic inner pipeline.

        Every runtime setting the inner graph consumes has a declared default in
        the nodes themselves, so an empty config is a valid configuration.
        """
        pass

    # ── Runtime config seeding ────────────────────────────────────────────────

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed the declared runtime config and the caller context into inner state.

        A graph's ``self.config`` never reaches a node by itself — ``execute()``
        takes ``state`` alone, so State is the only route from configuration into
        a node. The outer GraphNode forwards config/config.yaml's declared
        ``retrieval`` settings via _parent_config() →
        ``self.config["configurable"]``; here we copy the declared ``top_k`` and
        ``score_threshold`` (the clinical-confidence threshold) into the inner
        state so RetrieveNode / RerankFilterNode observe them end-to-end.
        Absent or unparseable values are omitted and the nodes fall back to
        their declared defaults.

        The caller's ``input_context`` is seeded here too: GraphNode.execute()
        does not forward it on subgraph.invoke(), so
        ClinicalGuidelinesGraphNode.extract_input() stashes it via the context
        bridge immediately before the inner invoke and this hook — called while
        BaseGraph.invoke builds the initial state — reads it back. Inner domain
        nodes keep their plain ``state["input_context"]`` reads.
        InputValidateNode validates every field of it before any of it is used.
        """
        configurable = (self.config or {}).get("configurable", {}) or {}
        seeded: dict[str, Any] = {"input_context": get_caller_input_context()}

        raw_top_k = configurable.get("top_k")
        if raw_top_k is not None:
            try:
                seeded["retrieval_top_k"] = int(raw_top_k)
            except (TypeError, ValueError):
                pass

        raw_threshold = configurable.get("score_threshold")
        if raw_threshold is not None:
            try:
                seeded["retrieval_score_threshold"] = float(raw_threshold)
            except (TypeError, ValueError):
                pass

        return seeded

    # ── Node registration ─────────────────────────────────────────────────────

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call — BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.
        Every key registered here is referenced in add_edges().
        All nodes are instantiated with empty parentheses — FunctionNode
        subclasses take no constructor arguments.
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # ── Edge wiring ───────────────────────────────────────────────────────────

    def add_edges(self) -> None:
        """Wire the linear retrieval-and-answer topology.

        Linear flow:
            input_validate → retrieve → rerank_filter → generate_answer
            → output_format → END.

        No conditional branching — every path through the pipeline is linear,
        so route() satisfies the abstract contract but is not used at runtime.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, state: AgentState) -> str:
        """Conditional routing — required by the BaseGraph abstract contract.

        Linear topology; add_conditional_edges() is not used, so this method
        is never called at runtime.  Returns END on error so an unexpected
        invocation does not re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # ── Output shape ──────────────────────────────────────────────────────────

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by ClinicalGuidelinesGraphNode.merge_output()
        in graph.py as the `sub_result` argument.  Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output() emits:   "answer", "generated_answer",
                                        "citations", "filtered_passages", "status"
            Outer merge_output() reads: sub_result.get(...) for each key above.

        ``error_log`` is emitted too, and it is load-bearing rather than
        incidental: on a validation rejection the inner node's message is what
        names the offending FIELD, and the framework reads exactly this key when
        it builds the error the outer GraphNode receives. Omitting it makes the
        rejection arrive at the caller as an empty failure — fail-closed, but
        with no way to tell which field was wrong.
        """
        return {
            # the reason must leave the subgraph or the outer graph cannot report it
            "error_code": state.get("error_code"),
            "answer": state.get("answer"),
            "generated_answer": state.get("generated_answer"),
            "citations": state.get("citations"),
            "filtered_passages": state.get("filtered_passages"),
            "status": state.get("status"),
            "error_log": state.get("error_log", []),
            "node_history": state.get("node_history", []),
            "correlation_id": state.get("correlation_id"),
        }
