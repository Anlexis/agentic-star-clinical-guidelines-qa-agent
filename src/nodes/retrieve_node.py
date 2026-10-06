"""AgentCore Platform v1.0"""

# HCR-C2-011 — RetrieveNode
# Inner domain node 2: retrieve candidate passages from the clinical
# guidelines knowledge base.
#
# This implementation performs deterministic lexical retrieval over a
# SYNTHETIC, in-module clinical-guidelines corpus. A deployment wires a real
# vector store (collection `hcr_clinical_guidelines__kb`) behind the same
# contract, driven by the declared retrieval settings seeded into State.
#
# Life-safety note: the corpus is a synthetic exemplar only — no patient
# records, no personal health information. It exists so the pipeline is
# runnable end-to-end without any external dependency.
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

_DEFAULT_TOP_K = 5

# ── Synthetic clinical-guidelines corpus (exemplar only — NOT real guidance) ──
# Each entry: id, title, source, text (advisory), keywords (lexical match set).
# Sources are labelled "(synthetic exemplar)" so no real MHLW/JCS text is implied.
_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "GL-HTN-001",
        "title": "Adult hypertension — initial management",
        "source": "Clinical Guidelines KB (synthetic exemplar)",
        "text": (
            "For newly diagnosed adult hypertension without compelling indications, "
            "lifestyle modification is advised first-line; pharmacotherapy is "
            "considered when blood pressure remains above target after review."
        ),
        "keywords": ["hypertension", "blood", "pressure", "adult", "management", "bp"],
    },
    {
        "id": "GL-SEP-002",
        "title": "Sepsis — initial resuscitation bundle",
        "source": "Clinical Guidelines KB (synthetic exemplar)",
        "text": (
            "In suspected sepsis, obtain cultures before antimicrobials where feasible, "
            "measure lactate, and begin empiric broad-spectrum therapy promptly per the "
            "initial resuscitation bundle."
        ),
        "keywords": ["sepsis", "resuscitation", "bundle", "lactate", "antibiotics", "infection"],
    },
    {
        "id": "GL-VTE-003",
        "title": "Venous thromboembolism — inpatient prophylaxis",
        "source": "Clinical Guidelines KB (synthetic exemplar)",
        "text": (
            "Assess VTE risk on admission; for at-risk medical inpatients without a high "
            "bleeding risk, pharmacological thromboprophylaxis is advised, otherwise "
            "mechanical prophylaxis is considered."
        ),
        "keywords": ["vte", "thromboembolism", "prophylaxis", "dvt", "bleeding", "inpatient"],
    },
    {
        "id": "GL-AC-004",
        "title": "Anticoagulation — common interaction cautions",
        "source": "Clinical Guidelines KB (synthetic exemplar)",
        "text": (
            "When starting anticoagulation, review concurrent agents that raise bleeding "
            "risk (for example antiplatelets and certain NSAIDs) and adjust monitoring "
            "of the interaction accordingly."
        ),
        "keywords": ["anticoagulation", "interaction", "bleeding", "warfarin", "nsaid", "drug"],
    },
    {
        "id": "GL-DKA-005",
        "title": "Diabetic ketoacidosis — early management priorities",
        "source": "Clinical Guidelines KB (synthetic exemplar)",
        "text": (
            "Early diabetic ketoacidosis management prioritises fluid resuscitation, "
            "electrolyte (notably potassium) correction, and an insulin infusion, with "
            "close monitoring of glucose and ketones."
        ),
        "keywords": ["dka", "ketoacidosis", "diabetic", "insulin", "potassium", "glucose"],
    },
    {
        "id": "GL-STR-006",
        "title": "Acute ischaemic stroke — time-critical pathway",
        "source": "Clinical Guidelines KB (synthetic exemplar)",
        "text": (
            "Acute ischaemic stroke is a time-critical pathway; confirm onset time, obtain "
            "urgent imaging, and assess eligibility for reperfusion therapy without delay."
        ),
        "keywords": ["stroke", "ischaemic", "reperfusion", "imaging", "acute", "onset"],
    },
]


def _score_passage(query_terms: List[str], entry: Dict[str, Any]) -> float:
    """Deterministic lexical relevance in [0, 1].

    Fraction of query terms found in the passage's keyword set or text.
    A perfectly matched query scores 1.0; partial matches scale linearly so
    the downstream score_threshold (>= 0.75) is meaningful.
    """
    if not query_terms:
        return 0.0
    haystack = set(entry.get("keywords", [])) | set(entry.get("text", "").lower().split())
    matched = sum(1 for t in query_terms if t in haystack)
    return round(matched / len(query_terms), 4)


class RetrieveNode(FunctionNode):
    """Retrieve top-k candidate passages for the clinical question.

    Deterministic lexical retrieval over a synthetic corpus. A deployment
    swaps in a real vector store behind the same declared retrieval settings,
    which reach this node through State.

    Inner node — ANONYMOUS trust (see module docstring).

    Input state keys:
        clinical_query:  str — JSON: {question, length, terms, context_chars}
        retrieval_top_k: int — number of candidates to keep; seeded from
                               configuration and optionally tightened by the caller

    Output state keys (partial dict):
        retrieved_passages: str  — JSON-serialised list of scored passages
        status:             str
        error_log:          list[str]  (only on ERROR)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> dict[str, Any]:
        # A reason settled earlier in the run is the real one: pass it through
        # untouched instead of doing work on input that was already declined.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        query: Dict[str, Any] = from_json(state.get("clinical_query"), {})
        terms: List[str] = list(query.get("terms", []))

        # Runtime precedence: the value in State wins, then the declared
        # default. ClinicalGuidelinesGraphNode._parent_config() forwards
        # config/config.yaml's retrieval.top_k, DomainWorkflowGraph
        # ._extra_initial_state() seeds it as retrieval_top_k, and
        # InputValidateNode may narrow it to a validated caller value. State is
        # the only route in — execute() takes state alone.
        raw_top_k = state.get("retrieval_top_k", _DEFAULT_TOP_K)
        if raw_top_k is None:
            raw_top_k = _DEFAULT_TOP_K
        try:
            top_k = int(raw_top_k)
        except (TypeError, ValueError):
            top_k = _DEFAULT_TOP_K

        if not terms:
            logger.warning("RetrieveNode: no query terms — returning empty candidate set")
            emit_trace_event(
                "retrieve_skipped",
                {"reason": "no_query_terms"},
                state,
            )
            return {
                "retrieved_passages": to_json([]),
                "status": AgentStatus.SUCCESS.value,
            }

        # Build scored candidates WITHOUT mutating the module-level corpus.
        scored: List[Dict[str, Any]] = []
        for entry in _CORPUS:
            score = _score_passage(terms, entry)
            if score <= 0.0:
                continue
            scored.append(
                {
                    "id": entry["id"],
                    "title": entry["title"],
                    "source": entry["source"],
                    "text": entry["text"],
                    "score": score,
                }
            )

        # Deterministic rank: score desc, then id asc for stable tie-breaking.
        scored.sort(key=lambda p: (-p["score"], p["id"]))
        candidates = scored[:top_k]

        logger.info(
            "RetrieveNode: terms=%d candidates=%d top_k=%d",
            len(terms),
            len(candidates),
            top_k,
        )
        emit_trace_event(
            "retrieve_complete",
            {
                "term_count": len(terms),
                "candidate_count": len(candidates),
                "top_k": top_k,
            },
            state,
        )

        return {
            "retrieved_passages": to_json(candidates),
            "status": AgentStatus.SUCCESS.value,
        }
