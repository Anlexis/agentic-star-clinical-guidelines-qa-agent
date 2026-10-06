# Template Design Specification — HCR-C2-011 Clinical Guidelines Q&A Agent

## Position in the AgentCore Architecture

| Item | Value |
|---|---|
| Agent class | ClinicalGuidelinesQAAgent |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Pattern | Cat 2 — retrieval-and-answer pipeline in a two-layer nested workflow |

**Three-layer separation**

- **State** — flat TypedDict composition (never Pydantic; checkpoint serialization is msgpack, which
  Pydantic objects silently corrupt). All dict/list-valued fields are stored as JSON-serialized
  strings.
- **Node** — framework inheritance; a node overrides `execute(self, state: dict) -> dict` and
  nothing else.
- **Graph** — composition. `register_nodes()` substitutes nodes into the fixed backbone slots; the
  nested layer is introduced with a `GraphNode`.

## Domain Context

A clinical guidelines question-answering assistant for licensed healthcare professionals. Clinical
staff and residents ask about treatment protocols, drug-interaction cautions and guideline
recommendations; the agent retrieves from a clinical-guidelines knowledge base and returns an
**advisory-only** answer with citations.

**Life-safety guardrails**

- `temperature: 0.0` — deterministic, reproducible clinical output when a real language model is
  wired in.
- Retrieval confidence floor `score_threshold: 0.75` — low-confidence evidence is dropped rather
  than used to ground an advisory answer; an explicit "insufficient evidence" response is returned
  instead. A caller may raise this floor but never lower it.
- **Synthetic knowledge base only** — the bundled corpus is a synthetic exemplar. No patient
  records and no personal health information are held in the knowledge base or in State.
- **Mandatory advisory disclaimer on every response**: *"This output is advisory only. Clinical
  judgment of the attending physician takes precedence."* Its presence is enforced at the output
  gate, not merely produced upstream.

## Architecture Overview

### Backbone (outer AgentBaseGraph — fixed 5-node pipeline)

```
START → initialize → pre_process → main(GraphNode) → post_process → finalize → END
                                         ↓ (retry, max_retry)
                                       pre_process
```

### Inner domain workflow (DomainWorkflowGraph — linear 5-node pipeline)

```
START → input_validate → retrieve → rerank_filter → generate_answer
          → output_format → END
```

### Node configuration

| Node | Class | File | Trust | Responsibility | Input keys | Output keys |
|------|-------|------|-------|---------------|------------|-------------|
| initialize | InitializeNode | framework | — | session init | — | session_id, schema_version |
| pre_process | PreProcessNode | src/nodes/pre_process_node.py | VERIFIED_EXTERNAL | caller trust gate; size, injection and identifier screening; channel label check | user_input, input_context | validated_input, enriched_context |
| main | ClinicalGuidelinesGraphNode | src/graph/graph.py | — | delegates to DomainWorkflowGraph; bridges input_context | validated_input, input_context | answer, generated_answer, citations, filtered_passages |
| post_process | PostProcessNode | src/nodes/post_process_node.py | ANONYMOUS | output gate: disallowed-content scan + mandatory-disclaimer enforcement + block containment | answer | formatted_output, result (+ the `_CLEARED_ON_BLOCK` inventory, emptied on a block) |
| finalize | FinalizeNode | framework | — | response metadata | — | response_metadata, total_time_ms |
| input_validate (inner) | InputValidateNode | src/nodes/input_validate_node.py | ANONYMOUS | caller-data contract; query-term extraction | validated_input, input_context | clinical_query, retrieval_top_k, retrieval_score_threshold |
| retrieve (inner) | RetrieveNode | src/nodes/retrieve_node.py | ANONYMOUS | top-k retrieval from the guidelines corpus | clinical_query, retrieval_top_k | retrieved_passages |
| rerank_filter (inner) | RerankFilterNode | src/nodes/rerank_filter_node.py | ANONYMOUS | rerank + confidence-floor filter | retrieved_passages, retrieval_score_threshold | filtered_passages |
| generate_answer (inner) | GenerateAnswerNode | src/nodes/generate_answer_node.py | ANONYMOUS | grounded advisory answer + citations | clinical_query, filtered_passages | generated_answer, citations |
| output_format (inner) | OutputFormatNode | src/nodes/output_format_node.py | ANONYMOUS | assemble answer + citations + disclaimer | generated_answer, citations | answer, result |

### Data flow

```
user_input (clinical question, free text)          input_context (structured caller parameters)
    │                                                   │
    ▼ PreProcessNode (VERIFIED_EXTERNAL)                 │
validated_input   (screened, normalised question)        │
enriched_context  (JSON string)                          │
    │                                                    │
    ▼ ClinicalGuidelinesGraphNode ──── context bridge ───┘
    │   → DomainWorkflowGraph
    │   InputValidateNode  → clinical_query (JSON string) [+ validated retrieval overrides]
    │   RetrieveNode       → retrieved_passages (JSON string)
    │   RerankFilterNode   → filtered_passages (JSON string, score ≥ threshold)
    │   GenerateAnswerNode → generated_answer (str), citations (JSON string)
    │   OutputFormatNode   → answer (str), result (str)
    ▼ merge_output
answer, generated_answer, citations, filtered_passages → outer state
    │
    ▼ PostProcessNode (output gate)
formatted_output (gated answer), result
```

**The caller context bridge.** `GraphNode.execute()` invokes the inner graph without forwarding the
outer state's `input_context`, so a nested template would see `{}` on every inner read.
`src/graph/context_bridge.py` closes that gap with a `ContextVar`: the outer
`ClinicalGuidelinesGraphNode.extract_input()` stashes the caller context immediately before the
inner invoke, and `DomainWorkflowGraph._extra_initial_state()` reads it back while the inner initial
state is being built. A `ContextVar` keeps the hand-off per-thread, so concurrent invocations in one
process cannot see each other's context. This is proven end to end, not at node level.

### State definition

| Field | Type | Purpose | Producer |
|-------|------|---------|----------|
| validated_input | NotRequired[Optional[str]] | Screened, normalised clinical question | PreProcessNode |
| enriched_context | NotRequired[Optional[str]] | JSON: {source, channel} | PreProcessNode |
| retrieval_top_k | NotRequired[Optional[int]] | Candidate count in force for this run | graph seed / InputValidateNode |
| retrieval_score_threshold | NotRequired[Optional[float]] | Confidence floor in force for this run | graph seed / InputValidateNode |
| clinical_query | NotRequired[Optional[str]] | JSON: {question, length, terms, context_chars} | InputValidateNode |
| retrieved_passages | NotRequired[Optional[str]] | JSON: scored candidate passages | RetrieveNode |
| filtered_passages | NotRequired[Optional[str]] | JSON: passages ≥ threshold | RerankFilterNode |
| generated_answer | NotRequired[Optional[str]] | Grounded advisory answer text | GenerateAnswerNode |
| citations | NotRequired[Optional[str]] | JSON: citation list {id, title, source} | GenerateAnswerNode |
| answer | NotRequired[Optional[str]] | Final formatted answer document | OutputFormatNode |
| result | NotRequired[Optional[str]] | Gated caller-facing content | OutputFormatNode / PostProcessNode |

**Serialization constraint**: all dict/list-valued fields use JSON-serialised `Optional[str]`.
`to_json()` / `from_json()` are defined in `src/schemas/state.py` and used at every producer and
consumer boundary — one contract end to end.

**Prohibited**: re-declaring `formatted_output` (inherited from AgentState); credentials or personal
health information in State; Pydantic models.

## Caller-Data Contract

`POST /invoke` accepts the clinical question as `input` and optional structured parameters as
`input_context` (the framework's first-class invocation parameter). The HTTP adapter caps the whole
serialized `input_context` at 256 KB; every field is then validated individually by the node that
owns the contract.

| Field | Owner | Rule |
|---|---|---|
| `channel` | PreProcessNode | must match `[a-z0-9_]{1,32}` — a caller-controlled string that is stored and later rendered is output injection unless it is locked to an inert identifier |
| `patient_context` | InputValidateNode | string, ≤ 1000 characters after whitespace normalisation; direct-identifier-shaped tokens are redacted first. Its terms widen retrieval; it is never rendered into the answer |
| `top_k` | InputValidateNode | finite whole number in 1–20 |
| `score_threshold` | InputValidateNode | finite number between the **configured floor** and 1.0 — a caller may make retrieval stricter, never looser, so no request can widen the evidence base the agent will ground on |

**Every caller-controlled number goes through `_finite_in_range()`**, which rejects booleans
(`isinstance(True, int)` is True in Python, so a bare numeric check would admit `true` as 1),
non-numeric types, `NaN`, `±Infinity` and out-of-range magnitudes. This is not defensive
decoration: `float("NaN")` parses cleanly and arrives intact through raw JSON, and every IEEE
comparison against `NaN` is False — so a `NaN` confidence floor would admit *every* passage, a
fail-open on precisely the decision this template exists to make. Validation therefore fails
**closed**, and the error names the offending field and never echoes the rejected value.

Absent caller data is not an error: the pipeline degrades to the values declared in
`config/config.yaml`, and still computes a real answer from the question alone.

## Security Configuration

| Layer | Gate | Implementation |
|-------|------|---------------|
| Caller trust | Trust enforcement | PreProcessNode `required_trust_level = VERIFIED_EXTERNAL`; the HTTP adapter promotes a bearer-authenticated caller to that level and never demotes middleware-established trust |
| Input screening | Injection refusal | PreProcessNode `detect_prompt_injection()` — instruction-override, instruction-disclosure, persona-override and role-tag shapes are refused **inside `execute()`**, so the guarantee holds on any path that reaches this node rather than depending on a gate running in front of it |
| Input screening | Identifier redaction | PreProcessNode `strip_direct_identifiers()` on the question; the same screen is re-applied by InputValidateNode to `input_context.patient_context`, which does not pass through PreProcessNode and is outside the framework's own input mask |
| Input validation | Bounded caller data | InputValidateNode `_finite_in_range()` + structural caps (see the caller-data contract above) |
| Output gate | Disallowed content | PostProcessNode `_security_gate_output()` — a module-level function called from `execute()`, scanning recursively for API keys, signed tokens, bearer tokens, credential assignments, and direct-identifier shapes. Credential recognition is the **union** of those domain patterns and the framework's own `detect_credentials()`: the framework scans every node result with that detector and *raises* on a hit, and a raise makes the node wrapper discard the node's whole return — including this gate's containment. A shape the framework catches and the domain set missed would therefore be a containment **bypass**, not merely a narrower gate, so the two cannot drift apart by construction |
| Output gate | Block containment | Blocking is not a status flip. The gate overwrites **every** field carrying answer text or a payload (`formatted_output`, `result`, `answer`, `generated_answer`, `citations`, `filtered_passages` — the inventory `_CLEARED_ON_BLOCK`, held in step with `merge_output()` by a test), and the `formatted_output` replacement is the closed-set envelope `{"reason": "output_withheld"}` — deliberately **non-empty**: the envelope resolves `formatted_output or result`, so a falsy stand-in would not suppress that fallback but ACTIVATE it. One module-level `_contain(reason, new_errors)` helper builds every ERROR return (both gate arms and an already-errored state) |
| Output gate | Mandatory disclaimer | PostProcessNode attaches the advisory disclaimer if absent, then **verifies** it; a response that still lacks it is blocked rather than shipped |
| Error surface | Closed-set labels only | On any non-success status the invoke envelope carries `error: {"reason": <code>}` with `code` in `ERROR_REASONS = {workflow_failed, output_withheld}` (PostProcessNode's own reason when the gate ran; `workflow_failed` otherwise — an errored `main` routes straight to `finalize`), `output: null`, every answer slot `null`, and **no `error_log` key**. `error_log` is node-authored text — the framework writes `[Node] <message>` plus a full traceback with absolute source paths into it whenever a node raises, and an entry can quote whatever the failing node was handed (an identifier, a name, an e-mail, an upstream response body) — so summarising or redacting it is not a closed-set contract; it stays the internal channel the state reducer appends to and the audit trail reads. `get_output()` is the one choke point both the outer-node and inner-subgraph routes pass through |
| Audit | Event logging | `emit_trace_event()` in every node's `execute()`, on both the success and rejection paths |
| Credentials | Handling | No credentials or personal health information in State; secrets are reached only through the invocation context, and this template declares none |

### The output invariant, and why there is no rounding grid

A common output-boundary control in this fleet is a numeric **precision grid** — monetary values
snapped to a stated rounding boundary so a report can never leak an exact figure. **It does not
apply here and is deliberately not implemented**: this template renders no monetary aggregates and
publishes no numeric report. Its stated output invariants are different ones — no disallowed
content, and the advisory disclaimer present on every response — and those are what the gate
enforces, completely, for every representation.

That decision also avoids a real hazard. A rounding grid reads any standalone three-letter
uppercase word as a currency marker, which would mangle exactly the identifiers this domain renders
(guideline citation keys such as `GL-HTN-001`) and, worse, could rewrite an identifier-shaped
sequence into something the pattern scan no longer recognises — destroying the evidence and shipping
the result. Because this gate performs no numeric rewriting, the scan runs on the answer exactly as
produced. Both directions are pinned by tests: guideline citation keys pass through byte-identical,
and identifier-shaped sequences are redacted rather than mangled.

### Runtime configuration

`config/agent.yaml` is the **static manifest** — a flat document read at root level (identity,
category, entry-point class, required trust level, and the compile-time `requires` block). It holds
no runtime values.

`config/config.yaml` holds the runtime parameters:

```yaml
max_retry: 3
timeout_s: 30

llm:
  system_prompt_template: prompts/hcr_qa.j2
  temperature: 0.0
  max_tokens: 4000

retrieval:
  collection: hcr_clinical_guidelines__kb
  top_k: 5
  score_threshold: 0.75
  hybrid_search: false

security:
  s3_gate_enabled: true
```

The whole document is passed to the graph constructor by `src/api/server.py`, so `max_retry` — which
the framework base class validates and the backbone's retry router consumes — is actually in force
rather than falling back to a framework default. The `retrieval` block reaches the domain nodes
through `ClinicalGuidelinesGraphNode._parent_config()` → `DomainWorkflowGraph._extra_initial_state()`,
which seeds it into State: a node's `execute()` takes `state` alone, so State is the only route from
configuration into a node.

## Framework Utilization

### Shared components used
- [x] Invocation context (`session_id`, caller trust level, `input_context`)
- [x] Module-level `_security_gate_output()` in `post_process_node.py` — recursive output scan
- [x] `emit_trace_event()` — at least one domain-specific event per node `execute()`
- [x] `to_json()` / `from_json()` helpers in `src/schemas/state.py` — the serialisation contract

### Composition pattern

- **Pattern**: two-layer nested — a `GraphNode` wrapping an inner `BaseGraph`
- **Outer graph**: `ClinicalGuidelinesQAAgent(AgentBaseGraph)` — fixed 5-node backbone
- **Inner graph**: `DomainWorkflowGraph(BaseGraph)` — 5-node linear pipeline
- **Error propagation**: propagate — an inner failure raises `SubgraphError`, which the framework
  converts into `status=error` with the field-naming message intact, and the backbone routes
  straight to finalize (post_process is skipped, so no partially-formed answer can be emitted)

## Import Isolation Confirmation
- [x] The template does not import the platform-internal SDK
- [x] Import targets: `framework/` and `shared/` only

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Framework base class | AgentBaseGraph | AutonomousBaseGraph | AgentBaseGraph | Fixed sequential pipeline; no reasoning loop required |
| Composition pattern | flat | nested GraphNode | nested | 5 sequential retrieval steps behind one backbone slot |
| Retrieval confidence | answer on any hit | floor at 0.75 | floor + insufficient-evidence path | Never ground an advisory answer on weak evidence |
| Caller threshold control | free choice | tighten-only | tighten-only | A caller must not be able to widen the evidence base below the deployment's declared floor |
| State dict fields | bare dict | JSON-serialised str | JSON-serialised str | msgpack checkpoint safety |
| Answer generation | live language model | deterministic extractive grounding | deterministic extractive grounding | Runs with no external dependency and cannot hallucinate; the declared `llm` settings are the wiring point for a real model |
| Knowledge base | real clinical data | synthetic exemplar corpus | synthetic corpus | No personal health information; guideline text only |
| Disclaimer enforcement | produced by a domain node | verified at the output gate | verified at the output gate | One choke point every response passes; a future edit upstream cannot silently drop it |
