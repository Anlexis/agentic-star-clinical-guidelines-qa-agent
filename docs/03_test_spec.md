# Test Specification — HCR-C2-011 Clinical Guidelines Q&A Agent

## 1. Test Strategy

- **Agent:** HCR-C2-011 — Clinical Guidelines Q&A Agent (two-layer nested graph:
  outer `AgentBaseGraph` backbone + inner `DomainWorkflowGraph`).
- **Coverage target:** ≥ 90% of the branches in `src/nodes/` and `src/graph/`.
- **Test types:** unit (per node + graph wiring) · proof-of-boundary (framework
  security and serialization contracts, server boot, end-to-end `/invoke`) ·
  integration (full `Graph().invoke()`).
- **Framework provisioning:** `framework` (agenticstar-agentcore) is installed
  from the package registry. The tests import the real modules — there are no
  stub nodes anywhere in this suite.
- **Audit events:** `emit_trace_event` is patched at the node module level, never
  via a `sys.modules` stub (which would break the real `shared` package the
  framework loads at import time).

### Life-safety invariants under test

This template answers over a **synthetic** clinical-guidelines knowledge base
only (no personal health information). Four properties are asserted end to end:

1. **Confidence-floor abstention** — `RerankFilterNode` drops any passage below
   the score threshold (0.75 by default); an all-low-confidence retrieval yields
   an empty filtered set rather than grounding an advisory answer on weak
   evidence.
2. **Grounded-only generation** — `GenerateAnswerNode` synthesises the answer
   ONLY from passages that survived the filter; with none, it returns an explicit
   *insufficient-evidence* advisory instead of fabricating guidance.
3. **The advisory disclaimer is present on every response** — produced upstream
   by `GenerateAnswerNode` / `OutputFormatNode` and independently **enforced** at
   the output gate: `PostProcessNode` attaches it if absent, verifies it, and
   blocks the response if verification fails.
4. **Direct identifiers never reach the answer** — the input screen redacts
   identifier-shaped tokens on both caller channels, and the output gate blocks
   any that appear anyway.

### Caller-contract invariants under test

5. **The template owns its refusals** — the prompt-injection screen is asserted by
   calling `execute()` DIRECTLY, with no framework wrapper in front of it, so the
   guarantee is the template's own rather than a gate's. Both directions are
   covered: attack forms are refused, and ordinary clinical wording reusing the
   same vocabulary ("what are the *instructions* for an insulin infusion?") is
   not.
6. **Every caller-supplied number is finite and bounded** — a parametrized matrix
   (`"NaN"`, `"Infinity"`, `"-Infinity"`, `float("nan")`, `float("inf")`,
   `float("-inf")`, `True`, a non-numeric string, a non-numeric type) is applied
   to *each* numeric field, at node level and again through `/invoke`.
7. **The confidence floor can be tightened, never lowered** — a caller request to
   drop below the configured floor is rejected.

### Test file map

| File | Tests | Scope |
|------|-------|-------|
| `tests/unit/test_nodes.py` | 131 | All 7 nodes (outer pre/post_process + 5 inner nodes), the caller-data contract, the output invariant, the context bridge, runtime-config wiring, and outer/inner graph wiring |
| `tests/unit/test_output_envelope.py` | 25 | The outer `invoke()` envelope (`ClinicalGuidelinesQAAgent.get_output`): success control vs. every non-success status; the output gate's block containment and its `_CLEARED_ON_BLOCK` inventory guard; the error surface projects no `error_log` entry in any form; detector parity with the framework recogniser (both directions) |
| `tests/unit/test_error_envelope_closed_set.py` | 73 | The caller-visible ERROR envelope is a closed set: PostProcessNode's `formatted_output` on every non-success path (errored state, each gate arm, disclaimer unverifiable — through `node(state)` and `execute()`) and `get_output()` on every non-success state (every non-success status, gate block, non-dict / foreign / non-string / empty reason) carry only the module's declared constants, stay truthy, withhold every answer slot, and never carry a sentinel seeded into `error_log` (a name, an e-mail and a token-shaped fragment, nested keys and values walked); the builder refuses an unknown reason without echoing it; the probe is shown to find the sentinel where it lives |
| `tests/unit/test_framework_compliance_tc06_tc07.py` | 2 | The framework's final input/output gates cannot be overridden |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | 7 | PB-6 per-node + backbone invoke order (VERIFIED_EXTERNAL), trust-gate denial, payload alignment |
| `tests/proof_of_boundary/test_invoke_e2e.py` | 23 | PB end-to-end through the real ASGI `POST /invoke` (bearer auth), including output-gate containment with its clean-path control |
| `tests/proof_of_boundary/test_pb_error_envelope_closed_set.py` | 15 | PB end-to-end through the real ASGI `POST /invoke`: a sentinel seeded into `error_log` on the data path (inner node returns the upstream body / inner node raises with it / outer node raises with it / logged then the output gate blocks) appears nowhere in the body; every such body is `error: {"reason"}`, `output: null`, no `error_log` key; clean-path control and probe self-check |
| `tests/proof_of_boundary/test_server_boot.py` | 5 | PB-BOOT — the entry point imports, compiles, and carries the declared runtime config |
| `tests/proof_of_boundary/test_import_isolation.py` | 1 | PB-4 platform-internal import isolation (AST scan) |
| `tests/proof_of_boundary/test_state_safety.py` | 1 | PB-2/PB-5 State serialization + credential safety (AST scan) |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | 1 (skip) | PB-7 interrupt propagation — skip stub, no cross-boundary propagation in this template |
| `tests/integration/test_outer_invoke_returns_domain_result.py` | 2 | The compiled outer graph surfaces the domain result; the output gate withholds it when blocked |

### Canonical valid payload (PB-6 `_VALID_PAYLOAD`)

The invoke payload is a **free-text clinical question**, not a JSON document. The
backbone invoke test and `deploy/invoke_payload.json` MUST use the identical
string — asserted by `test_invoke_payload_matches_pb6`:

```
What is the initial management of adult hypertension and blood pressure?
```

Retrieval: the non-stopword terms (initial / management / adult / hypertension /
blood / pressure) match the synthetic `GL-HTN-001` passage with a lexical score
of `5/6 ≈ 0.83`, at or above the `0.75` floor, so one passage survives and a
grounded (non-abstention) advisory answer is produced.

## 2. Framework Compliance Tests (Mandatory)

| TC-ID | Test | Expected result | Where |
|-------|------|----------------|-------|
| TC-01 | State contract: flat `TypedDict`, domain fields `NotRequired`, no Pydantic/dataclass | AST scan: 0 violations | `test_state_safety.py` |
| TC-02 | Empty / whitespace / oversized question rejected at PreProcessNode | `status=error`, error_log populated | `TestPreProcessNode` |
| TC-03 | No signed token or credential in State | AST scan: 0 violations | `test_state_safety.py` |
| TC-04 | `execute(self, state)` contract — no `_invoke_impl` | signature `(self, state)`; `_invoke_impl` absent | `TestExecuteContract` |
| TC-05 | `emit_trace_event()` called inside each node `execute()` | ≥1 domain event per node, on success and rejection paths | `TestPreProcessNode`, `TestInputValidateNode`, … |
| TC-06 | The framework's default input gate cannot be overridden | overriding raises at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-07 | The framework's default output gate cannot be overridden | overriding raises at class definition | `test_framework_compliance_tc06_tc07.py` |
| TC-08 | `required_trust_level` enforced in `__call__` before `execute()` | ANONYMOUS caller → refused; VERIFIED_EXTERNAL → admitted | `TestTrustGate` |
| TC-08a | Outer `PreProcessNode` = VERIFIED_EXTERNAL; inner nodes + post_process = ANONYMOUS | trust levels asserted per node | `test_trust_level_*` |
| TC-11 | Output gate on post_process | credential pattern → blocked + `status=error`; clean → pass | `TestPostProcessNode`, `TestOutputInvariant` |

## 3. Proof-of-Boundary Tests (Mandatory)

| PB-ID | Boundary | Test | Expected result | Where |
|-------|----------|------|----------------|-------|
| PB-2 | State serialization | AST scan of `src/schemas/state.py` | primitives only; no Pydantic/dataclass | `test_state_safety.py` |
| PB-4 | Import isolation | AST scan of `src/` | 0 platform-internal imports | `test_import_isolation.py` |
| PB-5 | Checkpoint safety | no credential-named fields or prohibited types in State | inspection pass | `test_state_safety.py` |
| PB-6 | Invoke execution order (per node) | `__call__`: node_start → input gate → `execute()` → output gate → node_complete | order verified for every `src/nodes/` class | `TestInvokeOrder` |
| PB-6b | Backbone invoke order | full `Graph().invoke(_VALID_PAYLOAD, ctx=VERIFIED_EXTERNAL)` | `status=success`; node_history = `[Initialize, PreProcess, ClinicalGuidelinesGraphNode, PostProcess, Finalize]` | `TestBackboneInvokeOrder` |
| PB-6c | Real external caller | `InvocationContext(caller_trust_level=VERIFIED_EXTERNAL)` — **never** `for_internal()` | inner ANONYMOUS nodes accept the passthrough trust; SUCCESS end to end | `TestBackboneInvokeOrder` |
| PB-6d | Trust denial at node level | `PreProcessNode` called with an ANONYMOUS caller | the gate denies before `execute()`; `status=error` | `TestTrustGate` |
| PB-6e | Payload alignment | `deploy/invoke_payload.json["input"] == _VALID_PAYLOAD` | the deployment's first invoke exercises the PB-6 payload | `test_invoke_payload_matches_pb6` |
| PB-7 | Interrupt propagation | skip stub — `propagate_hitl=False`, no cross-boundary interrupt checkpoint | skipped with reason (a real assertion once propagation is wired) | `test_pb7_hitl_interrupt_propagation.py` |
| PB-BOOT | Server boot | `import src.api.server` | app + compiled agent built; declared `max_retry` / `timeout_s` / `score_threshold` in force | `test_server_boot.py` |

### End-to-end through `POST /invoke` (`test_invoke_e2e.py`)

The app is driven through its real ASGI interface (no test client — `httpx` is
only a transitive dependency and must not become a test requirement). Every
request crosses the entry-point authentication, the outer trust and input
screens, the context bridge into the inner graph, all five domain nodes, and the
output gate.

| Case | Expected result |
|------|----------------|
| Question alone | grounded, cited advisory answer with the disclaimer; structured result rides along |
| `patient_context` supplied | the cited guideline **changes** (`GL-HTN-001` → `GL-SEP-002`) — the question text alone cannot produce that answer, so the bridge is proven end to end |
| `score_threshold: 0.95` | the same question abstains; `filtered_passages` empty; disclaimer still present |
| Malformed caller data (11 parametrized cases across `score_threshold`, `top_k`, `patient_context`) | `status=error`, `error: {"reason": "workflow_failed"}`, `output: null`, no answer/citations/filtered_passages, no `error_log` key, neither the field name nor a traceback or file path in the body (which field was rejected is held at unit level) |
| Rejected value echo | the supplied value appears nowhere in the response |
| **Output-gate containment** — clean-path control | the same request over the untouched corpus returns the full grounded advisory: `answer` with the disclaimer, `GL-HTN-001` cited, `result == formatted_output`. A refuse-everything gate would pass every containment row below, and for a clinical template that is the worse outcome — so this control is load-bearing, not decorative |
| **Output-gate containment** — blocked answer | fault injected on the DATA path (the retrieval transport returns a passage carrying a disallowed value, as a drifted knowledge base would — never by patching the gate): `status=error`, `PostProcessNode` present in `node_history` (proving the block happened AT the gate, not upstream), `result` and every structured domain field `None`, no guideline text, no citation key, no echo of the refused value, `output: null`, `error: {"reason": "output_withheld"}` and no `error_log` key (the gate's own message stays internal) |
| **Output-gate containment** — error surface | a credential shape the framework recognises is refused inside the inner graph; the body is `error: {"reason": "workflow_failed"}` with no `error_log` key, no `Traceback`, no `.py`, no absolute source path and no fragment of the framework's message |
| `input_context` over 256 KB | `413` at the adapter, before the graph runs |
| Instruction-override payload | refused; no answer produced |
| Question containing a record number | answer is grounded, the identifier is redacted, and the raw digits appear nowhere in the response |
| Rendered guideline keys | `[GL-HTN-001]` survives the output gate byte-identical |
| Missing bearer token | `401` with a generic body |

## 4. Business Logic Tests

| BL-ID | Test | Input | Expected result | Where |
|-------|------|-------|----------------|-------|
| BL-01 | Happy-path grounded Q&A | `_VALID_PAYLOAD` | advisory answer with the `CLINICAL GUIDELINES Q&A` header + `GL-HTN-001` citation + disclaimer | `TestBackboneInvokeOrder`, `TestInvokeEndToEnd` |
| BL-02 | Query-term extraction | clinical question | stopwords dropped; content terms kept and deduped | `test_valid_question_extracts_terms` |
| BL-03 | Lexical retrieval + ranking | hypertension terms | `GL-HTN-001` first; scores sorted descending | `TestRetrieveNode` |
| BL-04 | `top_k` limiting | `top_k=1`, multi-match terms | exactly 1 candidate returned | `test_top_k_limits_result_count` |
| BL-05 | Confidence-floor abstention | all candidates < 0.75 | filtered set empty | `test_high_threshold_abstention_drops_low_confidence` |
| BL-06 | Seeded threshold in force | `score_threshold=0.55` | a 0.60 passage is kept | `test_seeded_threshold_overrides_the_default` |
| BL-07 | Grounded answer + citations | 1 filtered passage | answer cites `GL-HTN-001`; citations non-empty | `test_grounded_answer_cites_filtered_passages` |
| BL-08 | Insufficient-evidence advisory | 0 filtered passages | explicit "insufficient" advisory + disclaimer; empty citations | `test_no_evidence_returns_insufficient_advisory` |
| BL-09 | Disclaimer on every path | grounded AND abstention | the disclaimer is in every answer, and is enforced at the gate | `test_disclaimer_always_present`, `TestOutputInvariant` |
| BL-10 | Output assembly | generated_answer + citations | header + citations block + disclaimer; `result == answer` | `TestOutputFormatNode` |
| BL-11 | Graph key coupling | inner `get_output` ↔ outer `merge_output` | 5 coupled keys mapped; `merge_output` returns changed keys only | `TestOuterGraphComposition`, `TestInnerDomainGraph` |
| BL-12 | Corpus immutability | any retrieve invocation | the module-level corpus is unchanged (no cross-invocation leak) | `test_module_corpus_not_mutated` |
| BL-13 | Caller context widens retrieval | `patient_context` with domain terms | the terms reach the query; the cited guideline changes | `test_patient_context_terms_widen_retrieval`, `test_caller_context_crosses_the_bridge_and_changes_the_answer` |
| BL-14 | Runtime config is live | `config/config.yaml` | `max_retry` reaches the graph; `retrieval.*` reaches the inner state | `TestRuntimeConfigWiring` |

### Negative / boundary cases

| Case | Node | Expected |
|------|------|----------|
| empty `user_input` | PreProcessNode | `status=error`, "empty" |
| whitespace-only `user_input` | PreProcessNode | `status=error` |
| question > 4000 chars | PreProcessNode | `status=error`, "exceeds" |
| instruction-override payload (7 forms) | PreProcessNode | `status=error`, nothing carried forward, payload not echoed |
| ordinary clinical wording (6 forms) | PreProcessNode | `status=success` — no false positives |
| `channel` not `[a-z0-9_]{1,32}` | PreProcessNode | `status=error` naming the field; value not echoed |
| identifier-shaped tokens in the question | PreProcessNode | redacted before State is written |
| question too short (< 3 chars) | InputValidateNode | `status=error` |
| all-stopword question (no terms) | InputValidateNode | `status=error`, "term" |
| `patient_context` non-string / > 1000 chars | InputValidateNode | `status=error` naming the field |
| `top_k` non-finite matrix / 0 / 21 / 1e9 / 1.5 | InputValidateNode | `status=error` naming the field |
| `score_threshold` non-finite matrix / below the floor | InputValidateNode | `status=error` naming the field |
| `input_context` not an object | InputValidateNode | `status=error` |
| no query terms | RetrieveNode | empty candidate set, `status=success` |
| no candidates | RerankFilterNode | empty filtered set, `status=success` |
| empty `generated_answer` | OutputFormatNode | advisory fallback + disclaimer, `status=success` |
| empty `answer` | PostProcessNode | fallback message, `status=success` |
| credential / identifier leak in output (8 forms) | PostProcessNode | blocked, `status=error`, original content withheld |
| guideline citation keys (6 forms) | PostProcessNode | byte-identical passthrough |
| structural tokens (`90d`, years, bare counts, numbered headings) | PostProcessNode | byte-identical passthrough |
| disclaimer attachment broken | PostProcessNode | blocked rather than shipped (fail-closed) |

## 5. Retrieval Quality Thresholds

Retrieval and grounding quality is evaluated against the synthetic question set
before promotion — calibrated higher than a generic retrieval template because
the output is clinical:

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| `context_precision` | ≥ 0.80 | retrieved passages must be on topic before anything is grounded on them |
| `faithfulness` | ≥ 0.90 | answer claims must be supported by cited passages (no invented guidance) |
| `answer_relevancy` | ≥ 0.75 | the answer addresses the clinical question |
| abstention rate on out-of-corpus questions | 100% | any question with no passage at or above the floor MUST return the insufficient-evidence advisory |

## 6. Test Execution Summary

- Execution: `pytest tests/`.
- Total: **201 tests — 200 passed, 1 skipped** (the interrupt-propagation skip
  stub, by design), with zero warnings from this repository.
- Runner: pytest under the real framework wheel (the version central CI installs),
  not the CI stubs.
- Type check: `mypy src` — clean. Lint: `ruff check` and `ruff format --check` —
  both clean.
- Coverage: node and graph modules exercised on success, abstention, rejection
  and blocked-output paths, including the full end-to-end path through
  `POST /invoke`.
