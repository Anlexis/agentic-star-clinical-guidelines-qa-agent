# Clinical Guidelines Q&A Agent

AI agent for answering questions about clinical guidelines, built with Agentic Star.

> **Category**: Cat 2 (a domain pipeline: retrieve clinical guideline passages, then answer from them)
> **Industry**: Healthcare
> **Template ID**: HCR-C2-011

## Overview

A question-answering agent for licensed healthcare professionals. It takes a clinical question
("what is the initial management of adult hypertension?"), retrieves the matching passages from a
clinical-guidelines knowledge base, and returns an **advisory-only** answer that cites the passages
it was grounded in.

Three properties are enforced rather than merely documented, because the domain is life-adjacent:

- **It abstains rather than guesses.** Passages below the configured confidence threshold (0.75 by
  default) are dropped, and a question with no confident match returns an explicit
  *insufficient evidence* response instead of an unsupported answer.
- **It only says what a retrieved passage says.** The answer is composed from the filtered passages
  alone; nothing is generated beyond them.
- **Every answer carries the advisory disclaimer.** The output gate verifies its presence and
  blocks the response if it is missing, so no clinical text can ship without it.

The bundled knowledge base is a small **synthetic** exemplar corpus — it contains no patient records
and no personal health information, and exists so the pipeline runs end to end out of the box.
Replace it with your own guideline sources.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >=3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode: the graph's start-up path constructs the agent, validates its configuration and binds a
secret provider against the platform, and a failure at any of those steps raises rather than
leaving a partially wired agent serving requests. This is intentional — a half-running agent is
worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Calling the agent

`POST /invoke` takes the clinical question as `input`. Structured parameters are optional and ride
in `input_context`; every one of them is validated against explicit bounds, and an out-of-range or
non-finite value is rejected outright rather than silently ignored.

```json
{
  "input": "What is the initial management of adult hypertension and blood pressure?",
  "input_context": {
    "patient_context": "adult, newly diagnosed, no compelling indications",
    "top_k": 5,
    "score_threshold": 0.80,
    "channel": "ward_console"
  }
}
```

| Field | Type | Bounds |
|---|---|---|
| `patient_context` | string | ≤ 1000 characters; direct-identifier-shaped tokens are redacted before use. Its terms widen retrieval; it is never rendered into the answer. |
| `top_k` | number | whole number, 1–20 |
| `score_threshold` | number | finite, between the configured floor and 1.0 — a caller may make retrieval **stricter**, never looser |
| `channel` | string | inert label matching `[a-z0-9_]{1,32}` |

Omitting `input_context` entirely is valid: the agent falls back to the values declared in
`config/config.yaml`.

## Configuration

| File | Purpose |
|---|---|
| `config/agent.yaml` | The static manifest — identity, category, entry-point class, required trust level, and the secrets/extras the agent needs at compile time. Flat: every key is read at root level. |
| `config/config.yaml` | Runtime parameters — `max_retry`, `timeout_s`, and the `retrieval` / `llm` / `security` blocks. Passed to the graph constructor. |

## Project Structure

```
src/          agent implementation (nodes, graphs, schemas, HTTP adapter)
tests/        unit, integration and boundary tests
config/       agent manifest and runtime configuration
prompts/      answer-generation prompt template
docs/         design specification and test specification
```

`docs/02_design.md` describes the architecture and the security configuration; `docs/03_test_spec.md`
describes what is tested and why.

## Customising

1. Adjust `config/config.yaml` for your own environment and confidence thresholds.
2. Replace the synthetic guideline corpus in `src/nodes/retrieve_node.py` with your own knowledge
   source, keeping the same passage shape.
3. Review the node implementations under `src/nodes/` for domain-specific logic.
4. Re-run the test suite.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
