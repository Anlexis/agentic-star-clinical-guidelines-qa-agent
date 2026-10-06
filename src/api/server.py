"""AgentCore Platform v1.0"""

# Standalone HTTP entry point for the agent.
# Entry points are adapters only — no business logic here.
# For platform-level routing, the gateway calls agent.invoke() directly.

import json
import os
import secrets
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from framework.secrets.context import bound_secrets
from shared.secrets import factory as secrets_factory
from src.graph.graph import Graph, load_runtime_config

app = FastAPI(title="Agent")

# Runtime parameters come from config/config.yaml — max_retry and timeout_s are
# read and validated by the graph itself, so they must be handed to the
# constructor rather than left to the framework defaults.
agent = Graph(config=load_runtime_config())
agent.compile()
# The namespace matches the `namespace` field of config/agent.yaml.
agent.provision_secrets(secrets_factory(namespace="hcr", agent_name="ClinicalGuidelinesQAAgent"))

# Upper bound on the serialized input_context (bytes). The domain intake node
# additionally validates every field individually; this is the transport-level
# cap that keeps an oversized body from reaching the graph at all.
_MAX_INPUT_CONTEXT_BYTES = 256 * 1024


class InvokeRequest(BaseModel):
    input: str
    session_id: str = ""
    # Structured invocation parameters. Validated field-by-field by the domain
    # intake node (patient_context / top_k / score_threshold / channel);
    # anything absent falls back to the configured defaults.
    input_context: Optional[dict[str, Any]] = None


@app.post("/invoke")
async def invoke(req: InvokeRequest, request: Request) -> dict[str, Any]:
    trust = getattr(request.state, "trust_level", TrustLevel.ANONYMOUS)
    # Standalone caller authentication: when INVOKE_AUTH_TOKEN is set on the
    # server environment, callers that no upstream middleware vouched for (still
    # ANONYMOUS) must present it as a bearer token and then run at
    # VERIFIED_EXTERNAL. Middleware-established trust is never demoted.
    # This adapter is the entry-point auth boundary — a deployment-level caller
    # credential, not an agent secret, so ctx.secrets does not apply (no
    # invocation context exists before authentication).
    expected = os.environ.get("INVOKE_AUTH_TOKEN")
    if expected and trust is TrustLevel.ANONYMOUS:
        supplied = request.headers.get("authorization", "")
        # Compare bytes: compare_digest raises TypeError on non-ASCII str input
        # (headers decode as latin-1), which would 500 instead of the generic 401.
        if not secrets.compare_digest(supplied.encode(), f"Bearer {expected}".encode()):
            # Generic body on purpose — do not leak whether the token was absent,
            # malformed, or wrong.
            raise HTTPException(status_code=401, detail="Token is invalid or expired.")
        trust = TrustLevel.VERIFIED_EXTERNAL

    input_context = req.input_context or {}
    if input_context and len(json.dumps(input_context, default=str)) > _MAX_INPUT_CONTEXT_BYTES:
        raise HTTPException(status_code=413, detail="input_context exceeds the maximum allowed size.")

    with bound_secrets(agent._secrets_provider):
        ctx = InvocationContext(
            session_id=req.session_id or str(uuid4()),
            caller_trust_level=trust,
            caller_id=getattr(request.state, "caller_id", ""),
        )
        result: dict[str, Any] = agent.invoke(req.input, ctx=ctx, input_context=input_context)
        return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "ClinicalGuidelinesQAAgent"}
