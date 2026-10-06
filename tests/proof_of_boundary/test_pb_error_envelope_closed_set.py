# PB — The ERROR envelope is a closed set at the boundary the caller sees:
# POST /invoke through the real ASGI interface (src/api/server.py).
#
# Every non-success body carries `error: {"reason": <constant>}`, `output: null`
# and NO `error_log` key. error_log is node-authored text and stays internal.
#
# The fault is injected on the DATA path, never on the envelope: a sentinel — a
# name, an e-mail and a token-shaped fragment, the shape of an upstream response
# body an API client quotes into its error — is seeded into error_log by the
# nodes that write it in production:
#   - the retrieval transport fails and quotes the upstream body (inner node
#     returns status=error; the message crosses the subgraph boundary through
#     on_subgraph_error() onto the outer error_log);
#   - the retrieval transport RAISES with that message (the framework writes
#     "[RetrieveNode] <message>" plus a full traceback into the inner log);
#   - an OUTER node raises with it (the framework writes the same shape straight
#     into outer state);
#   - the outer PreProcessNode records it as a warning on a run whose answer the
#     output gate then blocks (the gate-block path, with the sentinel already in
#     the log when the gate's own message is appended).
# On every path the sentinel must appear nowhere in the body — nested keys and
# values walked — and the body must carry the closed-set envelope only.
#
# The clean-path control sits beside them: the same request over the untouched
# pipeline still returns the grounded answer, with no `error` key at all.

import asyncio
import json

import pytest

from framework.schemas.agent_status import AgentStatus

from src.api import server as server_module  # noqa: F401  (import = boot check)
from src.api.server import app
from src.nodes.post_process_node import _REASON_OUTPUT_WITHHELD, _REASON_WORKFLOW_FAILED, ERROR_REASONS

_TOKEN = "pb-closed-set-token"

# Grounds on GL-HTN-001 at a lexical score of ~0.83, above the 0.75 floor.
_HYPERTENSION_QUESTION = "What is the initial management of adult hypertension and blood pressure?"

# Assembled at runtime (never a committed literal): a name, an e-mail and a
# token-shaped fragment. Deliberately NOT credential-shaped and free of any
# trace fragment, so a redaction-based surface passes it straight through.
_FRAGMENTS = ("A. Tanaka", "a.tanaka@example.com", "sk-" + "live-xxx")
_SENTINEL = (
    "upstream said {'patient':'" + _FRAGMENTS[0] + "','email':'" + _FRAGMENTS[1] + "','token':'" + _FRAGMENTS[2] + "'}"
)
_MARKERS = (_SENTINEL, "upstream said", *_FRAGMENTS)
_NODE_AUTHORED = ("RetrieveNode:", "PreProcessNode:", "knowledge base call failed", "Traceback", ".py", "/Users/")


def _post(path: str, payload: dict) -> tuple[int, dict]:
    """POST through the real ASGI app with a bearer token."""
    body = json.dumps(payload).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"authorization", f"Bearer {_TOKEN}".encode()),
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }

    messages: list = []
    sent = {"body": b""}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    return start["status"], json.loads(sent["body"].decode() or "{}")


@pytest.fixture(autouse=True)
def token_configured(monkeypatch):
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)


@pytest.fixture(autouse=True)
def quiet_audit(monkeypatch):
    for mod in (
        "pre_process_node",
        "input_validate_node",
        "retrieve_node",
        "rerank_filter_node",
        "generate_answer_node",
        "output_format_node",
        "post_process_node",
    ):
        monkeypatch.setattr(f"src.nodes.{mod}.emit_trace_event", lambda *a, **k: None)


def _invoke(question: str = _HYPERTENSION_QUESTION) -> dict:
    status_code, body = _post("/invoke", {"input": question, "session_id": "pb-closed-set", "input_context": {}})
    assert status_code == 200, f"expected 200, got {status_code}: {body}"
    return body


def _strings(value):
    """Every string reachable in value: dict keys and values, list items, and
    the repr of anything else that is not a plain scalar."""
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
    texts = list(_strings(mapping))
    return [needle for needle in needles if any(needle in text for text in texts)]


# ── Fault injection on the data path ──────────────────────────────────────────


def _retrieval_returns_the_upstream_body(monkeypatch) -> None:
    """The retrieval transport fails and quotes the upstream response body into
    its error_log line — as an API client that interpolates the response does."""
    from src.nodes import retrieve_node

    def execute(self, state):
        return {
            "status": AgentStatus.ERROR.value,
            "error_log": [f"RetrieveNode: knowledge base call failed: {_SENTINEL}"],
        }

    monkeypatch.setattr(retrieve_node.RetrieveNode, "execute", execute)


def _retrieval_raises_with_the_upstream_body(monkeypatch) -> None:
    """The retrieval transport raises; the framework's node wrapper writes
    "[RetrieveNode] <message>" plus a full traceback into the inner error_log."""
    from src.nodes import retrieve_node

    def execute(self, state):
        raise RuntimeError(f"knowledge base call failed: {_SENTINEL}")

    monkeypatch.setattr(retrieve_node.RetrieveNode, "execute", execute)


def _outer_node_raises_with_the_upstream_body(monkeypatch) -> None:
    """An OUTER node raises: the same framework entry lands straight in outer state."""
    from src.nodes import pre_process_node

    def execute(self, state):
        raise RuntimeError(f"screening service call failed: {_SENTINEL}")

    monkeypatch.setattr(pre_process_node.PreProcessNode, "execute", execute)


def _sentinel_logged_then_gate_blocks(monkeypatch) -> None:
    """The outer PreProcessNode records the sentinel as a warning line on a
    SUCCESS return (the reducer appends it to the outer error_log), and the
    retrieval corpus carries a disallowed value so the output gate blocks: the
    gate-block path, with the sentinel already in the log when the gate's own
    message is appended."""
    from src.nodes import pre_process_node, retrieve_node

    original = pre_process_node.PreProcessNode.execute

    def execute(self, state):
        result = original(self, state)
        return {**result, "error_log": [f"PreProcessNode: upstream warning: {_SENTINEL}"]}

    monkeypatch.setattr(pre_process_node.PreProcessNode, "execute", execute)

    corpus = [dict(entry) for entry in retrieve_node._CORPUS]
    corpus[0] = dict(corpus[0])
    corpus[0]["text"] = f"{corpus[0]['text']} Contact MRN-1234567"
    monkeypatch.setattr(retrieve_node, "_CORPUS", corpus)


_ERROR_PATHS = [
    pytest.param(_retrieval_returns_the_upstream_body, _REASON_WORKFLOW_FAILED, id="inner-node-returns-error"),
    pytest.param(_retrieval_raises_with_the_upstream_body, _REASON_WORKFLOW_FAILED, id="inner-node-raises"),
    pytest.param(_outer_node_raises_with_the_upstream_body, _REASON_WORKFLOW_FAILED, id="outer-node-raises"),
    pytest.param(_sentinel_logged_then_gate_blocks, _REASON_OUTPUT_WITHHELD, id="gate-block-with-seeded-log"),
]


class TestErrorEnvelopeIsClosedSetOverInvoke:
    @pytest.mark.parametrize("inject, reason", _ERROR_PATHS)
    def test_body_carries_the_closed_set_envelope_only(self, inject, reason, monkeypatch):
        inject(monkeypatch)
        body = _invoke()

        assert body["status"] == "error", body
        assert set(body["error"]) == {"reason"}
        assert body["error"]["reason"] in ERROR_REASONS
        assert body["error"] == {"reason": reason}
        assert body["output"] is None
        assert body["formatted_output"] is None
        for key in ("result", "answer", "generated_answer", "citations", "filtered_passages"):
            assert body[key] is None, key

    @pytest.mark.parametrize("inject, reason", _ERROR_PATHS)
    def test_error_log_is_not_projected(self, inject, reason, monkeypatch):
        inject(monkeypatch)
        body = _invoke()

        assert body["status"] == "error", body
        assert "error_log" not in body

    @pytest.mark.parametrize("inject, reason", _ERROR_PATHS)
    def test_sentinel_appears_nowhere_in_the_body(self, inject, reason, monkeypatch):
        inject(monkeypatch)
        body = _invoke()

        assert body["status"] == "error", body
        assert _found(body, *_MARKERS) == [], body
        assert _found(body, *_NODE_AUTHORED) == [], body
        rendered = json.dumps(body)
        for marker in (*_MARKERS, *_NODE_AUTHORED):
            assert marker not in rendered

    def test_gate_block_happened_at_the_gate(self, monkeypatch):
        # The output_withheld reason is the gate's own: post_process ran and
        # refused. Without this the gate-block row could pass on a request that
        # failed upstream and never reached the gate.
        _sentinel_logged_then_gate_blocks(monkeypatch)
        body = _invoke()
        assert body["error"] == {"reason": _REASON_OUTPUT_WITHHELD}
        assert "PostProcessNode" in body["node_history"]
        assert "MRN-1234567" not in json.dumps(body)


class TestCleanPathControl:
    def test_the_same_request_still_returns_the_grounded_answer(self):
        body = _invoke()
        assert body["status"] == "success", body
        assert "GL-HTN-001" in body["output"]
        assert "error" not in body
        assert "error_log" not in body

    def test_the_probe_finds_the_sentinel_where_it_lives(self):
        # Verify the verifier: the same walk DOES find every marker in a body
        # that carries it, so the "nowhere" assertions above are not vacuous.
        leaky = {"status": "error", "error_log": [f"RetrieveNode: knowledge base call failed: {_SENTINEL}"]}
        assert set(_found(leaky, *_MARKERS)) == set(_MARKERS)
        assert _found({"deep": [{"key " + _SENTINEL: None}]}, _SENTINEL) == [_SENTINEL]
