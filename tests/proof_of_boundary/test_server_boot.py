# PB-BOOT — Server boot boundary: importing src.api.server must not raise.
#
# The standalone entry point (src/api/server.py) constructs the agent, calls
# compile(), and provisions secrets AT MODULE LEVEL — so an import failure is a
# deployment-time outage that node-level unit tests would never catch. The
# boundary proven here:
#
#   1. `import src.api.server` completes (the HTTP app and the compiled agent
#      are both built).
#   2. The module-level agent is THIS template's agent class, compiled, and
#      carries the runtime configuration declared in config/config.yaml — not
#      the framework defaults.
#   3. A fresh agent constructs and compiles via the supported path
#      (compile() -> register_nodes() -> graph build).
#
# Deterministic — no language model, no network, no socket bind (the app object
# is built but never served).

import importlib


class TestServerBoot:
    """PB-BOOT: the standalone HTTP entry point must import and compile."""

    def test_server_module_imports_without_raising(self):
        server = importlib.import_module("src.api.server")
        assert server.app is not None, "the HTTP app must be constructed at import"

    def test_module_level_agent_is_compiled(self):
        server = importlib.import_module("src.api.server")
        from src.graph.graph import ClinicalGuidelinesQAAgent

        assert isinstance(server.agent, ClinicalGuidelinesQAAgent)
        assert server.agent._compiled is not None, "server.py must compile() the agent at import time"

    def test_module_level_agent_carries_the_declared_runtime_config(self):
        """The declared values must be in force, not silently defaulted."""
        server = importlib.import_module("src.api.server")

        assert server.agent.config.get("max_retry") == 3
        assert server.agent.config.get("timeout_s") == 30
        assert server.agent.config.get("retrieval", {}).get("score_threshold") == 0.75

    def test_health_endpoint_reports_this_agent(self):
        server = importlib.import_module("src.api.server")
        payload = server.health()
        assert payload == {"status": "ok", "agent": "ClinicalGuidelinesQAAgent"}

    def test_fresh_agent_constructs_and_compiles(self):
        """The supported construction path: constructor -> compile()."""
        from src.graph.graph import ClinicalGuidelinesQAAgent

        agent = ClinicalGuidelinesQAAgent()
        agent.compile()
        assert agent._compiled is not None
        assert set(agent._nodes.keys()) == {
            "initialize",
            "pre_process",
            "main",
            "post_process",
            "finalize",
        }
