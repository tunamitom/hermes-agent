"""session_models.json file overrides must apply even without api_key.

The kanban app writes per-session model overrides (model/provider/base_url, no
api_key for custom providers) to ``<HERMES_HOME>/session_models.json``. The
gateway's resolver falls back to the file when no in-memory override exists —
but applying it via the in-memory-only path silently drops it after every
gateway restart (memory is empty; the file entry was read each turn and
discarded, so the turn resolved the config default instead).

Contract: a file-only override (no api_key) changes the resolved model AND
base_url for the turn, WITHOUT being written into in-memory session state —
the file is the live source of truth (kanban rewrites it on every selector
change) and a cached memory copy would shadow later file edits.
"""

import json

import pytest

import gateway.run as gw_run
import gateway.run_turn
from gateway.run import GatewayRunner


KEY = "agent:main:telegram:dm:123456789"


def _bare_runner() -> GatewayRunner:
    """GatewayRunner without __init__ (bare-object pattern used across gateway tests)."""
    return object.__new__(GatewayRunner)


def _write_override(tmp_path, entry: dict) -> None:
    (tmp_path / "session_models.json").write_text(json.dumps({KEY: entry}), encoding="utf-8")


@pytest.fixture
def resolver_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME + stubbed config/runtime resolution on the gateway.run seam."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gw_run, "_resolve_gateway_model", lambda user_config=None: "smart")
    monkeypatch.setattr(
        gw_run, "_resolve_runtime_agent_kwargs",
        lambda: {"model": "smart", "provider": "custom", "api_key": "EMPTY",
                 "base_url": "http://config-default/v1", "capabilities": {}, "args": []},
    )
    return tmp_path


def test_file_override_without_api_key_is_applied(resolver_env):
    """Kanban file entries carry no api_key — model+base_url must still apply to the turn."""
    _write_override(resolver_env, {
        "model": "fast", "provider": "custom", "base_url": "http://turin:8003/v1",
        "agentId": "main", "updatedAt": 0,
    })
    runner = _bare_runner()
    model, runtime = runner._resolve_session_agent_runtime(session_key=KEY)
    assert model == "fast"
    assert runtime["base_url"] == "http://turin:8003/v1"
    assert runtime["provider"] == "custom"


def test_file_override_does_not_shadow_in_memory_state(resolver_env):
    """File override is applied per-turn from the file, never cached into session state."""
    _write_override(resolver_env, {
        "model": "fast", "provider": "custom", "base_url": "http://turin:8003/v1",
        "agentId": "main", "updatedAt": 0,
    })
    runner = _bare_runner()
    runner._resolve_session_agent_runtime(session_key=KEY)
    state = runner._peek_session_state(KEY)
    assert state is None or state.conversation.model_override is None
