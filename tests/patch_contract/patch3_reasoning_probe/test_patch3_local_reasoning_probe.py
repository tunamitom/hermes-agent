"""Contract tests — patch 3: fix(gateway): unified local-endpoint probe for
MiMo/Kimi/DeepSeek/Qwen reasoning detection (agent/message_sanitization.py +
agent/reasoning_params.py).

Behavior under contract:
* ``probe_local_reasoning_families(base_url)`` probes local backends only
  (loopback/LAN/.ts.net hosts): strategy 1 sglang ``/get_model_info`` ->
  ``model_type`` substring; strategy 2 ``/v1/models`` -> model id / alias
  ``root`` substring; returns family -> bool, ``{}`` for non-local endpoints;
* confirmed results cache for the process; an all-false (failed) probe caches
  briefly so a router hiccup never pins
  detection off;
* ``stale_thinking_reaches_wire`` and ``ReasoningParamsMixin``
  (``_needs_thinking_reasoning_pad`` / ``_needs_{mimo,kimi,deepseek,qwen}_tool_reasoning``,
  probe-only for qwen) fall back to the probe when the rule table misses an
  alias route.

Integration boundary: real urllib requests against a loopback sglang/router
stub serving ``/get_model_info`` and ``/v1/models``.
"""

from __future__ import annotations

import json

import pytest


def _ollama_router(models):
    """Stub OLLAMA endpoint: native /api/version plus an OpenAI-style /v1/models."""
    def router(req):
        path = req.path.rstrip("/")
        if path.endswith("/api/version"):
            return 200, "application/json", json.dumps({"version": "0.9.1"})
        if path.endswith("/models"):
            return 200, "application/json", json.dumps(models)
        return 404, "application/json", json.dumps({"error": "not found"})
    return router


def _router(get_model_info, models, fail_paths=()):
    """Stub LLM endpoint: get_model_info JSON and /v1/models listing.

    ``get_model_info`` / ``models`` may be None (-> 404) or a JSON payload;
    ``fail_paths`` lists request-path suffixes answered with 500.
    """
    def router(req):
        path = req.path.rstrip("/")
        if any(path.endswith(f) for f in fail_paths):
            return 500, "application/json", json.dumps({"error": "boom"})
        if path.endswith("/get_model_info"):
            if get_model_info is None:
                return 404, "application/json", json.dumps({"error": "not found"})
            return 200, "application/json", json.dumps(get_model_info)
        if path.endswith("/models"):
            if models is None:
                return 404, "application/json", json.dumps({"error": "not found"})
            return 200, "application/json", json.dumps(models)
        return 404, "application/json", json.dumps({"error": "not found"})

    return router


# ── Integration: family detection over real HTTP ────────────────────────────

def test_sglang_model_info_detects_family(stub_http_server):
    """Strategy 1: sglang /get_model_info ``model_type`` substring detection."""
    from agent import message_sanitization as ms

    svc = stub_http_server(_router({"model_type": "mimo-pro"}, {"data": []}))
    url = svc.base_url.rstrip("/") + "/v1"

    detected = ms.probe_local_reasoning_families(url)

    assert detected == {"mimo": True, "kimi": False, "deepseek": False, "qwen": False}
    assert any(r.path.endswith("/get_model_info") for r in svc.requests)


def test_alias_router_models_root_detects_family(stub_http_server):
    """Strategy 2: an alias router (id "fast") resolves the family through the
    alias ``root`` when /get_model_info is absent."""
    from agent import message_sanitization as ms

    svc = stub_http_server(_router(
        None,
        {"data": [
            {"id": "fast", "root": "kimi-k2", "_alias": True},
        ]}))
    url = svc.base_url.rstrip("/") + "/v1"

    detected = ms.probe_local_reasoning_families(url)

    assert detected["kimi"] is True
    assert detected["mimo"] is False


def test_deepseek_and_qwen_detection(stub_http_server):
    """All four families are recognized from either strategy."""
    from agent import message_sanitization as ms

    svc = stub_http_server(_router({"model_type": "deepseek-v3.2"}, None))
    url = svc.base_url.rstrip("/") + "/v1"
    assert ms.probe_local_reasoning_families(url)["deepseek"] is True

    svc2 = stub_http_server(_router(
        None, {"data": [{"id": "qwen3-235b-a22b", "root": "qwen3-235b-a22b"}]}))
    url2 = svc2.base_url.rstrip("/") + "/v1"
    assert ms.probe_local_reasoning_families(url2)["qwen"] is True


def test_non_local_endpoint_is_never_probed(monkeypatch):
    """Remote endpoints return {} and generate no traffic."""
    from agent import message_sanitization as ms

    calls = []
    transport = lambda *a, **kw: calls.append(a)

    for remote in ("https://api.openai.com/v1", "https://openrouter.ai/api/v1",
                   "http://example.com:8000/v1"):
        assert ms.probe_local_reasoning_families(remote, _urlopen=transport) == {}

    assert calls == [], "remote URLs must never reach the transport"


def test_confirmed_result_is_cached(stub_http_server):
    """A confirmed probe answer serves from cache: one round of HTTP per URL."""
    from agent import message_sanitization as ms

    svc = stub_http_server(_router({"model_type": "mimo-pro"}, {"data": []}))
    url = svc.base_url.rstrip("/") + "/v1"

    first = ms.probe_local_reasoning_families(url)
    hits = len(svc.requests)
    second = ms.probe_local_reasoning_families(url)

    assert first == second
    assert len(svc.requests) == hits, "cached probe re-hit the endpoint"


def test_failed_probe_retries_after_cooldown(stub_http_server, monkeypatch):
    """An all-false (endpoint down) probe must NOT pin detection off: after the
    cooldown the probe re-runs and recovers."""
    from agent import message_sanitization as ms

    import time
    clock = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: clock)
    svc = stub_http_server(_router(
        {"model_type": "mimo-pro"}, {"data": []},
        fail_paths=("/get_model_info", "/models")))
    url = svc.base_url.rstrip("/") + "/v1"

    down = ms.probe_local_reasoning_families(url)
    assert down["mimo"] is False and not any(down.values())

    svc.server.router = _router({"model_type": "mimo-pro"}, {"data": []})  # endpoint back up
    clock += 601
    recovered = ms.probe_local_reasoning_families(url)
    assert recovered["mimo"] is True, "failed probe pinned detection off"


# ── Integration: the pad/wire predicates key off the probe ──────────────────

def test_stale_thinking_reaches_wire_falls_back_to_probe(stub_http_server):
    """On an alias route the rule table misses (model "fast"), the wire-truth
    predicate must answer via the local probe."""
    from agent import message_sanitization as ms

    svc = stub_http_server(_router(
        None, {"data": [{"id": "fast", "root": "mimo-pro", "_alias": True}]}))
    url = svc.base_url.rstrip("/") + "/v1"

    assert ms.stale_thinking_reaches_wire("chat_completions", "custom", "fast", url) is True
    # No api_mode (bare compressor) branch falls back to the probe too.
    assert ms.stale_thinking_reaches_wire(None, "custom", "fast", url) is True


def test_mixin_pad_and_family_detection_key_off_probe(stub_http_server):
    """ReasoningParamsMixin: pad fallback + per-family detection resolve through
    the probe on alias routes; qwen is probe-only by design."""
    from agent.reasoning_params import ReasoningParamsMixin

    class FakeAgent(ReasoningParamsMixin):
        pass

    svc = stub_http_server(_router(
        None, {"data": [{"id": "fast", "root": "mimo-pro", "_alias": True}]}))
    url = svc.base_url.rstrip("/") + "/v1"

    agent = FakeAgent()
    agent.provider = "custom"
    agent.model = "fast"  # defeats every rule-table substring
    agent.base_url = url
    agent.api_mode = "chat_completions"
    agent.session_id = None

    assert agent._needs_thinking_reasoning_pad() is True
    assert agent._needs_mimo_tool_reasoning() is True
    assert agent._needs_kimi_tool_reasoning() is False
    assert agent._needs_deepseek_tool_reasoning() is False
    assert agent._probe_local_for_reasoning_model("mimo") is True


def test_qwen_needs_reasoning_is_probe_only(stub_http_server):
    """Qwen has no echo rule (a name substring would pad ollama-served qwen):
    detection is probe-only — negative probe, negative verdict, even for a
    qwen-looking model id."""
    from agent.reasoning_params import ReasoningParamsMixin

    class FakeAgent(ReasoningParamsMixin):
        pass

    svc = stub_http_server(_router(None, {"data": [{"id": "qwen3:8b", "root": "qwen3:8b"}]}))
    url = svc.base_url.rstrip("/") + "/v1"

    agent = FakeAgent()
    agent.provider = "custom"
    agent.model = "qwen3-8b"  # name alone must never trigger padding
    agent.base_url = "http://localhost:1/v1"  # unreachable: probe says no
    agent.api_mode = "chat_completions"
    agent.session_id = None

    assert agent._needs_qwen_tool_reasoning() is False

    agent.base_url = url  # endpoint actually serving qwen
    assert agent._needs_qwen_tool_reasoning() is True


def test_ollama_endpoint_is_never_padded(stub_http_server):
    """Upstream contract: ollama models are replayed as reasoning but NEVER padded.

    The endpoint identifies as Ollama via /api/version, so even a qwen-looking listing
    stays family-less — neither the qwen predicate nor the pad path may fire. This is
    the guard for the "ollama qwen must never be padded" claim.
    """
    from agent import message_sanitization as ms
    from agent.reasoning_params import ReasoningParamsMixin

    class FakeAgent(ReasoningParamsMixin):
        pass

    svc = stub_http_server(_ollama_router(
        {"data": [{"id": "qwen3:8b", "root": "qwen3:8b"}]}))
    url = svc.base_url.rstrip("/") + "/v1"

    assert ms.probe_local_reasoning_families(url) == {
        "mimo": False, "kimi": False, "deepseek": False, "qwen": False}

    agent = FakeAgent()
    agent.provider = "custom"
    agent.model = "qwen3:8b"
    agent.base_url = url
    agent.api_mode = "chat_completions"
    agent.session_id = None

    assert agent._needs_qwen_tool_reasoning() is False
    assert agent._needs_thinking_reasoning_pad() is False
