"""Contract tests — patch 2: feat(vision): endpoint-published capability lookup
+ capability-first auto routing (agent/image_routing.py).

Behavior under contract:
* OpenAI-compatible endpoints may publish ``supports_vision`` per model on
  ``/v1/models``; ``_lookup_endpoint_supports_vision`` reads it (strict bool
  coercion, TTL-cached per (base_url, model)) and ``_lookup_supports_vision``
  consults it AFTER the config override and BEFORE the static probes;
* ``decide_image_input_mode`` in ``auto`` is capability-first ONLY for an
  endpoint-published verdict: True -> native, False/unknown -> text — even over
  an explicit ``auxiliary.vision`` backend;
* any other verdict source keeps upstream ordering (explicit auxiliary.vision
  is the de-facto text route), and ``agent.image_input_mode: native`` remains
  the absolute override.

Integration boundary: real httpx GETs against a loopback OpenAI-compatible
``/v1/models`` stub; routing decisions assert the endpoint was consulted.
"""

from __future__ import annotations

import json

import pytest

from conftest import run_async  # (sys.path bootstrap side effect)


def _models_router(entries):
    """Router serving the OpenAI-compatible /v1/models listing."""
    def router(req):
        if req.path.rstrip("/").endswith("/models"):
            return 200, "application/json", json.dumps({"data": entries})
        return 404, "application/json", json.dumps({"error": "not found"})

    return router


def _cfg_for(svc, *, aux_vision=False, image_input_mode=None, supports_vision=None):
    cfg = {"model": {"provider": "custom", "base_url": svc.base_url.rstrip("/") + "/v1"}}
    if supports_vision is not None:
        cfg["model"]["supports_vision"] = supports_vision
    if aux_vision:
        cfg["auxiliary"] = {"vision": {"provider": "openai", "model": "gpt-4o-mini"}}
    if image_input_mode:
        cfg["agent"] = {"image_input_mode": image_input_mode}
    return cfg


def _sentinel_probe(calls, verdict):
    def probe(*args, **kwargs):
        calls.append(args)
        return verdict

    return probe


# ── Integration: routing consults the endpoint's /v1/models ─────────────────

def test_endpoint_published_true_beats_explicit_aux_vision(stub_http_server, monkeypatch):
    """Capability-first: an endpoint-published ``supports_vision: true`` routes
    native even when the user configured an auxiliary.vision backend."""
    # Upstream guardrails belong to this patch-sensitive scenario.
    _check_native_mode_is_absolute_override(stub_http_server, monkeypatch)
    _check_text_mode_short_circuits_without_consulting_endpoint(stub_http_server)
    import agent.image_routing as ir

    svc = stub_http_server(_models_router(
        [{"id": "alias-fast", "root": "mimo-pro", "supports_vision": True}]))
    calls = []
    monkeypatch.setattr(ir, "_VISION_PROBES", (("sentinel", _sentinel_probe(calls, True)),))

    mode = ir.decide_image_input_mode(
        "custom", "alias-fast", _cfg_for(svc, aux_vision=True))

    assert calls == []
    assert mode == "native"
    assert any(r.path.endswith("/v1/models") for r in svc.requests), \
        "decide_image_input_mode never consulted the endpoint listing"


def test_endpoint_published_false_routes_to_text(stub_http_server, monkeypatch):
    """Endpoint-published ``supports_vision: false`` routes text — the alias swap
    fix: the endpoint's own listing is the per-model truth."""
    import agent.image_routing as ir

    svc = stub_http_server(_models_router(
        [{"id": "alias-fast", "supports_vision": False}]))
    calls = []
    monkeypatch.setattr(ir, "_VISION_PROBES", (("sentinel", _sentinel_probe(calls, True)),))

    mode = ir.decide_image_input_mode("custom", "alias-fast", _cfg_for(svc))

    assert calls == []
    assert mode == "text"
    assert any(r.path.endswith("/v1/models") for r in svc.requests)


def test_unpublished_field_keeps_upstream_aux_pin(stub_http_server, monkeypatch):
    """No published field -> NOT capability-first: the upstream pin holds and an
    explicit auxiliary.vision backend stays the de-facto text route even when a
    static probe claims the main model has vision."""
    # Upstream guardrails belong to this patch-sensitive scenario.
    _check_static_verdict_decides_when_endpoint_unpublished(stub_http_server, monkeypatch)
    import agent.image_routing as ir

    svc = stub_http_server(_models_router([{"id": "alias-fast"}]))
    calls = []
    monkeypatch.setattr(
        ir, "_VISION_PROBES", (("sentinel", _sentinel_probe(calls, True)),))

    mode = ir.decide_image_input_mode(
        "custom", "alias-fast", _cfg_for(svc, aux_vision=True))

    assert mode == "text"
    assert calls, "static verdict source was not consulted after an unpublished field"


def _check_static_verdict_decides_when_endpoint_unpublished(stub_http_server, monkeypatch):
    """Without a published field and without an aux backend, the static verdict
    keeps upstream semantics: True -> native, unknown -> text."""
    import agent.image_routing as ir

    svc = stub_http_server(_models_router([{"id": "alias-fast"}]))
    calls = []
    monkeypatch.setattr(
        ir, "_VISION_PROBES", (("sentinel", _sentinel_probe(calls, True)),))

    assert ir.decide_image_input_mode("custom", "alias-fast", _cfg_for(svc)) == "native"

    monkeypatch.setattr(
        ir, "_VISION_PROBES", (("sentinel", _sentinel_probe(calls, None)),))
    assert ir.decide_image_input_mode("custom", "alias-fast", _cfg_for(svc)) == "text"


def test_endpoint_models_listing_is_cached(stub_http_server, monkeypatch):
    """One GET per TTL per endpoint: repeated image turns must not re-fetch."""
    import agent.image_routing as ir

    svc = stub_http_server(_models_router(
        [{"id": "alias-fast", "supports_vision": True}]))
    calls = []
    monkeypatch.setattr(ir, "_VISION_PROBES", (("sentinel", _sentinel_probe(calls, True)),))
    cfg = _cfg_for(svc)

    assert ir.decide_image_input_mode("custom", "alias-fast", cfg) == "native"
    assert ir.decide_image_input_mode("custom", "alias-fast", cfg) == "native"
    assert ir.decide_image_input_mode("custom", "alias-fast", cfg) == "native"

    listing_fetches = [r for r in svc.requests if r.path.endswith("/v1/models")]
    assert len(listing_fetches) == 1, f"expected 1 listing fetch, got {len(listing_fetches)}"


# ── Unit: lookup + coercion + absolute override ─────────────────────────────

def test_lookup_endpoint_supports_vision_variants(stub_http_server):
    """_lookup_endpoint_supports_vision: published bools pass through, a listed
    model without the field and an unlisted model both return None (fall
    through)."""
    import agent.image_routing as ir

    svc = stub_http_server(_models_router([
        {"id": "Yes-Vision", "supports_vision": True},
        {"id": "no-vision", "supports_vision": False},
        {"id": "unspecified"},
    ]))
    base = svc.base_url.rstrip("/") + "/v1"

    assert ir._lookup_endpoint_supports_vision(base, "yes-vision", "") is True
    assert ir._lookup_endpoint_supports_vision(base, "NO-VISION", "") is False
    assert ir._lookup_endpoint_supports_vision(base, "unspecified", "") is None
    assert ir._lookup_endpoint_supports_vision(base, "not-listed", "") is None
    assert ir._lookup_endpoint_supports_vision("", "yes-vision", "") is None


def test_published_types_route_without_truthiness_coercion(stub_http_server, monkeypatch):
    """Only published booleans override the fallback; integers/objects do not."""
    import agent.image_routing as ir
    values = [True, False, None, 1, {"nested": True}]
    svc = stub_http_server(_models_router([
        {"id": f"opaque-{i}", "supports_vision": value}
        for i, value in enumerate(values)]))
    calls = []
    monkeypatch.setattr(ir, "_VISION_PROBES", (("fallback", _sentinel_probe(calls, False)),))
    for i, value in enumerate(values):
        calls.clear()
        mode = ir.decide_image_input_mode("custom", f"opaque-{i}", _cfg_for(svc))
        assert mode == ("native" if value is True else "text")
        assert bool(calls) == (not isinstance(value, bool))


def _check_native_mode_is_absolute_override(stub_http_server, monkeypatch):
    """agent.image_input_mode: native wins over everything, without any lookup."""
    import agent.image_routing as ir

    svc = stub_http_server(_models_router(
        [{"id": "alias-fast", "supports_vision": False}]))
    calls = []
    monkeypatch.setattr(ir, "_VISION_PROBES", (("sentinel", _sentinel_probe(calls, True)),))

    mode = ir.decide_image_input_mode(
        "custom", "alias-fast", _cfg_for(svc, aux_vision=True, image_input_mode="native"))
    assert calls == []
    assert mode == "native"
    assert svc.requests == [], "explicit mode must decide without network"


def _check_text_mode_short_circuits_without_consulting_endpoint(stub_http_server):
    """agent.image_input_mode: text decides up front; the endpoint is not asked."""
    import agent.image_routing as ir

    svc = stub_http_server(_models_router(
        [{"id": "alias-fast", "supports_vision": True}]))

    mode = ir.decide_image_input_mode(
        "custom", "alias-fast", _cfg_for(svc, image_input_mode="text"))
    assert mode == "text"
    assert svc.requests == []

def test_alias_refresh_and_cache_keys(stub_http_server, monkeypatch):
    import time
    import agent.image_routing as ir
    entries = [{"id": "opaque", "supports_vision": True},
               {"id": "other", "supports_vision": False}]
    svc = stub_http_server(_models_router(entries))
    second = stub_http_server(_models_router([{"id": "opaque", "supports_vision": False}]))
    clock = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: clock)
    calls = []
    monkeypatch.setattr(ir, "_VISION_PROBES", (("conflict", _sentinel_probe(calls, True)),))
    assert ir.decide_image_input_mode("custom", "opaque", _cfg_for(svc)) == "native"
    assert ir.decide_image_input_mode("custom", "other", _cfg_for(svc)) == "text"
    assert ir.decide_image_input_mode("custom", "opaque", _cfg_for(second)) == "text"
    entries[0]["supports_vision"] = False
    # The operator-facing contract is a short cache: allow ten minutes, without
    # depending on an internal TTL constant or cache representation.
    clock += 601
    assert ir.decide_image_input_mode("custom", "opaque", _cfg_for(svc)) == "text"
    assert calls == []


def test_config_overrides_endpoint(stub_http_server):
    import agent.image_routing as ir
    svc = stub_http_server(_models_router([
        {"id": "yes", "supports_vision": True},
        {"id": "no", "supports_vision": False}]))
    assert ir.decide_image_input_mode("custom", "yes", _cfg_for(svc)) == "native"
    assert ir.decide_image_input_mode("custom", "no", _cfg_for(svc)) == "text"
    assert ir.decide_image_input_mode("custom", "yes", _cfg_for(svc, supports_vision=False)) == "text"
    assert ir.decide_image_input_mode("custom", "no", _cfg_for(svc, supports_vision=True)) == "native"


@pytest.mark.parametrize("published, expected", [("true", "native"), ("false", "text")])
def test_published_boolean_strings(stub_http_server, monkeypatch, published, expected):
    """Known production regression: missing token sets must fail, never xfail."""
    import agent.image_routing as ir
    svc = stub_http_server(_models_router([{"id": "opaque", "supports_vision": published}]))
    calls = []
    monkeypatch.setattr(ir, "_VISION_PROBES",
                        (("conflict", _sentinel_probe(calls, expected != "native")),))
    assert ir.decide_image_input_mode("custom", "opaque", _cfg_for(svc)) == expected
    assert calls == []
