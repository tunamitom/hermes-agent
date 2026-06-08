"""Local reasoning-family probe on alias routes.

Two invariants for ``probe_local_reasoning_families``:
- an alias route (router publishes e.g. ``fast`` -> ``mimo-pro``) detects the family
  through the alias ``root``, and the compaction estimators' wire-truth predicate
  agrees with the pad path (else the trigger and the tail walk disagree);
- a failed probe (endpoint hiccup) is retried after a cooldown instead of pinning
  detection off for the process lifetime.
"""
import json
import urllib.error

import pytest

from agent import message_sanitization as ms


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


class _FakeUrlopen:
    """/get_model_info 404s (non-sglang router); /v1/models serves queued payloads."""

    def __init__(self, payloads, fail_first=0):
        self.calls = 0
        self._payloads = list(payloads)
        self._fail_first = fail_first

    def __call__(self, req, timeout=None):
        self.calls += 1
        if self._fail_first:
            self._fail_first -= 1
            raise urllib.error.URLError("endpoint restarting")
        if req.full_url.endswith("/get_model_info"):
            raise urllib.error.HTTPError(req.full_url, 404, "not found", None, None)
        return _Resp(self._payloads.pop(0) if self._payloads else {"data": []})


_ROUTER_MODELS = {
    "data": [
        {"id": "mimo-pro", "root": "/model"},
        {"id": "fast", "root": "mimo-pro", "_alias": True},
    ]
}


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    ms._reasoning_probe_cache.clear()
    yield
    ms._reasoning_probe_cache.clear()


def test_alias_route_detects_family_and_estimator_agrees_with_wire():
    fake = _FakeUrlopen([_ROUTER_MODELS])
    url = "http://127.0.0.1:8003/v1"
    detected = ms.probe_local_reasoning_families(url, _urlopen=fake)
    assert detected["mimo"] is True  # session id "fast" resolves through the alias root
    assert ms.needs_reasoning_echo("custom", "fast", url) is False  # the table misses it
    assert ms.stale_thinking_reaches_wire("chat_completions", "custom", "fast", url) is True


def test_failed_probe_retries_instead_of_pinning_negative(monkeypatch):
    monkeypatch.setattr(ms, "_REASONING_PROBE_RETRY_S", 0.0)
    fake = _FakeUrlopen([_ROUTER_MODELS], fail_first=2)  # both strategies down, once
    url = "http://127.0.0.1:8004/v1"
    assert ms.probe_local_reasoning_families(url, _urlopen=fake)["mimo"] is False
    assert fake.calls == 2
    # Cooldown elapsed: the next call must re-probe and recover, not serve the stale miss.
    assert ms.probe_local_reasoning_families(url, _urlopen=fake)["mimo"] is True