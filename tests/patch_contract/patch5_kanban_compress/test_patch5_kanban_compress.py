"""Contract tests — patch 5: feat(kanban): per-session model overrides +
/api/sessions/{id}/compress bridge (gateway/platforms/api_server.py +
session_models.json file fallback loader).

Behavior under contract:
* ``POST /api/sessions/{session_id}/compress`` is registered in the route table
  and the ``session_compress`` capability, resolves the session's live
  SessionSource and injects a synthetic ``/compress`` MessageEvent
  (allow_gateway_control=True) into the gateway's real message pipeline
  (``runner._handle_message``) — body ``mode``/``focus`` become the slash args;
* the response carries the pipeline's reply plus a fresh ``contextTokens``:
  the value the pipeline persisted (``last_prompt_tokens``) preferred over a
  history re-estimate; empty body = plain full compress (never a 400);
* ``GatewayAgentCacheMixin._load_session_models_file_override`` reads
  ``<HERMES_HOME>/session_models.json`` (kanban model selector): known fields
  only (``model, provider, base_url, api_key, api_mode, max_tokens``), 7-day
  TTL on ``updatedAt``, api_key-less entries valid;
* ``_resolve_session_agent_runtime`` falls back to that file when no in-memory
  /model override exists (without poisoning in-memory state), and
  ``_apply_session_model_override`` accepts an explicit override;
* ``_persist_manual_compression`` stores the post-compression estimate the
  reply quotes (``after_tokens``) into ``last_prompt_tokens``.

Integration boundary: real aiohttp route dispatch over the adapter's actual
``_http_route_table`` (loopback TestClient) for the compress bridge, and the
real file/resolver path for the session_models.json loader.
"""

from __future__ import annotations

import datetime as _dt
import json

import pytest

from conftest import run_async

SESSION_KEY = "agent:main:telegram:dm:123456789"


# ── Fakes mirroring the gateway seams (session store + runner) ─────────────

def _entry_with_origin(session_id="sess-1"):
    from gateway.session import Platform, SessionEntry, SessionSource

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="8355087887", chat_type="dm",
        user_id="8355087887")
    now = _dt.datetime.now()
    return SessionEntry(
        session_key=SESSION_KEY, session_id=session_id,
        created_at=now, updated_at=now, origin=source,
        platform=Platform.TELEGRAM, chat_type="dm")


class _FakeStore:
    def __init__(self, entry):
        self._entry = entry

    def lookup_by_session_id(self, session_id):
        return self._entry if self._entry and self._entry.session_id == session_id else None


class _FakeRunner:
    """Captures the injected event; returns a canned reply string."""

    def __init__(self, entry, reply="🗜️ Compacted.", persist_tokens=None):
        self.session_store = _FakeStore(entry)
        self.reply = reply
        self.captured = None
        self.persist_tokens = persist_tokens
        self._entry = entry

    async def _handle_message(self, event):
        self.captured = event
        # Mirror the real pipeline: manual /compress persists its post-compression
        # request-size estimate into last_prompt_tokens.
        if self.persist_tokens is not None and self._entry is not None:
            self._entry.last_prompt_tokens = self.persist_tokens
        return self.reply

    def _resolve_profile_home_for_source(self, source):
        import pathlib

        return pathlib.Path("/tmp/hermes-test-home")


def _adapter_with_runner(runner):
    from gateway.platforms.api_server import APIServerAdapter

    from gateway.config import PlatformConfig
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter.gateway_runner = runner
    return adapter


async def _http_request(adapter, method, url, *, body=None, auth="Bearer test-key"):
    """Dispatch through the production route table using a real aiohttp client.

    Returns (status:int, payload:dict|None, response_text:str).
    """
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    app = web.Application()
    for route_method, path, handler in adapter._http_route_table():
        app.router.add_route(route_method, path, handler)

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        headers = {"Authorization": auth} if auth else {}
        if body is None:
            resp = await client.request(method, url, headers=headers, data=b"")
        else:
            resp = await client.request(method, url, headers=headers, json=body)
        text = await resp.text()
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        return resp.status, payload, text
    finally:
        await client.close()


async def _http_compress(adapter, session_id, **kwargs):
    return await _http_request(adapter, "POST", f"/api/sessions/{session_id}/compress", **kwargs)


def _patch_history(monkeypatch, messages, calls=None):
    from gateway.platforms.api_server import APIServerAdapter

    async def _hist(self, session_id):
        if calls is not None:
            calls.append(session_id)
        return messages

    monkeypatch.setattr(APIServerAdapter, "_conversation_history_for_session", _hist)


# ── Integration: the compress bridge over real HTTP route dispatch ──────────

def test_compress_route_dispatches_real_slash_event(monkeypatch):
    entry = _entry_with_origin("sess-1")
    runner = _FakeRunner(entry)
    adapter = _adapter_with_runner(runner)
    _patch_history(monkeypatch, [{"role": "user", "content": "hi"}])

    status, payload, _text = run_async(_http_compress(adapter, "sess-1", body={}))

    assert status == 200
    assert payload["ok"] is True
    assert payload["reply"] == "🗜️ Compacted."
    assert payload["session_id"] == "sess-1"
    assert "contextTokens" in payload

    evt = runner.captured
    assert evt is not None, "the bridge did not inject an event into the pipeline"
    assert evt.get_command() == "compress"
    assert evt.text == "/compress"
    for field in ("platform", "chat_id", "chat_type", "user_id", "thread_id"):
        assert getattr(evt.source, field) == getattr(entry.origin, field)
    assert evt.allow_gateway_control is True


def test_compress_body_args_forwarded_to_slash_command(monkeypatch):
    entry = _entry_with_origin("sess-1")
    runner = _FakeRunner(entry)
    adapter = _adapter_with_runner(runner)
    _patch_history(monkeypatch, [])

    status, _payload, _text = run_async(
        _http_compress(adapter, "sess-1", body={"mode": "here", "focus": "server migration"}))

    assert status == 200
    assert runner.captured.text == "/compress here server migration"


def test_compress_empty_body_is_plain_full_compress(monkeypatch):
    """Empty/non-JSON body = full compress — _read_json_body's 400 must not fire."""
    entry = _entry_with_origin("sess-1")
    runner = _FakeRunner(entry)
    adapter = _adapter_with_runner(runner)
    _patch_history(monkeypatch, [])

    status, _payload, _text = run_async(_http_compress(adapter, "sess-1", body=None))

    assert status == 200
    assert runner.captured.text == "/compress"


def test_compress_unknown_session_404():
    runner = _FakeRunner(None)
    adapter = _adapter_with_runner(runner)

    status, _payload, text = run_async(_http_compress(adapter, "sess-missing", body={}))

    assert status == 404
    assert "session_not_live" in text


def test_compress_no_runner_503():
    adapter = _adapter_with_runner(None)

    status, _payload, text = run_async(_http_compress(adapter, "sess-1", body={}))

    assert status == 503
    assert "gateway_unavailable" in text


def test_compress_prefers_persisted_estimate(monkeypatch):
    """The pipeline stores its post-compression estimate in last_prompt_tokens —
    the response must return THAT (what the reply quotes), not re-estimate."""
    entry = _entry_with_origin("sess-1")
    runner = _FakeRunner(entry, persist_tokens=32455)
    adapter = _adapter_with_runner(runner)
    history_calls = []
    _patch_history(monkeypatch, [{"role": "user", "content": "hi"}], history_calls)

    status, payload, _text = run_async(_http_compress(adapter, "sess-1", body={}))

    assert status == 200
    assert payload["contextTokens"] == 32455
    assert history_calls == [], "persisted estimate must win over a history re-estimate"


def test_compress_history_estimate_fallback(monkeypatch, hermes_home):
    """No persisted estimate (0): fall back to the history estimate under the
    session's profile home."""
    from agent.model_metadata import estimate_request_tokens_rough

    entry = _entry_with_origin("sess-1")
    runner = _FakeRunner(entry)
    adapter = _adapter_with_runner(runner)
    messages = [{"role": "user", "content": "hi"}]
    history_calls = []
    from hermes_constants import get_hermes_home
    profile = hermes_home / "secondary"
    profile.mkdir()
    runner._resolve_profile_home_for_source = lambda source: profile
    async def scoped_history(self, session_id):
        history_calls.append((session_id, get_hermes_home()))
        return messages if get_hermes_home() == profile else []
    from gateway.platforms.api_server import APIServerAdapter
    monkeypatch.setattr(APIServerAdapter, "_conversation_history_for_session", scoped_history)

    status, payload, _text = run_async(_http_compress(adapter, "sess-1", body={}))

    assert status == 200
    assert payload["contextTokens"] == estimate_request_tokens_rough(messages)
    assert history_calls == [("sess-1", profile)]
    assert get_hermes_home() == hermes_home


# ── Route registration (dispatch table + capability surface) ───────────────

def test_compress_capability_advertised_over_http():
    adapter = _adapter_with_runner(_FakeRunner(None))
    status, payload, _ = run_async(_http_request(adapter, "GET", "/v1/capabilities"))
    assert status == 200
    capability = payload["endpoints"]["session_compress"]
    assert capability["method"] == "POST"
    assert capability["path"] == "/api/sessions/{session_id}/compress"


# ── session_models.json file fallback loader ───────────────────────────────

def _make_runner():
    from gateway.run import GatewayRunner

    from gateway.config import GatewayConfig
    return GatewayRunner(GatewayConfig())


def _write_models_file(home, entry):
    (home / "session_models.json").write_text(
        json.dumps({SESSION_KEY: entry}), encoding="utf-8")


def test_file_override_loader_filters_to_known_fields(hermes_home):
    runner = _make_runner()
    _write_models_file(hermes_home, {
        "model": "fast", "provider": "custom", "base_url": "http://turin:8003/v1",
        "api_mode": "chat_completions", "max_tokens": 8192,
        "agentId": "main", "updatedAt": 0, "junk": "ignored",
    })

    loaded = runner._load_session_models_file_override(SESSION_KEY)

    assert loaded == {
        "model": "fast", "provider": "custom", "base_url": "http://turin:8003/v1",
        "api_mode": "chat_completions", "max_tokens": 8192,
    }
    assert "api_key" not in loaded, "api_key-less kanban entries must load as-is"

    _write_models_file(hermes_home, {
        "model": "fast", "provider": "custom", "base_url": "http://turin:8003/v1",
        "api_key": "sk-secret", "updatedAt": 0,
    })
    assert runner._load_session_models_file_override(SESSION_KEY)["api_key"] == "sk-secret"


def test_file_override_loader_ttl_and_misses(hermes_home):
    import time

    runner = _make_runner()

    assert runner._load_session_models_file_override(SESSION_KEY) is None  # no file

    # Stale entry (older than 7 days) is ignored.
    _write_models_file(hermes_home, {
        "model": "fast", "provider": "custom", "base_url": "http://turin:8003/v1",
        "updatedAt": int(time.time() * 1000) - 8 * 24 * 60 * 60 * 1000,
    })
    assert runner._load_session_models_file_override(SESSION_KEY) is None

    # updatedAt == 0 means "no timestamp": never expires (kanban legacy entries).
    _write_models_file(hermes_home, {
        "model": "fast", "provider": "custom", "base_url": "http://turin:8003/v1",
        "updatedAt": 0,
    })
    assert runner._load_session_models_file_override(SESSION_KEY) is not None

    # Fresh entry survives; unknown session key and malformed files are None.
    _write_models_file(hermes_home, {
        "model": "fast", "provider": "custom", "base_url": "http://turin:8003/v1",
        "updatedAt": int(time.time() * 1000),
    })
    assert runner._load_session_models_file_override(SESSION_KEY) is not None
    assert runner._load_session_models_file_override("no-such-session") is None

    (hermes_home / "session_models.json").write_text("{not json", encoding="utf-8")
    assert runner._load_session_models_file_override(SESSION_KEY) is None


@pytest.mark.parametrize("available", [False, True])
def test_file_override_applies_without_api_key_and_without_memory(hermes_home, monkeypatch, available):
    """Kanban file entries carry no api_key: model+base_url must still reach the
    turn, applied from the FILE without being cached into in-memory state."""
    import gateway.run as gw_run

    monkeypatch.setattr(gw_run, "_resolve_gateway_model",
                        lambda user_config=None: "smart")
    monkeypatch.setattr(
        gw_run, "_resolve_runtime_agent_kwargs",
        lambda: {"model": "smart", "provider": "custom", "api_key": "EMPTY",
                 "base_url": "http://config-default/v1", "capabilities": {}, "args": []})
    def provider_runtime(*args, **kwargs):
        if not available:
            raise ValueError("provider intentionally unavailable")
        return {"provider": "custom", "api_key": "provider-key", "base_url": "http://provider/v1"}
    monkeypatch.setattr(gw_run, "_resolve_runtime_agent_kwargs_for_provider", provider_runtime)
    _write_models_file(hermes_home, {
        "model": "fast", "provider": "custom", "base_url": "http://turin:8003/v1",
        "agentId": "main", "updatedAt": 0,
    })

    runner = _make_runner()
    model, runtime = runner._resolve_session_agent_runtime(session_key=SESSION_KEY)

    assert model == "fast"
    assert runtime["base_url"] == "http://turin:8003/v1"
    assert runtime["provider"] == "custom"
    _write_models_file(hermes_home, {"model": "new-choice", "provider": "custom",
                                    "base_url": "http://new-choice/v1"})
    model, runtime = runner._resolve_session_agent_runtime(session_key=SESSION_KEY)
    assert (model, runtime["base_url"]) == ("new-choice", "http://new-choice/v1")
    runner._session_state(SESSION_KEY).conversation.model_override = {
        "model": "in-chat", "provider": "custom", "base_url": "http://in-chat/v1",
        "api_key": "memory-key"}
    model, runtime = runner._resolve_session_agent_runtime(session_key=SESSION_KEY)
    assert (model, runtime["base_url"]) == ("in-chat", "http://in-chat/v1")


def test_apply_session_model_override_accepts_explicit_override():
    """The override dict may be supplied directly (file fallback): partial
    overrides don't clobber untouched runtime fields."""
    runner = _make_runner()
    runtime = {"model": "smart", "provider": "openai", "api_key": "cfg-key",
               "base_url": "http://config-default/v1"}

    model, runtime = runner._apply_session_model_override(
        SESSION_KEY, "smart", runtime,
        override={"model": "fast", "provider": "custom",
                  "base_url": "http://turin:8003/v1"})

    assert model == "fast"
    assert runtime["base_url"] == "http://turin:8003/v1"
    assert runtime["provider"] == "custom"
    assert runtime["api_key"] == "cfg-key", "absent override fields must not clobber defaults"

@pytest.mark.parametrize("rotate", [False, True])
def test_http_compress_real_slash_persists_result(hermes_home, monkeypatch, rotate):
    """HTTP -> real slash handler -> compression result -> durable session store."""
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock
    from gateway.session import SessionStore
    from agent import conversation_compression_manual as manual

    runner = _make_runner()
    entry = runner.session_store.get_or_create_session(_entry_with_origin().origin)
    for i in range(12):
        runner.session_store.append_to_transcript(
            entry.session_id, {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"})
    old_id = entry.session_id
    agent = NS(session_id="rotated-child" if rotate else old_id,
               _last_compaction_in_place=not rotate, context_compressor=NS())
    monkeypatch.setattr(runner, "_build_manual_compression_agent", AsyncMock(return_value=agent))
    monkeypatch.setattr(runner, "_resolve_session_agent_runtime",
                        lambda **kw: ("test-model", {"api_key": "test-key"}))
    monkeypatch.setattr(runner, "_resolve_session_reasoning_config", lambda **kw: None)
    monkeypatch.setattr(runner, "_cleanup_agent_resources_off_loop", AsyncMock())
    # The expensive compression engine is the seam; caller and persistence are real.
    def compress(agent, messages, request, **kw):
        if rotate:
            # A real compressor creates the continuation row before returning.
            from hermes_state import SessionDB
            db = SessionDB()
            try:
                db.create_session(agent.session_id, source="telegram", parent_session_id=old_id)
            finally:
                db.close()
        return manual.CompressResult(
            status="compressed", before_messages=messages,
            after_messages=[{"role": "assistant", "content": "summary"}],
            before_tokens=60000, after_tokens=32455, request=request,
            summary={"headline": "Compacted", "token_line": "60000 → 32455", "note": ""})
    monkeypatch.setattr(manual, "compress_now", compress)
    # HTTP routing is tested separately; dispatch directly to the real slash
    # entry to avoid authorization/provider setup unrelated to compression.
    monkeypatch.setattr(runner, "_handle_message", runner._handle_compress_command)
    adapter = _adapter_with_runner(runner)
    status, payload, _ = run_async(_http_compress(adapter, old_id, body={}))
    assert status == 200
    assert "32455" in payload["reply"]
    assert payload["contextTokens"] == 32455
    reopened = SessionStore(hermes_home / "sessions", runner.config)
    assert payload["session_id"] == agent.session_id
    assert reopened.lookup_by_session_id(agent.session_id).last_prompt_tokens == 32455
    if rotate:
        assert reopened.load_transcript(agent.session_id)[0]["content"] == "summary"


def test_rotated_history_uses_profile_and_restores_home(hermes_home, monkeypatch):
    from hermes_constants import get_hermes_home
    from agent.model_metadata import estimate_request_tokens_rough
    profile = hermes_home / "profiles" / "secondary"
    profile.mkdir(parents=True)
    messages = [{"role": "user", "content": "profile-specific transcript"}]
    (profile / "history.json").write_text(json.dumps(messages))
    (hermes_home / "history.json").write_text("[]")
    entry = _entry_with_origin()
    runner = _FakeRunner(entry)
    async def rotate(event):
        entry.session_id = "rotated"
        return "compacted"
    runner._handle_message = rotate
    runner._resolve_profile_home_for_source = lambda source: profile
    calls = []
    async def history(self, session_id):
        calls.append((session_id, get_hermes_home()))
        return json.loads((get_hermes_home() / "history.json").read_text())
    from gateway.platforms.api_server import APIServerAdapter
    monkeypatch.setattr(APIServerAdapter, "_conversation_history_for_session", history)
    async def main():
        before = get_hermes_home()
        result = await _http_compress(_adapter_with_runner(runner), "sess-1", body={})
        assert get_hermes_home() == before
        return result
    status, payload, _ = run_async(main())
    assert status == 200
    assert payload["session_id"] == "rotated"
    assert payload["contextTokens"] == estimate_request_tokens_rough(messages)
    assert calls == [("rotated", profile)]
