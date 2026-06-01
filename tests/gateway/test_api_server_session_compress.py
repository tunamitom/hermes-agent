"""POST /api/sessions/{session_id}/compress — thin bridge to the /compress slash pipeline.

The route exists for kanban's "Take a breather" button: it must run the EXACT slash-command
path (busy gating, profile scoping, codex_app_server handling), not a parallel compression
implementation. These tests pin that contract: synthetic MessageEvent construction, reply
passthrough, fresh token estimate for the usage pill, and the failure modes kanban can hit.
"""
import asyncio
import datetime as _dt
from unittest.mock import MagicMock

import pytest

from gateway.session import SessionEntry, SessionSource, Platform
from gateway.platforms.api_server import APIServerAdapter
from gateway.slash_commands_session import GatewaySessionCommandsMixin


def _entry_with_origin(session_id: str = "sess-1"):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="8355087887", chat_type="dm", user_id="8355087887")
    now = _dt.datetime.now()
    return SessionEntry(
        session_key="agent:main:telegram:dm:8355087887", session_id=session_id,
        created_at=now, updated_at=now, origin=source, platform=Platform.TELEGRAM, chat_type="dm")


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


def _adapter_with_runner(tmp_path, runner, monkeypatch):
    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._api_key = "test-key"
    adapter._pending_agent_requests = 0
    adapter.gateway_runner = runner
    adapter._session_dbs = {}
    return adapter


class _FakeRequest:
    def __init__(self, session_id, json_body=None, has_body=True, headers=None):
        self.match_info = {"session_id": session_id}
        self.headers = headers or {"Authorization": "Bearer test-key"}
        self._json = json_body
        self.can_read_body = has_body
        self.path = f"/api/sessions/{session_id}/compress"
        self.method = "POST"
        self.app = {}

    async def json(self):
        if self._json is None:
            raise ValueError("no body")
        return self._json


def _patch_read_json(monkeypatch, body=None, err=None):
    async def _read(self, request):
        return (body or {}), err
    monkeypatch.setattr(APIServerAdapter, "_read_json_body", _read)


def _patch_history(monkeypatch, messages):
    async def _hist(self, session_id):
        return messages
    monkeypatch.setattr(APIServerAdapter, "_conversation_history_for_session", _hist)


async def _ok_request(adapter, request):
    return await adapter._handle_session_compress(request)


def test_compress_route_registered():
    adapter = MagicMock(spec=APIServerAdapter)
    paths = {(m, p) for m, p, _ in APIServerAdapter._http_route_table(adapter)}
    assert ("POST", "/api/sessions/{session_id}/compress") in paths


@pytest.mark.asyncio
async def test_compress_dispatches_real_slash_event(tmp_path, monkeypatch):
    entry = _entry_with_origin("sess-1")
    runner = _FakeRunner(entry)
    adapter = _adapter_with_runner(tmp_path, runner, monkeypatch)
    _patch_read_json(monkeypatch)
    _patch_history(monkeypatch, [{"role": "user", "content": "hi"}])

    resp = await _ok_request(adapter, _FakeRequest("sess-1", json_body={}))

    assert resp.status == 200
    import json as _json
    payload = _json.loads(resp.text)
    assert payload["ok"] is True
    assert payload["reply"] == "🗜️ Compacted."
    # The injected event must be a REAL slash command bound to the live session source.
    evt = runner.captured
    assert evt is not None
    assert evt.get_command() == "compress"
    assert evt.source is entry.origin
    assert evt.allow_gateway_control is True


@pytest.mark.asyncio
async def test_compress_focus_args_forwarded(tmp_path, monkeypatch):
    entry = _entry_with_origin("sess-1")
    runner = _FakeRunner(entry)
    adapter = _adapter_with_runner(tmp_path, runner, monkeypatch)
    _patch_read_json(monkeypatch, body={"focus": "server migration"})
    _patch_history(monkeypatch, [])

    resp = await _ok_request(adapter, _FakeRequest("sess-1", json_body={"focus": "server migration"}))
    assert resp.status == 200
    assert runner.captured.text == "/compress server migration"


@pytest.mark.asyncio
async def test_compress_unknown_session_404(tmp_path, monkeypatch):
    runner = _FakeRunner(None)
    adapter = _adapter_with_runner(tmp_path, runner, monkeypatch)
    _patch_read_json(monkeypatch)

    resp = await _ok_request(adapter, _FakeRequest("sess-missing"))
    assert resp.status == 404


@pytest.mark.asyncio
async def test_compress_no_runner_503(tmp_path, monkeypatch):
    adapter = _adapter_with_runner(tmp_path, None, monkeypatch)
    _patch_read_json(monkeypatch)

    resp = await _ok_request(adapter, _FakeRequest("sess-1"))
    assert resp.status == 503


def _patch_history_recording(monkeypatch, messages, calls):
    async def _hist(self, session_id):
        calls.append(session_id)
        return messages
    monkeypatch.setattr(APIServerAdapter, "_conversation_history_for_session", _hist)


@pytest.mark.asyncio
async def test_compress_prefers_persisted_estimate(tmp_path, monkeypatch):
    """The pipeline stores its post-compression estimate in last_prompt_tokens — the response
    must return THAT (what the reply quotes) and not re-estimate from history."""
    entry = _entry_with_origin("sess-1")
    runner = _FakeRunner(entry, persist_tokens=32455)
    adapter = _adapter_with_runner(tmp_path, runner, monkeypatch)
    _patch_read_json(monkeypatch)
    history_calls = []
    _patch_history_recording(monkeypatch, [{"role": "user", "content": "hi"}], history_calls)

    resp = await _ok_request(adapter, _FakeRequest("sess-1", json_body={}))

    import json as _json
    payload = _json.loads(resp.text)
    assert payload["contextTokens"] == 32455
    assert history_calls == []


@pytest.mark.asyncio
async def test_compress_history_estimate_fallback(tmp_path, monkeypatch):
    """No persisted estimate (store value 0): fall back to the history estimate."""
    from agent.model_metadata import estimate_request_tokens_rough
    entry = _entry_with_origin("sess-1")
    runner = _FakeRunner(entry)
    adapter = _adapter_with_runner(tmp_path, runner, monkeypatch)
    _patch_read_json(monkeypatch)
    messages = [{"role": "user", "content": "hi"}]
    history_calls = []
    _patch_history_recording(monkeypatch, messages, history_calls)

    resp = await _ok_request(adapter, _FakeRequest("sess-1", json_body={}))

    import json as _json
    payload = _json.loads(resp.text)
    assert payload["contextTokens"] == estimate_request_tokens_rough(messages)
    assert history_calls == ["sess-1"]


# --- _persist_manual_compression stores the estimate the reply quotes -------------------

class _FakeAsyncStore:
    def __init__(self):
        self.updates = []
        self.rewritten = []

    async def update_session(self, session_key, last_prompt_tokens=None, touch_activity=True):
        self.updates.append((session_key, last_prompt_tokens))

    async def rewrite_transcript(self, session_id, messages):
        self.rewritten.append((session_id, list(messages)))
        return True

    async def _save(self):
        pass


class _FakeCommands(GatewaySessionCommandsMixin):
    def __init__(self):
        self.async_session_store = _FakeAsyncStore()
        self.topic_bound = []

    def _sync_telegram_topic_binding(self, source, entry, reason=""):
        self.topic_bound.append((source, entry.session_id, reason))


class _FakeAgent:
    def __init__(self, session_id, in_place=False):
        self.session_id = session_id
        self._last_compaction_in_place = in_place


def _compression_entry(session_id="sess-1"):
    import datetime as _dt
    now = _dt.datetime.now()
    return SessionEntry(session_key="agent:aerith:telegram:dm:8355087887", session_id=session_id,
                        created_at=now, updated_at=now)


@pytest.mark.asyncio
async def test_persist_manual_compression_stores_after_tokens_in_place():
    cmds = _FakeCommands()
    entry = _compression_entry()
    await cmds._persist_manual_compression(
        _FakeAgent("sess-1", in_place=True), entry, None, [{"role": "assistant", "content": "x"}],
        after_tokens=32455)
    assert cmds.async_session_store.updates == [("agent:aerith:telegram:dm:8355087887", 32455)]


@pytest.mark.asyncio
async def test_persist_manual_compression_stores_after_tokens_on_rotation():
    cmds = _FakeCommands()
    entry = _compression_entry("sess-1")
    await cmds._persist_manual_compression(
        _FakeAgent("sess-2"), entry, None, [{"role": "assistant", "content": "x"}],
        after_tokens=32455)
    assert entry.session_id == "sess-2"
    assert cmds.async_session_store.rewritten == [("sess-2", [{"role": "assistant", "content": "x"}])]
    assert cmds.async_session_store.updates == [("agent:aerith:telegram:dm:8355087887", 32455)]
