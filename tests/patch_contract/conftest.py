"""Shared fixtures for the hermes-agent local-patch contract suite.

The suite lives OUTSIDE the repo (staging) and imports the real checkout at
``/Users/tom/.hermes/hermes-agent`` (override with ``HERMES_PATCH_REPO``).
Every test is a plain sync test that drives async code with ``asyncio.run``
(no pytest plugins beyond core pytest).  All network traffic is loopback:
stub HTTP servers bind 127.0.0.1 on a random port.

Design contract: tests FAIL loudly (AttributeError/AssertionError) when a
patch's symbols or behavior are gone — never skip silently.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import tempfile
import socket
import ipaddress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

# Isolate import-time configuration (including network.env), not just test bodies.
_original_env = dict(os.environ)
_import_home = tempfile.TemporaryDirectory(prefix="patch-contract-")
os.environ.clear()
for key in ("PATH", "LANG", "HERMES_PATCH_REPO", "SYSTEMROOT"):
    if key in _original_env:
        os.environ[key] = _original_env[key]
os.environ.update(HOME=_import_home.name, HERMES_HOME=_import_home.name,
                  NO_PROXY="*", HERMES_DISABLE_LAZY_INSTALLS="1")
sys.dont_write_bytecode = True

_blocked_connections = []
_real_getaddrinfo = socket.getaddrinfo
_original_execs = {name: getattr(os, name) for name in ("execv", "execve", "execvp", "execvpe")}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
def _local_address(address):
    if isinstance(address, tuple):
        host = address[0]
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = host == "localhost"
        if not allowed:
            _blocked_connections.append(address)
            raise AssertionError(f"Unexpected outbound connection: {address}")

def _connect(self, address):
    _local_address(address)
    return _real_connect(self, address)

def _connect_ex(self, address):
    _local_address(address)
    return _real_connect_ex(self, address)

def _getaddrinfo(host, port, *args, **kwargs):
    if host is not None:
        _local_address((host, port))
    return _real_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = _getaddrinfo
socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex

def pytest_unconfigure(config):
    socket.getaddrinfo = _real_getaddrinfo
    for name, original in _original_execs.items():
        setattr(os, name, original)
    socket.socket.connect = _real_connect
    socket.socket.connect_ex = _real_connect_ex
    os.environ.clear()
    os.environ.update(_original_env)
    _import_home.cleanup()

_REL_REPO = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(
    os.environ.get(
        "HERMES_PATCH_REPO",
        str(_REL_REPO) if (_REL_REPO / "agent" / "message_sanitization.py").is_file() else "/Users/tom/.hermes/hermes-agent",
    )
)
if not (REPO_ROOT / "agent" / "message_sanitization.py").is_file():
    raise RuntimeError(
        f"hermes-agent checkout not found at {REPO_ROOT} — set HERMES_PATCH_REPO"
    )
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Hermes launch-trampoline guard ──────────────────────────────────────────
# ``import gateway.run`` pulls in ``hermes_bootstrap``, whose module-level
# ``prepare_launch()`` may ``os.execv`` the process into the managed
# interpreter when the checkout's dependency generation is stale (e.g. another
# update is mid-flight) — silently REPLACING the test process mid-run.
# HERMES_DISABLE_LAZY_INSTALLS is prepare_launch's documented opt-out (it
# returns None before any completion work).  The exec guard below turns any
# remaining re-exec attempt into a loud RuntimeError instead of a silent hijack.
os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"

for _exec_name in ("execv", "execve", "execvp", "execvpe"):
    def _blocked_exec(*_args, __name=_exec_name, **_kwargs):
        raise RuntimeError(
            f"os.{__name}() re-exec attempted during the patch-contract suite — "
            "a hermes launch trampoline tried to replace the test process"
        )
    setattr(os, _exec_name, _blocked_exec)


def run_async(coro):
    """Run a coroutine to completion from a sync test (no pytest-asyncio)."""
    return asyncio.run(coro)


# ── Environment isolation ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _env_isolation(tmp_path, monkeypatch):
    _blocked_connections.clear()
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path),
                      NO_PROXY="*", HERMES_DISABLE_LAZY_INSTALLS="1",
                      PATH=saved.get("PATH", ""))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    token = set_hermes_home_override(str(tmp_path))
    try:
        yield
    finally:
        reset_hermes_home_override(token)
        os.environ.clear()
        os.environ.update(saved)
        assert not _blocked_connections, f"Unexpected outbound attempts: {_blocked_connections}"


@pytest.fixture
def hermes_home(_env_isolation, tmp_path):
    return tmp_path


# ── Loopback HTTP stub servers ───────────────────────────────────────────────

class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep test output clean
        pass

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def _read_body(self) -> bytes:
        # httpx may send chunked bodies (no Content-Length); de-chunk manually.
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            chunks = []
            while True:
                size_line = self.rfile.readline().strip()
                size = int(size_line.split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()  # trailing CRLF
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.readline()  # CRLF after each chunk
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _handle(self):
        body = self._read_body()
        payload = None
        try:
            payload = json.loads(body) if body else None
        except Exception:
            payload = None
        if payload is None and body:
            # Tolerate form-encoded bodies: urlencoded fields, JSON values inside.
            from urllib.parse import parse_qsl

            payload = {}
            for key, value in parse_qsl(body.decode("utf-8", "replace")):
                try:
                    payload[key] = json.loads(value)
                except Exception:
                    payload[key] = value
        req = SimpleNamespace(
            method=self.command,
            path=self.path,
            headers={k: v for k, v in self.headers.items()},
            body=body,
            json=payload if isinstance(payload, dict) else None,
        )
        self.server.requests.append(req)
        status, content_type, resp_body = self.server.router(req)
        if isinstance(resp_body, str):
            resp_body = resp_body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)


@pytest.fixture(scope="session")
def stub_http_server():
    """Factory for loopback stub HTTP servers.

    ``start(router)`` -> namespace(base_url, requests, port).  The router is
    ``fn(req) -> (status:int, content_type:str, body:str|bytes)`` and each
    request is recorded on ``requests`` (SimpleNamespace with .method/.path/
    .headers/.body/.json).
    """
    servers = []

    def start(router):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        srv.router = router
        srv.requests = []
        thread = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        servers.append(srv)
        return SimpleNamespace(
            base_url=f"http://127.0.0.1:{srv.server_address[1]}",
            port=srv.server_address[1],
            requests=srv.requests,
            server=srv,
        )

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


# ── Telegram Bot API stub + real PTB plumbing ────────────────────────────────

TELEGRAM_BOT_ID = 42
TELEGRAM_BOT_USERNAME = "hermes_test_bot"
# PTB validates the token's shape; the secret part must be long enough.
TELEGRAM_BOT_TOKEN = "424242424:" + "Abcdef0123456789" * 3  # 48-char secret


@pytest.fixture
def telegram_api_stub(stub_http_server):
    """Stub Telegram Bot API server (getMe / setMessageReaction / default ok).

    Every request is recorded on ``requests`` as SimpleNamespace(path, json).
    """
    def router(req):
        method = req.path.rstrip("/").rsplit("/", 1)[-1]
        if method == "getMe":
            result = {
                "id": TELEGRAM_BOT_ID,
                "is_bot": True,
                "first_name": "Hermes",
                "username": TELEGRAM_BOT_USERNAME,
                "can_join_groups": True,
                "can_read_all_group_messages": False,
                "supports_inline_queries": False,
            }
        else:
            result = True
        return 200, "application/json", json.dumps({"ok": True, "result": result})

    svc = stub_http_server(router)
    svc.token = TELEGRAM_BOT_TOKEN
    return svc


@pytest.fixture
def make_telegram_bot(telegram_api_stub):
    """Factory for a REAL python-telegram-bot Bot pointed at the stub server.

    Call ``await bot.initialize()`` (serves getMe) before reading ``bot.id``.
    """
    import telegram

    def make():
        return telegram.Bot(
            token=telegram_api_stub.token,
            base_url=telegram_api_stub.base_url.rstrip("/") + "/",
        )

    return make


@pytest.fixture
def make_telegram_adapter():
    """Construct the real adapter; replace only its external Bot transport."""
    def make(bot, **extra):
        from gateway.config import Platform, PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = TelegramAdapter(PlatformConfig(
            enabled=True, token=TELEGRAM_BOT_TOKEN, extra=dict(extra)))
        adapter._bot = bot
        return adapter

    return make


@pytest.fixture
def make_group_update():
    """Build a realistic Telegram Update JSON payload for a forum-group message.

    Defaults: supergroup ``-1001234567890`` (is_forum), topic 500, human
    sender 777, text free of @handles.  ``reply_to`` is a full message dict;
    ``quote`` is a TextQuote dict.
    """
    def make(*, message_id=505, thread_id=500, reply_to=None, quote=None,
             text="hello there", sender_id=777, sender_is_bot=False):
        chat = {
            "id": -1001234567890,
            "type": "supergroup",
            "title": "Ops",
            "is_forum": True,
        }
        message = {
            "message_id": message_id,
            "date": 1760000000,
            "chat": chat,
            "from": {
                "id": sender_id,
                "is_bot": sender_is_bot,
                "first_name": "Alice",
            },
            "message_thread_id": thread_id,
            "is_topic_message": True,
            "text": text,
        }
        if reply_to is not None:
            message["reply_to_message"] = reply_to
        if quote is not None:
            message["quote"] = quote
        return {"update_id": 9001, "message": message}

    return make


@pytest.fixture
def bot_reply_message():
    """A reply_to_message dict authored by the bot (used with make_group_update)."""
    def make(*, message_id=517, thread_id=500, text="bot topic card"):
        return {
            "message_id": message_id,
            "date": 1759999000,
            "chat": {
                "id": -1001234567890,
                "type": "supergroup",
                "title": "Ops",
                "is_forum": True,
            },
            "from": {
                "id": TELEGRAM_BOT_ID,
                "is_bot": True,
                "first_name": "Hermes",
                "username": TELEGRAM_BOT_USERNAME,
            },
            "message_thread_id": thread_id,
            "is_topic_message": True,
            "text": text,
        }

    return make