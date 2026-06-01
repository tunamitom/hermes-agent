"""Startup environment precedence and selective compression notice delivery."""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

from conftest import REPO_ROOT, run_async


def test_network_env_import_precedence(hermes_home):
    profile = hermes_home / "profiles" / "secondary"
    profile.mkdir(parents=True)
    shared = hermes_home / ".hermes"
    shared.mkdir()
    (shared / "network.env").write_text(
        "# comment\n\nPATCH_SHARED=shared\nPATCH_PROFILE=shared\nPATCH_EXPLICIT=shared\ninvalid\n")
    (profile / "network.env").write_text("PATCH_PROFILE=profile\nPATCH_EXPLICIT=profile\n")
    env = dict(os.environ, HERMES_HOME=str(profile), HOME=str(hermes_home),
               PYTHONPATH=str(REPO_ROOT), PATCH_EXPLICIT="process",
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c",
        "import hermes_constants, os, json; print(json.dumps({k: os.getenv(k) "
        "for k in ('PATCH_SHARED', 'PATCH_PROFILE', 'PATCH_EXPLICIT')}))"],
        cwd=hermes_home, env=env, capture_output=True, text=True, timeout=20, check=True)
    assert json.loads(result.stdout) == {
        "PATCH_SHARED": "shared", "PATCH_PROFILE": "profile", "PATCH_EXPLICIT": "process"}


def _event(chat_id="123", message_id="456"):
    """Local copy of the event builder: patch folders stay self-contained (no
    cross-folder imports), so each folder also passes on its own."""
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.session import Platform, SessionSource

    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="private",
            user_id="42",
            user_name="TestUser",
        ),
        message_id=message_id,
    )


def test_progress_suppression_is_selective(monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway import warning_notifications as warnings
    runner = GatewayRunner(GatewayConfig())
    notices = AsyncMock()
    adapter = NS(emit_warning=notices)
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: adapter)
    monkeypatch.setattr(runner, "_hmwa_hygiene_cancel_or_adopt", AsyncMock(return_value=None))
    monkeypatch.setattr(runner, "_hmwa_hygiene_stamp", lambda *a: None)
    monkeypatch.setattr(runner, "_hmwa_hygiene_record_failure_cooldown", AsyncMock())
    import gateway.run as gw
    monkeypatch.setattr(gw, "_record_hygiene_cooldown", lambda *a: None)
    config = {}
    monkeypatch.setattr(warnings, "effective_user_config", lambda: config)
    fence = NS(commit_watermark_fenced=False, is_cancelled=False,
               deadline_exceeded=False, progress_observed=False,
               seconds_since_progress=lambda: 2)
    attempt = NS(commit_fence=fence, wait_started=time.monotonic(), agent=NS(), meta={})
    hs = NS(total_ceiling_seconds=3600, timeout_seconds=1)
    source = _event().source
    entry = NS(session_id="notice-session")

    async def invoke(handler):
        try:
            raise TimeoutError("controlled compression timeout")
        except TimeoutError:
            try:
                await handler(attempt, hs, entry, "notice-key", source)
            except TimeoutError:
                pass

    async def main():
        for handler in (runner._hmwa_hygiene_on_turn_hold, runner._hmwa_hygiene_on_timeout):
            config.clear()
            notices.reset_mock()
            await invoke(handler)
            assert notices.await_count == 1, "default delivery must remain enabled"
            config["display"] = {"suppress_compression_progress_notices": True}
            notices.reset_mock()
            await invoke(handler)
            assert notices.await_count == 0, "opt-in must silence progress notices"
            # General actionable warnings remain deliverable under the narrow setting.
            assert warnings.warning_notifications_enabled(Platform.TELEGRAM)
            await runner._hmwa_hygiene_notify(source, {}, "Compression failed: repair credentials", "failure")
            assert notices.await_count == 1
    run_async(main())
