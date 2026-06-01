"""Contract tests — patch 4: feat(gateway): customizable reaction emojis +
reaction on follow-up messages.

Behavior under contract:
* ``TelegramAdapter._reaction_emoji(env_var, default)``: TELEGRAM_REACTION_*
  env overrides the default emoji; an explicitly empty value means "no reaction"
  (skip or clear), never fall back to the default;
* lifecycle hooks: ``on_processing_start`` -> start emoji (default 👀),
  ``on_processing_complete`` -> 👍/👎 (or clear on empty value / CANCELLED) and
  then clears every pending busy reaction;
* ``on_busy_received(event, mode)`` reacts on the follow-up message (steer 🤔,
  queue 👀, interrupt ⚡, unknown -> 👀; TELEGRAM_REACTION_STEERED/QUEUED/
  INTERRUPTED override), tracks it for cleanup, returns True (suppress the
  text ack); with reactions disabled it returns False and sends nothing;
* ``BasePlatformAdapter.on_busy_received`` default returns False (no visual
  feedback, keep the text ack);
* config.yaml ``reaction_*`` keys bridge to the TELEGRAM_REACTION_* env vars.

Integration boundary: real python-telegram-bot ``Bot`` (initialized against a
loopback Bot API stub) driving the adapter's real reaction hooks — every
setMessageReaction call is asserted on the wire.
"""

from __future__ import annotations

import os

from conftest import run_async


def _reaction_calls(telegram_api_stub):
    """(chat_id, message_id, reaction-payload-or-None) per setMessageReaction call."""
    calls = []
    for req in telegram_api_stub.requests:
        if not req.path.rstrip("/").endswith("/setMessageReaction"):
            continue
        body = req.json or {}
        calls.append((body.get("chat_id"), body.get("message_id"), body.get("reaction") or None))
    return calls


def _emoji_of(reaction_payload):
    """Emoji string from PTB's serialized reaction payload (or None when clear)."""
    if reaction_payload is None:
        return None
    import json as _json
    blob = _json.dumps(reaction_payload, ensure_ascii=False)
    return blob  # callers assert on substring to stay serialization-agnostic


def _event(chat_id="123", message_id="456"):
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


def _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch, **extra):
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    bot = make_telegram_bot()
    return bot, make_telegram_adapter(bot, **extra)


# ── Integration: reaction lifecycle over the real Bot API wire ─────────────

def _check_processing_start_sets_default_eyes(make_telegram_bot, make_telegram_adapter,
                                            telegram_api_stub, monkeypatch):
    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)

    async def main():
        await bot.initialize()
        try:
            await adapter.on_processing_start(_event())
        finally:
            await bot.shutdown()

    run_async(main())
    calls = _reaction_calls(telegram_api_stub)
    assert len(calls) == 1 and calls[0][:2] == (123, 456)
    assert "\U0001f440" in _emoji_of(calls[0][2]), "start must react 👀"


def test_processing_start_env_override_and_empty_skips(
        make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch):
    # Upstream guardrails belong to this patch-sensitive scenario.
    _check_processing_start_sets_default_eyes(make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch)
    telegram_api_stub.requests.clear()
    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)

    monkeypatch.setenv("TELEGRAM_REACTION_START", "\U0001f525")

    async def main():
        await bot.initialize()
        try:
            await adapter.on_processing_start(_event())
        finally:
            await bot.shutdown()

    run_async(main())
    calls = _reaction_calls(telegram_api_stub)
    assert len(calls) == 1 and "\U0001f525" in _emoji_of(calls[0][2])

    # Explicit empty value: skip the reaction entirely (no default fallback).
    telegram_api_stub.requests.clear()
    monkeypatch.setenv("TELEGRAM_REACTION_START", "")
    run_async(main())
    assert _reaction_calls(telegram_api_stub) == []


def _check_processing_complete_success_and_failure_reactions(
        make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch):
    from gateway.platforms.event import ProcessingOutcome

    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)

    async def main():
        await bot.initialize()
        try:
            await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
            await adapter.on_processing_complete(_event(), ProcessingOutcome.FAILURE)
        finally:
            await bot.shutdown()

    run_async(main())
    calls = _reaction_calls(telegram_api_stub)
    assert len(calls) == 2
    assert "\U0001f44d" in _emoji_of(calls[0][2]), "SUCCESS must react 👍"
    assert "\U0001f44e" in _emoji_of(calls[1][2]), "FAILURE must react 👎"


def test_processing_complete_env_overrides(
        make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch):
    # Upstream guardrails belong to this patch-sensitive scenario.
    _check_processing_complete_success_and_failure_reactions(make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch)
    telegram_api_stub.requests.clear()
    _check_processing_complete_cancelled_clears(make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch)
    telegram_api_stub.requests.clear()
    from gateway.platforms.event import ProcessingOutcome

    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)
    monkeypatch.setenv("TELEGRAM_REACTION_SUCCESS", "✅")
    monkeypatch.setenv("TELEGRAM_REACTION_FAILURE", "❌")

    async def main():
        await bot.initialize()
        try:
            await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
            await adapter.on_processing_complete(_event(), ProcessingOutcome.FAILURE)
        finally:
            await bot.shutdown()

    run_async(main())
    calls = _reaction_calls(telegram_api_stub)
    assert "✅" in _emoji_of(calls[0][2])
    assert "❌" in _emoji_of(calls[1][2])


def test_processing_complete_empty_emoji_clears(
        make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch):
    """Empty env value on success: clear the reaction instead of applying 👍."""
    from gateway.platforms.event import ProcessingOutcome

    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)
    monkeypatch.setenv("TELEGRAM_REACTION_SUCCESS", "")

    async def main():
        await bot.initialize()
        try:
            await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
        finally:
            await bot.shutdown()

    run_async(main())
    calls = _reaction_calls(telegram_api_stub)
    assert calls == [(123, 456, None)], f"expected a clear (reaction null), got {calls}"


def _check_processing_complete_cancelled_clears(
        make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch):
    from gateway.platforms.event import ProcessingOutcome

    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)

    async def main():
        await bot.initialize()
        try:
            await adapter.on_processing_complete(_event(), ProcessingOutcome.CANCELLED)
        finally:
            await bot.shutdown()

    run_async(main())
    assert _reaction_calls(telegram_api_stub) == [(123, 456, None)]


def test_on_busy_received_mode_emojis_and_tracking(
        make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch):
    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)

    async def main():
        await bot.initialize()
        try:
            results = {}
            for mode in ("steer", "queue", "interrupt", "unknown_mode"):
                results[mode] = await adapter.on_busy_received(_event(), mode)
            return results
        finally:
            await bot.shutdown()

    results = run_async(main())
    assert all(results.values()), "on_busy_received must return True (suppress text ack)"

    calls = _reaction_calls(telegram_api_stub)
    emojis = [_emoji_of(c[2]) for c in calls]
    assert len(calls) == 4
    assert "\U0001F914" in emojis[0], "steer must react 🤔"
    assert "\U0001F440" in emojis[1], "queue must react 👀"
    assert "⚡" in emojis[2] or "\u26A1" in emojis[2], "interrupt must react ⚡"
    assert "\U0001F440" in emojis[3], "unknown mode falls back to 👀"



def test_on_busy_received_env_override_and_disabled_gate(
        make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch):
    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)
    monkeypatch.setenv("TELEGRAM_REACTION_STEERED", "\U0001f44d")

    async def main():
        await bot.initialize()
        try:
            assert await adapter.on_busy_received(_event(), "steer") is True
            return True
        finally:
            await bot.shutdown()

    run_async(main())
    calls = _reaction_calls(telegram_api_stub)
    assert "\U0001f44d" in _emoji_of(calls[0][2]), "TELEGRAM_REACTION_STEERED must override 🤔"

    # Reactions disabled: no reaction, no suppression (text ack keeps flowing).
    telegram_api_stub.requests.clear()
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)

    async def disabled():
        await bot.initialize()
        try:
            return await adapter.on_busy_received(_event(), "steer")
        finally:
            await bot.shutdown()

    assert run_async(disabled()) is False
    assert _reaction_calls(telegram_api_stub) == []


def test_processing_complete_clears_pending_busy_reactions(
        make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch):
    from gateway.platforms.event import ProcessingOutcome

    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)


    async def main():
        await bot.initialize()
        try:
            await adapter.on_busy_received(_event(message_id="789"), "steer")
            await adapter.on_busy_received(_event(message_id="790"), "queue")
            telegram_api_stub.requests.clear()
            await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
        finally:
            await bot.shutdown()

    run_async(main())
    calls = _reaction_calls(telegram_api_stub)
    clears = [c for c in calls if c[2] is None]
    assert len(clears) == 2, f"pending busy reactions must be cleared, got {calls}"
    assert {(c[0], str(c[1])) for c in clears} == {(123, "789"), (123, "790")}

    assert "\U0001f44d" in _emoji_of(next(c for c in calls if c[2] is not None)[2]), \
        "the trigger message still gets its success reaction"


# ── Unit: defaults, YAML bridge, base-adapter contract ─────────────────────

def test_reaction_emoji_resolution(make_telegram_adapter, monkeypatch):
    """_reaction_emoji: default when unset, env override, explicit empty stays empty."""
    from types import SimpleNamespace

    adapter = make_telegram_adapter(SimpleNamespace(id=1))
    fn = getattr(type(adapter), "_reaction_emoji", None) or getattr(adapter, "_reaction_emoji", None)
    assert fn, "_reaction_emoji is gone"

    monkeypatch.delenv("TELEGRAM_REACTION_START", raising=False)
    assert fn("TELEGRAM_REACTION_START", "\U0001f440") == "\U0001f440"
    monkeypatch.setenv("TELEGRAM_REACTION_START", "\U0001f525")
    assert fn("TELEGRAM_REACTION_START", "\U0001f440") == "\U0001f525"
    monkeypatch.setenv("TELEGRAM_REACTION_START", "")
    assert fn("TELEGRAM_REACTION_START", "\U0001f440") == "", \
        "explicit empty must not fall back to the default"


def test_yaml_reaction_keys_bridge_to_env(monkeypatch):
    """config.yaml telegram.reaction_* keys materialize as TELEGRAM_REACTION_* env vars."""
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    for var in ("TELEGRAM_REACTION_START", "TELEGRAM_REACTION_SUCCESS",
                "TELEGRAM_REACTION_FAILURE", "TELEGRAM_REACTION_STEERED",
                "TELEGRAM_REACTION_QUEUED", "TELEGRAM_REACTION_INTERRUPTED"):
        monkeypatch.delenv(var, raising=False)

    _apply_yaml_config({}, {
        "reaction_start": "\U0001f525",
        "reaction_success": "✅",
        "reaction_failure": "❌",
        "reaction_steer": "\U0001F914",
        "reaction_queue": "\U0001F440",
        "reaction_interrupt": "⚡",
    })

    assert os.environ["TELEGRAM_REACTION_START"] == "\U0001f525"
    assert os.environ["TELEGRAM_REACTION_SUCCESS"] == "✅"
    assert os.environ["TELEGRAM_REACTION_FAILURE"] == "❌"
    assert os.environ["TELEGRAM_REACTION_STEERED"] == "\U0001F914"
    assert os.environ["TELEGRAM_REACTION_QUEUED"] == "\U0001F440"
    assert os.environ["TELEGRAM_REACTION_INTERRUPTED"] == "⚡"


def test_base_adapter_on_busy_received_defaults_to_no_suppression():
    """BasePlatformAdapter.on_busy_received: False by default (text ack survives)."""
    from gateway.platforms.base import BasePlatformAdapter

    fn = getattr(BasePlatformAdapter, "on_busy_received", None)
    assert fn, "BasePlatformAdapter.on_busy_received hook is gone"

    async def main():
        from types import SimpleNamespace

        # The default hook never touches self — bind it to a bare namespace.
        result = await BasePlatformAdapter.on_busy_received(SimpleNamespace(), _event(), "steer")
        assert result is False

    run_async(main())

def test_gateway_busy_followup_reaction_suppresses_only_text_ack(
        make_telegram_bot, make_telegram_adapter, telegram_api_stub, monkeypatch):
    """Deliver into the gateway busy-session entry, preserving hook/ack wiring."""
    from unittest.mock import AsyncMock
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.platforms.event import ProcessingOutcome

    runner = GatewayRunner(GatewayConfig())
    bot, adapter = _prepare(make_telegram_bot, make_telegram_adapter, monkeypatch)
    monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: adapter)
    monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
    monkeypatch.setattr(runner, "_admit_bot_message_for_source", lambda source: True)
    monkeypatch.setattr(runner, "_effective_busy_input_mode", lambda source: "steer")
    monkeypatch.setattr(runner, "_route_plaintext_approval_while_busy", AsyncMock(return_value=False))
    # No running agent: real steer resolution falls back to queue.
    ack = AsyncMock()
    monkeypatch.setattr(runner, "_send_busy_reply", ack)
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_REACTION_QUEUED", "🔥")

    async def main():
        await bot.initialize()
        try:
            assert await runner._handle_active_session_busy_message(_event(message_id="789"), "busy-a")
            assert ack.await_count == 0
            assert "🔥" in _emoji_of(_reaction_calls(telegram_api_stub)[-1][2])
            await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
            assert (123, 789, None) in _reaction_calls(telegram_api_stub)
            telegram_api_stub.requests.clear()
            await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
            assert not any(c[1] == 789 for c in _reaction_calls(telegram_api_stub))
            monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
            assert await runner._handle_active_session_busy_message(_event(message_id="790"), "busy-b")
            assert ack.await_count == 1
        finally:
            await bot.shutdown()
    run_async(main())
