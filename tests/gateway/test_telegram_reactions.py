"""Tests for Telegram message reactions tied to processing lifecycle hooks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


def _make_adapter(**extra_env):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token")
    adapter._pending_busy_reactions = set()
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    return adapter


def _make_event(chat_id: str = "123", message_id: str = "456") -> MessageEvent:
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


# ── _reactions_enabled ───────────────────────────────────────────────






def test_explicit_env_wins_over_materialized_yaml_default(monkeypatch):
    """TELEGRAM_REACTIONS=true must beat the stock ``reactions: false`` in config.yaml (#109032).

    Fresh installs materialize the whole default config tree, so ``_apply_yaml_config`` seeds
    ``extra["reactions"] = False`` even when the user never chose a value; the reader must still
    honour the explicitly set env var, like ``yaml_env_setter`` documents for the bridge.
    """
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = False
    assert adapter._reactions_enabled() is True


def test_scoped_miss_does_not_leak_default_profile_env(monkeypatch):
    """Under multiplex a scoped miss must not read another profile's process-env value (#72348)."""
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")  # default profile's bridged value
    adapter = _make_adapter()
    adapter.config.extra["reactions"] = False  # this profile's own YAML
    set_multiplex_active(True)
    token = set_secret_scope({"TELEGRAM_BOT_TOKEN": "222:b2"})
    try:
        assert adapter._reactions_enabled() is False
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


@pytest.mark.asyncio
async def test_set_reaction_calls_bot_api(monkeypatch):
    """_set_reaction should call bot.set_message_reaction with correct args."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()

    result = await adapter._set_reaction("123", "456", "\U0001f440")

    assert result is True
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


# ── on_processing_start ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_start_handles_missing_ids(monkeypatch):
    """Should handle events without chat_id or message_id gracefully."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SimpleNamespace(chat_id=None),
        message_id=None,
    )

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_not_awaited()


# ── on_processing_complete ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_complete_success(monkeypatch):
    """Successful processing should set thumbs-up reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_SUCCESS", raising=False)
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44d",
    )


@pytest.mark.asyncio
async def test_on_processing_complete_failure(monkeypatch):
    """Failed processing should set thumbs-down reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_FAILURE", raising=False)
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44e",
    )


@pytest.mark.asyncio
async def test_on_processing_complete_skipped_when_disabled(monkeypatch):
    """Processing complete should not react when reactions are disabled."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_clears_reaction(monkeypatch):
    """Cancelled processing should clear the in-progress reaction.

    Without this clear, the 👀 reaction lingers on the user's message
    indefinitely (until another agent run swaps it for 👍/👎). On a
    ``/stop`` that ends a session, that reaction never gets cleaned up.
    """
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    # set_message_reaction with reaction=None clears all reactions on the
    # message (Bot API documented semantics; equivalent to Bot API 10.0's
    # deleteMessageReaction but works on PTB 22.6 already).
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction=None,
    )


@pytest.mark.asyncio
async def test_clear_reactions_handles_api_error_gracefully(monkeypatch):
    """API errors during clear should not propagate."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot.set_message_reaction = AsyncMock(side_effect=RuntimeError("no perms"))

    result = await adapter._clear_reactions("123", "456")
    assert result is False


@pytest.mark.asyncio
async def test_clear_reactions_returns_false_without_bot(monkeypatch):
    """_clear_reactions should return False when bot is not available."""
    adapter = _make_adapter()
    adapter._bot = None

    result = await adapter._clear_reactions("123", "456")
    assert result is False


# ── on_busy_received ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_busy_received_steer(monkeypatch):
    """Steer mode should set thinking-face reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_busy_received(event, "steer")

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001F914",
    )


@pytest.mark.asyncio
async def test_on_busy_received_queue(monkeypatch):
    """Queue mode should set eyes reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_busy_received(event, "queue")

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001F440",
    )


@pytest.mark.asyncio
async def test_on_busy_received_interrupt(monkeypatch):
    """Interrupt mode should set zap reaction."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_busy_received(event, "interrupt")

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\u26A1",
    )


@pytest.mark.asyncio
async def test_on_busy_received_custom_emoji(monkeypatch):
    """Custom env var should override default emoji."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.setenv("TELEGRAM_REACTION_STEERED", "\U0001f44d")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_busy_received(event, "steer")

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44d",
    )


@pytest.mark.asyncio
async def test_on_busy_received_skipped_when_disabled(monkeypatch):
    """on_busy_received should not react when reactions are disabled."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_busy_received(event, "steer")

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_on_busy_received_unknown_mode_uses_valid_fallback(monkeypatch):
    """Unknown mode should fall back to a valid Telegram reaction emoji."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_busy_received(event, "unknown_mode")

    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001F440",
    )
    assert ("123", "456") in adapter._pending_busy_reactions


@pytest.mark.asyncio
async def test_on_processing_complete_clears_pending_busy_reactions(monkeypatch):
    """When a run finishes, any busy reactions on follow-up messages should be cleared."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    monkeypatch.delenv("TELEGRAM_REACTION_SUCCESS", raising=False)
    adapter = _make_adapter()
    adapter._pending_busy_reactions = {("123", "789"), ("123", "790")}

    await adapter.on_processing_complete(_make_event(message_id="456"), ProcessingOutcome.SUCCESS)

    # Original message gets success reaction
    adapter._bot.set_message_reaction.assert_any_call(
        chat_id=123,
        message_id=456,
        reaction="\U0001f44d",
    )
    # Pending busy reactions are cleared (called with reaction=None)
    clear_calls = [
        c for c in adapter._bot.set_message_reaction.call_args_list
        if c.kwargs.get("reaction") is None
    ]
    assert len(clear_calls) == 2
    assert adapter._pending_busy_reactions == set()


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_clears_pending_busy_too(monkeypatch):
    """Cancellation should clear both the main message and any pending busy reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._pending_busy_reactions = {("123", "789")}

    await adapter.on_processing_complete(_make_event(message_id="456"), ProcessingOutcome.CANCELLED)

    # Main message cleared
    adapter._bot.set_message_reaction.assert_any_call(
        chat_id=123,
        message_id=456,
        reaction=None,
    )
    # Pending busy reaction also cleared
    adapter._bot.set_message_reaction.assert_any_call(
        chat_id=123,
        message_id=789,
        reaction=None,
    )
    assert adapter._pending_busy_reactions == set()


# ── config.py bridging ───────────────────────────────────────────────


def test_config_bridges_telegram_reactions(monkeypatch, tmp_path):
    """gateway/config.py bridges telegram.reactions to TELEGRAM_REACTIONS env var."""
    import hermes_yaml as yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({
        "telegram": {
            "reactions": True,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Use setenv (not delenv) so monkeypatch registers cleanup even when
    # the var doesn't exist yet — load_gateway_config will overwrite it.
    monkeypatch.setenv("TELEGRAM_REACTIONS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_REACTIONS") == "true"
