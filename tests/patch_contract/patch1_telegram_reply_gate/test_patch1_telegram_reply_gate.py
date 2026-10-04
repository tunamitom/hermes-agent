"""Contract tests — patch 1: feat(telegram): reply_to_bot_triggers knob +
_is_reply_to_bot topic-root filter.

Behavior under contract (plugins/platforms/telegram/adapter.py):
* a forum-topic message whose ``reply_to_message.message_id == message_thread_id``
  is THREADING (the topic-root anchor artifact) — NOT a reply-to-bot;
* a mid-thread reply (``reply_to_message.message_id != message_thread_id``) to a
  bot-authored message IS a reply-to-bot;
* a message carrying a ``quote`` (TextQuote) with text is a deliberate address
  and counts as reply-to-bot even at the topic root;
* the ``reply_to_bot_triggers`` config knob (default true; env
  TELEGRAM_REPLY_TO_BOT_TRIGGERS) gates whether reply-to-bot triggers group
  dispatch in ``_should_process_message`` — off means strict @mention only.

Integration boundary: realistic Telegram Update JSON -> ``Update.de_json`` ->
the adapter's real ``_should_process_message`` process/ignore decision, with a
real python-telegram-bot ``Bot`` initialized against a loopback Bot API stub.
"""

from __future__ import annotations

from conftest import (
    TELEGRAM_BOT_ID,
    TELEGRAM_BOT_USERNAME,
    run_async,
)


def _decide(bot, adapter, payload):
    """Update JSON -> real PTB Message -> the adapter's process/ignore decision."""
    from telegram import Update

    update = Update.de_json(payload, bot)
    message = update.effective_message
    assert message is not None, "Update.de_json did not yield a message"
    return adapter._should_process_message(message)


# ── Integration: Update JSON through _should_process_message ─────────────────

def test_topic_root_reply_is_threading_not_reply_to_bot(
        make_telegram_bot, make_telegram_adapter, make_group_update, bot_reply_message):
    """A reply anchored at the topic root (reply id == thread id) is the forum
    threading artifact and must NOT wake the bot."""
    # Upstream guardrails belong to this patch-sensitive scenario.
    _check_mid_thread_reply_to_bot_triggers(make_telegram_bot, make_telegram_adapter, make_group_update, bot_reply_message)
    _check_quoted_reply_triggers_even_at_topic_root(make_telegram_bot, make_telegram_adapter, make_group_update, bot_reply_message)
    _check_reply_to_human_does_not_trigger(make_telegram_bot, make_telegram_adapter, make_group_update)
    bot = make_telegram_bot()
    adapter = make_telegram_adapter(bot, require_mention=True)

    async def main():
        await bot.initialize()
        try:
            payload = make_group_update(
                thread_id=500, reply_to=bot_reply_message(message_id=500, thread_id=500))
            assert _decide(bot, adapter, payload) is False
        finally:
            await bot.shutdown()

    run_async(main())


def _check_mid_thread_reply_to_bot_triggers(make_telegram_bot, make_telegram_adapter,
                                         make_group_update, bot_reply_message):
    """A mid-thread reply to the bot's own message is a deliberate address: it
    must trigger even under require_mention."""
    bot = make_telegram_bot()
    adapter = make_telegram_adapter(bot, require_mention=True)

    async def main():
        await bot.initialize()
        try:
            payload = make_group_update(
                thread_id=500, reply_to=bot_reply_message(message_id=517, thread_id=500))
            assert _decide(bot, adapter, payload) is True
        finally:
            await bot.shutdown()

    run_async(main())


def _check_quoted_reply_triggers_even_at_topic_root(make_telegram_bot, make_telegram_adapter,
                                                 make_group_update, bot_reply_message):
    """A partial quote (TextQuote) is a deliberate address: counts as reply-to-bot
    even when the anchor is the topic root."""
    bot = make_telegram_bot()
    adapter = make_telegram_adapter(bot, require_mention=True)

    async def main():
        await bot.initialize()
        try:
            payload = make_group_update(
                thread_id=500,
                reply_to=bot_reply_message(message_id=500, thread_id=500),
                quote={"text": "quoted snippet", "position": 0},
            )
            assert _decide(bot, adapter, payload) is True
        finally:
            await bot.shutdown()

    run_async(main())


def test_reply_to_bot_triggers_knob_off_is_strict_mention_only(
        make_telegram_bot, make_telegram_adapter, make_group_update, bot_reply_message):
    """reply_to_bot_triggers=false: a mid-thread reply to the bot must NOT
    dispatch (strict @mention only)."""
    # Upstream guardrails belong to this patch-sensitive scenario.
    _check_direct_bot_mention_still_triggers_with_knob_off(make_telegram_bot, make_telegram_adapter, make_group_update)
    bot = make_telegram_bot()
    adapter = make_telegram_adapter(bot, require_mention=True, reply_to_bot_triggers=False)

    async def main():
        await bot.initialize()
        try:
            payload = make_group_update(
                thread_id=500, reply_to=bot_reply_message(message_id=517, thread_id=500))
            assert _decide(bot, adapter, payload) is False
        finally:
            await bot.shutdown()

    run_async(main())


def test_knob_off_also_vetoes_quoted_reply(make_telegram_bot, make_telegram_adapter,
                                           make_group_update, bot_reply_message):
    """The knob gates the whole reply-to-bot route: a quote-reply to the bot is
    equally ignored when the knob is off."""
    bot = make_telegram_bot()
    adapter = make_telegram_adapter(bot, require_mention=True, reply_to_bot_triggers=False)

    async def main():
        await bot.initialize()
        try:
            payload = make_group_update(
                thread_id=500,
                reply_to=bot_reply_message(message_id=517, thread_id=500),
                quote={"text": "ping", "position": 0},
            )
            assert _decide(bot, adapter, payload) is False
        finally:
            await bot.shutdown()

    run_async(main())


def _check_reply_to_human_does_not_trigger(make_telegram_bot, make_telegram_adapter,
                                        make_group_update):
    """A reply whose parent is another human is not an address to the bot."""
    bot = make_telegram_bot()
    adapter = make_telegram_adapter(bot, require_mention=True)

    async def main():
        await bot.initialize()
        try:
            human_parent = {
                "message_id": 517,
                "date": 1759999000,
                "chat": {"id": -1001234567890, "type": "supergroup",
                         "title": "Ops", "is_forum": True},
                "from": {"id": 888, "is_bot": False, "first_name": "Bob"},
                "message_thread_id": 500,
                "is_topic_message": True,
                "text": "what do you think?",
            }
            payload = make_group_update(thread_id=500, reply_to=human_parent)
            assert _decide(bot, adapter, payload) is False
        finally:
            await bot.shutdown()

    run_async(main())


def _check_direct_bot_mention_still_triggers_with_knob_off(
        make_telegram_bot, make_telegram_adapter, make_group_update):
    """Knob off restricts to explicit @mention — the mention route must survive."""
    bot = make_telegram_bot()
    adapter = make_telegram_adapter(bot, require_mention=True, reply_to_bot_triggers=False)

    async def main():
        await bot.initialize()
        try:
            payload = make_group_update(
                thread_id=500,
                text=f"hey @{TELEGRAM_BOT_USERNAME} please look",
            )
            # Attach a server-side mention entity covering the @handle (the
            # adapter trusts MessageEntity spans, not raw substrings).
            text = payload["message"]["text"]
            start = text.index("@")
            payload["message"]["entities"] = [{
                "type": "mention", "offset": start,
                "length": len(f"@{TELEGRAM_BOT_USERNAME}"),
            }]
            assert _decide(bot, adapter, payload) is True
        finally:
            await bot.shutdown()

    run_async(main())


# ── Unit: the patched predicates and knob ───────────────────────────────────

def _bare_message(**attrs):
    from types import SimpleNamespace

    defaults = dict(
        text="hello", caption=None, entities=[], caption_entities=[],
        message_thread_id=500,
    )
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def _bot_parent(message_id, thread_id=500):
    from types import SimpleNamespace

    return SimpleNamespace(
        message_id=message_id,
        from_user=SimpleNamespace(id=TELEGRAM_BOT_ID, is_bot=True,
                                 username=TELEGRAM_BOT_USERNAME),
        message_thread_id=thread_id,
    )


def _unit_bot():
    """Stand-in bot identity for pure-predicate tests (no HTTP needed)."""
    from types import SimpleNamespace

    return SimpleNamespace(id=TELEGRAM_BOT_ID, username=TELEGRAM_BOT_USERNAME)


def test_is_reply_to_bot_topic_root_vs_mid_thread(make_telegram_adapter):
    """_is_reply_to_bot: topic-root anchor => False; mid-thread => True; quote
    with text beats the anchor; quote with empty text does not."""
    adapter = make_telegram_adapter(_unit_bot())
    assert getattr(adapter, "_is_reply_to_bot", None), "_is_reply_to_bot is gone"

    # Topic-root anchor: reply_to.message_id == message_thread_id -> threading.
    msg = _bare_message(reply_to_message=_bot_parent(message_id=500), message_thread_id=500)
    assert adapter._is_reply_to_bot(msg) is False

    # Mid-thread reply -> reply-to-bot.
    msg = _bare_message(reply_to_message=_bot_parent(message_id=517), message_thread_id=500)
    assert adapter._is_reply_to_bot(msg) is True

    # Quote with text at the topic root -> deliberate address.
    from types import SimpleNamespace
    msg = _bare_message(reply_to_message=_bot_parent(message_id=500), message_thread_id=500,
                        quote=SimpleNamespace(text="quoted"))
    assert adapter._is_reply_to_bot(msg) is True

    # Quote present but empty text -> falls through to the anchor filter.
    msg = _bare_message(reply_to_message=_bot_parent(message_id=500), message_thread_id=500,
                        quote=SimpleNamespace(text=""))
    assert adapter._is_reply_to_bot(msg) is False

    # No thread id to compare against -> counts as reply-to-bot.
    msg = _bare_message(reply_to_message=_bot_parent(message_id=517), message_thread_id=None)
    assert adapter._is_reply_to_bot(msg) is True

    # Not a reply at all / reply to a non-bot -> False.
    assert adapter._is_reply_to_bot(_bare_message(reply_to_message=None)) is False
    from types import SimpleNamespace as NS
    msg = _bare_message(reply_to_message=NS(message_id=517,
                                            from_user=NS(id=777, is_bot=False)))
    assert adapter._is_reply_to_bot(msg) is False


def test_reply_to_bot_triggers_knob_resolution(make_telegram_adapter, monkeypatch):
    """_telegram_reply_to_bot_triggers: default true; YAML/extra false; scoped
    env var override."""
    adapter = make_telegram_adapter(_unit_bot())
    knob = getattr(adapter, "_telegram_reply_to_bot_triggers", None)
    assert knob, "_telegram_reply_to_bot_triggers knob is gone"

    assert knob() is True  # default

    adapter.config.extra = {"reply_to_bot_triggers": False}
    assert knob() is False

    adapter.config.extra = {"reply_to_bot_triggers": "false"}
    assert knob() is False

    adapter.config.extra = {}
    monkeypatch.setenv("TELEGRAM_REPLY_TO_BOT_TRIGGERS", "false")
    assert knob() is False

    monkeypatch.setenv("TELEGRAM_REPLY_TO_BOT_TRIGGERS", "true")
    adapter.config.extra = {"reply_to_bot_triggers": False}
    assert knob() is True  # explicit env beats YAML
