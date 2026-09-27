# SPDX-License-Identifier: Apache-2.0
"""Reasoning hygiene: degenerate-tail detection, MiMo replay trim, stream guard.

Contract tests for the 2026-09-27 reasoning-loop incident class: replayed
poisoned reasoning and live onset must both be neutralized, and every
non-MiMo path must stay byte-identical.
"""

from agent.message_sanitization import apply_reasoning_content_policy, reapply_reasoning_echo
from agent.reasoning_hygiene import (
    ReasoningLoopGuard,
    degenerate_tail,
    sanitize_reasoning_text,
)

# The production incident's attractor unit (hmm-loop corpse, 2026-09-27).
HMM_UNIT = "Hmm — hmm — hmm — hmm — hmm. Hmm: hmm — hmm. "
CLEAN_REASONING = (
    "Let me work through this. The user wants a budget report. "
    "First I need the quarterly numbers, then I compare them against "
    "the forecast. Let me check the spreadsheet data before answering."
)


def _poisoned_reasoning(clean_prefix_chars: int = 400) -> str:
    head = (CLEAN_REASONING + " ") * (clean_prefix_chars // len(CLEAN_REASONING) + 1)
    return head[:clean_prefix_chars] + HMM_UNIT * 60


# ── degenerate_tail ────────────────────────────────────────────────────────


def test_detects_hmm_loop_tail():
    hit = degenerate_tail(_poisoned_reasoning())
    assert hit is not None
    p, reps, cut = hit
    assert 2 <= p <= 96
    assert reps >= 16
    # The cut is walked back to the cycle's true start: the clean prefix.
    assert cut == 400


def test_clean_prose_is_not_degenerate():
    assert degenerate_tail(CLEAN_REASONING) is None


def test_single_char_spam_is_degenerate():
    text = CLEAN_REASONING + "!" * 300
    hit = degenerate_tail(text)
    assert hit is not None
    p, reps, cut = hit
    assert p == 1 and reps >= 256
    assert cut == len(CLEAN_REASONING)  # walked back to the run's start


def test_markdown_divider_and_ellipsis_are_not_degenerate():
    # Dividers and ellipses are tens of chars — far under the 256 single-run
    # floor — and their multi-char "cycles" contain no alphanumeric char, so
    # the p-scan skips them as separators. Probe case: an
    # 80-char divider must NOT fire (a streaming false positive kills the
    # turn, it does not trim).
    assert degenerate_tail("Section summary follows.\n" + "-" * 40) is None
    assert degenerate_tail("Section summary follows.\n" + "-" * 80) is None
    assert degenerate_tail("Section summary follows.\n" + "-" * 120) is None
    assert degenerate_tail("waiting" + "." * 40) is None
    assert degenerate_tail("rule" + "=" * 60) is None
    assert degenerate_tail("sep\n" + "- " * 40) is None  # dash-space separator
    assert degenerate_tail("table\n" + "|---|" * 20) is None


def test_long_single_char_spam_fires():
    # Real degenerate spam runs are thousands of chars (the 2026-09-27
    # !-spam was ~2.5k): a 300-char run fires; a 100-char burst does not.
    hit = degenerate_tail(CLEAN_REASONING + "!" * 300)
    assert hit is not None and hit[0] == 1
    assert degenerate_tail(CLEAN_REASONING + "!" * 100) is None


def test_uniform_runs_cannot_bypass_single_char_bar():
    """Review #7: "0"*32 used to fire as sixteen "00" repeats, under the
    256-char single-run bar. Uniform units are deferred to the single-char
    rule for zeros, letters, any character."""
    assert degenerate_tail("0" * 32) is None
    assert degenerate_tail("a" * 100) is None
    assert degenerate_tail("z" * 40) is None
    # At run length the bar is met, and it fires as a single-char cycle.
    hit = degenerate_tail(CLEAN_REASONING + "0" * 300)
    assert hit is not None and hit[0] == 1


def test_punctuation_garble_fires_at_double_evidence():
    """The @@-garble incident class fires in Hermes: no blanket
    punctuation exclusion. Punctuation-only cycles carry double evidence —
    32 repeats — so heavy garble fires while short bursts do not."""
    hit = degenerate_tail("@@-@@ " * 100)
    assert hit is not None and hit[0] == 6 and hit[1] >= 32
    assert degenerate_tail("@@-@@ " * 20) is None


def test_punctuation_window_covers_largest_supported_period():
    """WINDOW = P_MAX * PUNCT_R_MIN, so every supported punctuation period can
    accumulate its full 32-repeat evidence inside the window. Periods 65-96
    exceeded a 2048-char window and could never fire."""
    p65 = "@" + "!" * 64   # 65-char period: 32 reps = 2080 chars
    assert degenerate_tail(p65 * 32) is not None
    p67 = "@" + "!" * 66   # period 67: returned None under the 2048 window
    assert degenerate_tail(p67 * 100) is not None
    p96 = "@$" + "!" * 94  # 96-char period (P_MAX): 32 reps = 3072 exactly
    assert degenerate_tail(p96 * 33) is not None


def test_separator_cycles_are_formatting_not_degenerate():
    # Cycles built only from separator characters are markdown / table
    # decoration (narrow formatting exception).
    assert degenerate_tail("sep\n" + "- " * 40) is None   # dash-space separator
    assert degenerate_tail("table\n" + "|---|" * 20) is None
    assert degenerate_tail("rule\n" + "== " * 30) is None
    assert degenerate_tail("quote\n" + "— " * 30) is None  # em-dash divider


def test_short_repetition_burst_is_not_degenerate():
    # A chorus or refrain repeated a few times is legitimate writing.
    assert degenerate_tail("let it be " * 8 + "whisper words of wisdom") is None


def test_long_clean_head_does_not_mask_tail():
    """Detection reads only the tail window, and the cut walks the FULL text:
    a degenerate cycle is found and excised completely even behind a huge
    clean head (the real corpse carried ~20k chars of cycle)."""
    head = (CLEAN_REASONING + " ") * 120          # ~28k chars of clean prose
    head = head[:20000]
    text = head + HMM_UNIT * 60
    hit = degenerate_tail(text)
    assert hit is not None
    assert hit[2] == len(head)


# ── sanitize_reasoning_text ────────────────────────────────────────────────


def test_sanitize_trims_and_marks():
    text = _poisoned_reasoning()
    out = sanitize_reasoning_text(text)
    assert HMM_UNIT * 10 not in out
    assert "[reasoning trimmed" in out
    assert out.startswith(text[:100])  # clean prefix preserved


def test_sanitize_all_cycle_text_yields_stub_only():
    out = sanitize_reasoning_text(HMM_UNIT * 60)
    assert out == "[reasoning trimmed: degenerate repetition detected]"


def test_sanitize_clean_text_is_byte_identical():
    assert sanitize_reasoning_text(CLEAN_REASONING) == CLEAN_REASONING


# ── replay policy ──────────────────────────────────────────────────────────


def _assistant_msg(reasoning: str) -> dict:
    return {"role": "assistant", "content": "answer", "reasoning": reasoning}


def test_policy_trims_poisoned_reasoning_for_mimo():
    src = _assistant_msg(_poisoned_reasoning())
    api = {"role": "assistant", "content": "answer"}
    apply_reasoning_content_policy(src, api, needs_thinking_pad=True, sanitize_mimo=True)
    assert HMM_UNIT * 10 not in api["reasoning_content"]
    assert "[reasoning trimmed" in api["reasoning_content"]


def test_policy_replays_verbatim_without_mimo_flag():
    """Every other model / family path must be byte-identical (no sanitize)."""
    poisoned = _poisoned_reasoning()
    src = _assistant_msg(poisoned)
    api = {"role": "assistant", "content": "answer"}
    apply_reasoning_content_policy(src, api, needs_thinking_pad=True, sanitize_mimo=False)
    assert api["reasoning_content"] == poisoned


def test_policy_trims_poisoned_reasoning_content_primary_format():
    """Hermes stores streamed reasoning in reasoning_content (the primary
    format); the sanitizer must cover it, not only the 'reasoning' fallback
    (the original incident replayed this format verbatim)."""
    poisoned = _poisoned_reasoning()
    src = {"role": "assistant", "content": "answer", "reasoning_content": poisoned}
    api = {"role": "assistant", "content": "answer"}
    apply_reasoning_content_policy(src, api, needs_thinking_pad=True, sanitize_mimo=True)
    assert HMM_UNIT * 10 not in api["reasoning_content"]
    assert "[reasoning trimmed" in api["reasoning_content"]
    assert src["reasoning_content"] == poisoned  # stored history untouched


def test_policy_primary_format_verbatim_without_mimo_flag():
    poisoned = _poisoned_reasoning()
    src = {"role": "assistant", "content": "answer", "reasoning_content": poisoned}
    api = {"role": "assistant", "content": "answer"}
    apply_reasoning_content_policy(src, api, needs_thinking_pad=True, sanitize_mimo=False)
    assert api["reasoning_content"] == poisoned


def test_policy_space_pad_survives_sanitization():
    """The require-side space pad must pass through byte-identical."""
    src = {"role": "assistant", "content": "a", "tool_calls": [{"id": "x"}],
           "reasoning": "from another provider"}
    api = {"role": "assistant", "content": "a", "tool_calls": [{"id": "x"}]}
    apply_reasoning_content_policy(src, api, needs_thinking_pad=True, sanitize_mimo=True)
    assert api["reasoning_content"] == " "


def test_policy_source_message_is_never_mutated():
    poisoned = _poisoned_reasoning()
    src = _assistant_msg(poisoned)
    api = {"role": "assistant", "content": "answer"}
    apply_reasoning_content_policy(src, api, needs_thinking_pad=True, sanitize_mimo=True)
    assert src["reasoning"] == poisoned  # stored session history untouched


def test_reapply_reasoning_echo_threads_sanitize():
    msgs = [{"role": "assistant", "content": "a", "reasoning": _poisoned_reasoning()}]
    changed = reapply_reasoning_echo(msgs, needs_thinking_pad=True, sanitize_mimo=True)
    assert changed == 1
    assert "[reasoning trimmed" in msgs[0]["reasoning_content"]


def test_reapply_trims_existing_poisoned_reasoning_content():
    """The reapply path previously skipped turns whose reasoning_content was
    already present — poison on already-built api_messages went through
    unchanged. It must sanitize present values too."""
    msgs = [{"role": "assistant", "content": "a", "reasoning_content": _poisoned_reasoning()}]
    changed = reapply_reasoning_echo(msgs, needs_thinking_pad=True, sanitize_mimo=True)
    assert changed == 1
    assert "[reasoning trimmed" in msgs[0]["reasoning_content"]


def test_reapply_leaves_pads_and_clean_values_untouched():
    msgs = [
        {"role": "assistant", "content": "a", "reasoning_content": " "},
        {"role": "assistant", "content": "b", "reasoning_content": CLEAN_REASONING},
    ]
    changed = reapply_reasoning_echo(msgs, needs_thinking_pad=True, sanitize_mimo=True)
    assert changed == 0
    assert msgs[0]["reasoning_content"] == " "
    assert msgs[1]["reasoning_content"] == CLEAN_REASONING


def test_reapply_reasoning_echo_default_untouched():
    poisoned = _poisoned_reasoning()
    msgs = [{"role": "assistant", "content": "a", "reasoning": poisoned}]
    reapply_reasoning_echo(msgs, needs_thinking_pad=True)
    assert msgs[0]["reasoning_content"] == poisoned


# ── live-stream guard ──────────────────────────────────────────────────────


def test_guard_fires_on_accumulated_loop():
    guard = ReasoningLoopGuard()
    for i in range(80):
        if guard.feed(HMM_UNIT):
            break
    assert guard.feed(HMM_UNIT) or guard._fired  # fires within evidence + overshoot
    # Fires at most once per stream; later feeds stay quiet.
    assert guard.feed(HMM_UNIT * 5) is True  # latched


def test_guard_stays_silent_on_clean_stream():
    guard = ReasoningLoopGuard()
    pieces = [CLEAN_REASONING[i : i + 97] for i in range(0, len(CLEAN_REASONING) * 20, 97)]
    assert not any(guard.feed(p) for p in pieces)


def test_guard_fires_on_single_char_spam_stream():
    guard = ReasoningLoopGuard()
    assert not guard.feed(CLEAN_REASONING)
    fired = False
    for _ in range(40):
        if guard.feed("!!!!!!!!!!!!!!!!!"):
            fired = True
            break
    assert fired


def test_guard_detects_real_corpse_shape():
    """The exact unit from the 20260927-082712 corpse fires the guard."""
    guard = ReasoningLoopGuard()
    seed = "Okay, the user asked about restaurants. I should list options. "
    assert not guard.feed(seed)
    fired = False
    for _ in range(60):
        if guard.feed(HMM_UNIT):
            fired = True
            break
    assert fired


# ── abort response & post-hoc completed-response hygiene ──────────────────

from types import SimpleNamespace  # noqa: E402

from agent.chat_completion_helpers import (  # noqa: E402
    REASONING_LOOP_ABORT_NOTICE,
    _reasoning_loop_abort_response,
    _sanitize_completed_response_reasoning,
    _stream_final_text,
)


class _MockMimoAgent:
    model = "mimo-pro"
    _interrupt_requested = False

    def _needs_mimo_tool_reasoning(self):
        return True


class _MockOtherAgent(_MockMimoAgent):
    def _needs_mimo_tool_reasoning(self):
        return False


def _completed_response(reasoning: str | None, content: str | None,
                        details: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=None,
                              reasoning_content=reasoning, reasoning_details=details,
                              refusal=None)
    return SimpleNamespace(id="x", model="mimo-pro", usage=None, provider=None,
                           choices=[SimpleNamespace(index=0, message=message,
                                                    finish_reason="stop")])


def test_abort_response_shape():
    """The detector-aborted outcome is ONE terminal response: the notice is
    the answer, no tool calls, no reasoning, terminal finish
    no success event followed by an interruption, nothing retryable)."""
    resp = _reasoning_loop_abort_response("assistant", "mimo-pro", None)
    msg = resp.choices[0].message
    assert msg.content == REASONING_LOOP_ABORT_NOTICE
    assert msg.tool_calls is None
    assert msg.reasoning_content is None
    assert resp.choices[0].finish_reason == "stop"
    # The stream-end emitter surfaces the notice as the final text.
    assert _stream_final_text(resp) == REASONING_LOOP_ABORT_NOTICE


def test_post_hoc_drops_degenerate_reasoning_and_surfaces_notice():
    resp = _completed_response(_poisoned_reasoning(), content=None)
    assert _sanitize_completed_response_reasoning(_MockMimoAgent(), resp) is True
    msg = resp.choices[0].message
    assert msg.reasoning_content is None
    assert msg.content == REASONING_LOOP_ABORT_NOTICE


def test_post_hoc_keeps_visible_content():
    resp = _completed_response(_poisoned_reasoning(), content="partial answer")
    assert _sanitize_completed_response_reasoning(_MockMimoAgent(), resp) is True
    msg = resp.choices[0].message
    assert msg.reasoning_content is None
    assert msg.content == "partial answer"


def test_post_hoc_clean_response_untouched():
    resp = _completed_response(CLEAN_REASONING, content="fine")
    assert _sanitize_completed_response_reasoning(_MockMimoAgent(), resp) is False
    assert resp.choices[0].message.reasoning_content == CLEAN_REASONING


def test_post_hoc_non_mimo_is_noop():
    resp = _completed_response(_poisoned_reasoning(), content=None)
    before = resp.choices[0].message.reasoning_content
    assert _sanitize_completed_response_reasoning(_MockOtherAgent(), resp) is False
    assert resp.choices[0].message.reasoning_content == before


def test_post_hoc_none_and_odd_shapes_are_safe():
    assert _sanitize_completed_response_reasoning(_MockMimoAgent(), None) is False
    assert _sanitize_completed_response_reasoning(_MockMimoAgent(), object()) is False


def test_post_hoc_details_only_completed_response_is_checked():
    """A completed response carrying reasoning ONLY as readable reasoning_details
    entries is judged with the same extraction the streaming display uses."""
    details = [{"type": "reasoning.text", "text": _poisoned_reasoning()}]
    resp = _completed_response(None, content=None, details=details)
    assert _sanitize_completed_response_reasoning(_MockMimoAgent(), resp) is True
    msg = resp.choices[0].message
    assert msg.reasoning_content is None
    assert msg.reasoning_details is None
    assert msg.content == REASONING_LOOP_ABORT_NOTICE


def test_post_hoc_opaque_details_are_never_judged():
    # Encrypted/signature detail entries carry no readable text: never judged,
    # never altered.
    details = [{"type": "reasoning.encrypted", "data": "AAAA" * 400}]
    resp = _completed_response(None, content=None, details=details)
    assert _sanitize_completed_response_reasoning(_MockMimoAgent(), resp) is False
    assert resp.choices[0].message.reasoning_details == details


# ── abort delivery through the real _StreamingCall.run() ──────────────────

import pytest  # noqa: E402


def _run_streaming_call(agent, response, *, monitor):
    """Execute the REAL _StreamingCall.run() mechanics: worker thread completes
    instantly with *response*; *monitor(call)* stands in for the inline monitor
    loop (interrupt observation)."""
    from agent.chat_completion_helpers import _StreamingCall

    call = _StreamingCall(agent, {"model": "mimo-pro", "messages": []}, None)
    call._resolve_stale_timeout = lambda: None

    def _finish():
        call.result["response"] = response
        call._call_done.set()

    call._run_call = _finish
    call._monitor_loop = lambda: monitor(call)
    return call.run()


def test_run_delivers_loop_abort_response():
    """The detector-aborted turn REACHES the conversation: run() returns the
    abort response (the notice) because the guard sets no interrupt flag."""
    agent = _MockMimoAgent()
    resp = _reasoning_loop_abort_response("assistant", "mimo-pro", None)
    out = _run_streaming_call(agent, resp, monitor=lambda call: None)
    assert out is resp
    assert out.choices[0].message.content == REASONING_LOOP_ABORT_NOTICE
    # The detector abort must not masquerade as a user interruption.
    assert agent._interrupt_requested is False


def test_run_user_interrupt_monitor_first_discards_abort_response():
    """A genuine user interrupt observed by the monitor outranks the detector:
    run() raises and the abort response is discarded."""
    agent = _MockMimoAgent()
    resp = _reasoning_loop_abort_response("assistant", "mimo-pro", None)

    def monitor(call):
        call._monitor_interrupted["yes"] = True

    with pytest.raises(InterruptedError):
        _run_streaming_call(agent, resp, monitor=monitor)


def test_run_user_interrupt_flag_post_worker_discards_abort_response():
    """A genuine user interrupt flag set while the worker completed also
    discards the abort response (run's post-worker check)."""
    agent = _MockMimoAgent()
    agent._interrupt_requested = True
    resp = _reasoning_loop_abort_response("assistant", "mimo-pro", None)
    with pytest.raises(InterruptedError, match="post-worker"):
        _run_streaming_call(agent, resp, monitor=lambda call: None)
