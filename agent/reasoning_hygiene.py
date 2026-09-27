"""Reasoning-text hygiene for MiMo echo replay and live streams.

MiMo (and other echo-family reasoning models) replay every assistant turn's
stored chain-of-thought back to the provider (``reasoning_content``). Two
failure modes make that replay hazardous on degenerate reasoning:

1. REPLAY POISONING — an aborted mid-reasoning turn stores a degenerate tail
   (e.g. a locked ``hmm — hmm — hmm`` attractor). Replaying it verbatim puts
   the loop text back into context, where register imitation re-triggers it:
   every retry of a poisoned session burns again (2026-09-27 production
   incident: 8.5k-token corpse tail, ~280k tokens burned before the session
   was abandoned).
2. LIVE ONSET — a healthy turn can fall into the same attractor mid-stream
   and generate unbounded degenerate reasoning. The engine-side detector
   handled this; the Hermes-side guard replaces it so serving stays
   policy-free.

Detection is deliberately NARROW: exact consecutive tail cycles only — the
last ``p`` characters repeat byte-identically, ``r`` times in a row. Every
observed incident (hmm-cycles, ``!``-spam, ``@@``-garble) locks into an exact
cycle; fuzzy/near-repeat matching is where false positives come from, and a
false positive here only trims some reasoning text (cheap, wire-only), so
the evidence bar sits well below the engine detector's irreversible-kill
bar: 16 repeats of a ≤96-char cycle ≈ 400 tokens.

Formatting runs and separator cycles are excluded NARROWLY, on BOTH paths:
a streaming false positive kills the whole turn (abort is terminal), so the
single-character bar is high (>=256 identical chars — real spam runs are
thousands of chars, dividers are tens), uniform multi-char units (``00``,
``aa``) are deferred to that same single-character rule so they cannot bypass
it as short "cycles", and cycles built purely from separator characters
(``--``, ``- - -``, ``==``, table rules) are skipped as formatting. Every
OTHER punctuation-only cycle (e.g. the ``@@-@@`` garble class) is treated as
degenerate at DOUBLE evidence (32 repeats) — the proxy's coverage moves into
Hermes instead of being left behind.

The detection window is the last 2k chars (evidence always lives at the
tail); the reported cut is then walked back over the FULL text to the
contiguous cycle's true start, so the trim removes the entire loop even
when it spans tens of thousands of characters (the real corpse was ~20k
chars of cycle).

Scope: MiMo only. Callers gate on ``agent._needs_mimo_tool_reasoning()``;
every other model takes byte-identical code paths.
"""

from __future__ import annotations

# Evidence thresholds: 16 repeats of a <=96-char cycle ≈ 1.5k chars ≈ 400
# tokens — below the engine detector's 768-token kill, and no legitimate
# reasoning ends in 16 byte-identical consecutive repeats of a multi-char
# block. Punctuation-only cycles (no alphanumerics, not uniform, not
# separator formatting) carry double evidence: 32 repeats. Single-character
# runs need 256: real degenerate spam runs are thousands of chars (the
# 2026-09-27 !-spam was ~2.5k), while legitimate dividers/ellipses are tens —
# and a streaming false positive kills the turn rather than trimming text.
P_MAX = 96
R_MIN = 16
PUNCT_R_MIN = 32
SINGLE_MIN = 256
# Detection reads the last WINDOW chars. It must hold FULL evidence for every
# supported period: P_MAX * PUNCT_R_MIN (96 * 32 = 3072) — otherwise a long
# punctuation cycle can never accumulate its 32 repeats inside the window.
WINDOW = 3072

# Narrow formatting exception (no blanket punctuation exclusion):
# cycles built ONLY from these separator characters are markdown / table
# decoration. Anything else punctuation-only (e.g. "@@-@@ ") is judged as a
# degenerate cycle at the higher punctuation threshold.
_FORMATTING_CHARS = frozenset("-=_*~|#>.+ \t·—–")

_STUB = "[reasoning trimmed: degenerate repetition detected]"


def degenerate_tail(
    text: str,
    *,
    p_max: int = P_MAX,
    r_min: int = R_MIN,
    single_min: int = SINGLE_MIN,
    window: int = WINDOW,
) -> tuple[int, int, int] | None:
    """Exact consecutive tail-cycle detection.

    Returns ``(pattern_len, repeats, cut_index)`` where ``cut_index`` is the
    position in ``text`` where the contiguous repeated block begins (walked
    back over the full text, not just the detection window), or ``None``
    when the tail is not degenerate. Only the last ``window`` chars are
    examined for detection; the cut walk is unbounded but only runs on
    already-degenerate text.
    """
    t = text[-window:] if len(text) > window else text
    n = len(t)
    if n < 2:
        return None

    # Single-character cycle: a run of >= single_min identical chars at the tail.
    last = t[-1]
    run = 0
    i = n - 1
    while i >= 0 and t[i] == last:
        run += 1
        i -= 1
    if run >= single_min:
        # Walk the full text back to the run's true start.
        start = len(text) - run
        while start > 0 and text[start - 1] == last:
            start -= 1
        return 1, len(text) - start, start

    for p in range(2, p_max + 1):
        if p * 2 > n:
            break
        unit = t[-p:]
        if unit == unit[0] * p:
            # Uniform unit (any character): a plain character RUN. The
            # single-character rule (>= SINGLE_MIN) owns it — "0"*32 must not
            # sneak past that bar as sixteen "00" repeats.
            continue
        if not any(c.isalnum() for c in unit):
            if all(c in _FORMATTING_CHARS for c in unit):
                # Narrow formatting exception: separator-only cycles are
                # markdown / table decoration, not attractors.
                continue
            # Other punctuation-only cycles (e.g. "@@-@@ "): degenerate class
            # at double evidence.
            threshold = PUNCT_R_MIN
        else:
            threshold = r_min
        reps = 1
        while reps * p + p <= n and t[-p:] == t[-(reps + 1) * p : -reps * p]:
            reps += 1
        if reps >= threshold:
            # Detection proven inside the window; walk the cut back over the
            # full text to the start of the contiguous cycle so the trim
            # removes the whole loop, not just the tail beyond the evidence
            # threshold.
            cut = len(text) - reps * p
            while cut - p >= 0 and text[cut - p : cut] == text[cut : cut + p]:
                cut -= p
            return p, reps, cut
    return None


def sanitize_reasoning_text(text: str) -> str:
    """Trim a degenerate tail to its clean prefix and mark the cut.

    Clean text is returned byte-identical. The trimmed value differs from
    the stored original only on the wire (replay path), and the same stored
    text always sanitizes identically, so the request prefix stays stable
    after the one re-prefill that follows a poisoned turn.
    """
    hit = degenerate_tail(text)
    if hit is None:
        return text
    _, _, cut = hit
    prefix = text[:cut].rstrip()
    return f"{prefix}\n{_STUB}" if prefix else _STUB


class ReasoningLoopGuard:
    """Live-stream onset guard: feed reasoning deltas, abort when degenerate.

    Keeps a rolling tail buffer and checks it only every ``check_every`` new
    characters, so per-chunk cost is O(1) amortized. Returns ``True`` at most
    once per stream, at the first check whose evidence threshold is met.
    """

    def __init__(self, *, check_every: int = 256, window: int = WINDOW):
        self._buf = ""
        self._pending = 0
        self._check_every = check_every
        self._window = window
        self._fired = False

    def feed(self, delta: str) -> bool:
        """Append one reasoning delta; True when a degenerate tail is present."""
        if self._fired or not delta:
            return self._fired
        buf = self._buf + delta
        self._buf = buf[-self._window :] if len(buf) > self._window else buf
        self._pending += len(delta)
        if self._pending < self._check_every:
            return False
        self._pending = 0
        if degenerate_tail(self._buf, window=self._window) is not None:
            self._fired = True
        return self._fired
