# Local patch contracts

Run from the repo root:

```sh
/Users/tom/.hermes/hermes-agent/venv/bin/python -m pytest tests/patch_contract -q --confcutdir=tests/patch_contract
```

`--confcutdir=tests/patch_contract` is required — upstream's `tests/conftest.py`
poisons the run with 9 spurious failures when it loads.

Layout: one folder per patch (`patch<N>_<subject>/`), shared loopback harness in
`conftest.py`. Set `HERMES_PATCH_REPO` to test another checkout or exported source tree.
Missing patches must cause failures, never skips. Upstream-only guardrails are
companion checks inside patch-sensitive scenarios, not standalone passing tests.

The harness isolates HOME and HERMES_HOME before Hermes imports, removes ambient
credentials/proxies, binds/restores the context-local home for each test, and
rejects non-loopback connections and DNS lookups. Rejected traffic is recorded
and asserted at teardown even if production catches the transport exception.
Launch re-exec is prohibited. Bytecode writes into the checkout are disabled.
Loopback servers retain distinct ports through the session, so tests need no
knowledge of production cache dictionaries. Cache-expiry tests advance a clock
rather than sleeping or replacing private TTL constants.

## Review findings addressed

1. A follow-up enters the real gateway busy-session handler; real Telegram HTTP
   reactions suppress the text acknowledgement. Disabled reactions restore it,
   and completion clears follow-up reactions exactly once.
2. HTTP invokes the real slash compression handler and persistence code with a
   controlled compression-engine result. Both in-place and rotated outcomes are
   checked through the HTTP response and a reopened temporary session store.
   Redundant fake-store persistence tests were removed.
3. Static vision probes record calls and return conflicting verdicts; assertions
   run outside production's exception handlers.
4. An alias changes capabilities after controlled time advancement. Two models
   on one endpoint and the same model on two endpoints exercise cache isolation.
5. Reasoning detection has only an opaque alias with a family-bearing root.
   Remote URLs use an injected recording transport and must never call it.
6. Provider resolution explicitly succeeds or fails; later file rewrites change
   resolved model/endpoint, and a conflicting in-chat override wins.
7. History fallback checks the active profile home, its restoration, and lookup
   of the rotated session ID.
8. Import-time and per-test isolation, outbound rejection, and contextual home
   restoration replace the old opt-in home/Telegram-only environment isolation.
9. Real adapter/runner constructors replace bare objects. Tests observe reaction
   cleanup, equivalent clear payloads, routing fields, HTTP capability discovery,
   and cache behavior rather than object identity or private storage layout.
10. Fresh-process network.env precedence, default/opt-in selective progress
    notice delivery, config-over-endpoint vision precedence, and published
    boolean strings are covered.

## Verification on 2026-10-17

Source: `local-patches` rebased onto `origin/main` (`868b9f244f`, v0.22.2).

* Actual checkout: **53 passed in ~2s** — `venv/bin/python -m pytest
  tests/patch_contract -q --confcutdir=tests/patch_contract`.
* Patch-free `origin/main`: every test fails — no skips, no xfails. The suite
  never passes without the patches.
* Targeted mutations detected: lost busy-ack suppression, lost after_tokens
  forwarding (both persistence modes), endpoint-false precedence, permanent
  vision cache, ignored alias root, lost profile binding, lost
  compression-progress suppression.
* Each patch folder is self-contained (shared helpers live in `conftest.py`), so
  a folder also passes on its own.
* The 2026-10-16 `_TRUE_TOKENS`/`_FALSE_TOKENS` NameError in
  `agent/image_routing.py` is fixed inside patch 2 (boolean token frozensets).
