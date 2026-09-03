"""Contract tests for capability-aware ordered auxiliary vision routing."""

from __future__ import annotations

import pytest

from conftest import run_async  # (sys.path bootstrap side effect)


_BASE_URL = "http://router.test/v1"


def _chain():
    return [
        {
            "provider": "custom",
            "model": model,
            "base_url": _BASE_URL,
            "api_key": f"{model}-key",
        }
        for model in ("fast", "smart", "backup")
    ]


def _mock_auto_routing(monkeypatch, verdicts):
    import agent.auxiliary_client as ac
    import agent.image_routing as ir

    chain = _chain()
    lookups = []
    builds = []
    clients = {model: object() for model in ("fast", "smart", "backup")}

    monkeypatch.setattr(
        ac, "_get_auxiliary_task_config", lambda task: {"fallback_chain": chain}
    )
    monkeypatch.setattr(
        ac, "_fallback_entry_api_key", lambda entry: entry.get("api_key")
    )

    def lookup(base_url, model, api_key=""):
        lookups.append((base_url, model, api_key))
        return verdicts[model]

    def build(entry):
        model = entry["model"]
        builds.append(model)
        return clients[model], model

    def legacy_route_must_not_run(*args):
        raise AssertionError("legacy auto routing ran before the configured chain")

    monkeypatch.setattr(ir, "_lookup_endpoint_supports_vision", lookup)
    monkeypatch.setattr(ac, "_resolve_fallback_entry", build)
    monkeypatch.setattr(ac, "_vision_main_provider_client", legacy_route_must_not_run)
    return ac, chain, lookups, builds, clients


@pytest.mark.parametrize(
    "verdicts, expected_model, expected_lookups",
    [
        ({"fast": True, "smart": True, "backup": True}, "fast", ["fast"]),
        ({"fast": False, "smart": True, "backup": True}, "smart", ["fast", "smart"]),
        (
            {"fast": False, "smart": False, "backup": True},
            "backup",
            ["fast", "smart", "backup"],
        ),
    ],
)
def test_auto_vision_prefers_first_endpoint_published_capable_chain_entry(
    monkeypatch, verdicts, expected_model, expected_lookups
):
    ac, _chain_entries, lookups, builds, clients = _mock_auto_routing(
        monkeypatch, verdicts
    )

    result = ac._vision_auto_route(
        {"provider": "custom", "model": "main"}, None, None, False
    )

    assert result == ("custom", clients[expected_model], expected_model)
    assert [model for _base, model, _key in lookups] == expected_lookups
    assert builds == [expected_model]
    assert all(base == _BASE_URL for base, _model, _key in lookups)
    assert [key for _base, _model, key in lookups] == [
        f"{model}-key" for model in expected_lookups
    ]


def test_auto_vision_false_and_unknown_entries_fall_through_to_legacy_auto(monkeypatch):
    ac, _chain_entries, lookups, builds, _clients = _mock_auto_routing(
        monkeypatch, {"fast": False, "smart": None, "backup": False}
    )
    legacy_client = object()
    monkeypatch.setattr(
        ac,
        "_vision_main_provider_client",
        lambda *args: (legacy_client, "legacy-vision"),
    )

    result = ac._vision_auto_route(
        {"provider": "custom", "model": "main"}, None, None, False
    )

    assert result == ("custom", legacy_client, "legacy-vision")
    assert [model for _base, model, _key in lookups] == ["fast", "smart", "backup"]
    assert builds == []


def test_vision_request_fallback_skips_known_false_but_preserves_unknown(monkeypatch):
    ac, _chain_entries, lookups, builds, clients = _mock_auto_routing(
        monkeypatch, {"fast": False, "smart": None, "backup": True}
    )
    monkeypatch.setattr(
        ac, "_failed_backend_skip", lambda *args, **kwargs: lambda *args, **kwargs: False
    )
    monkeypatch.setattr(ac, "_is_provider_unhealthy", lambda *args: False)

    result = ac._try_configured_fallback_chain(
        "vision", "openai", failed_model="failed-model"
    )

    assert result == (clients["smart"], "smart", "fallback_chain[1](custom)")
    assert [model for _base, model, _key in lookups] == ["fast", "smart"]
    assert builds == ["smart"]
