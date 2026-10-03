"""Config loading/validation tests (no model code, no training)."""

import copy

import pytest

from stress_signals.config import ConfigError, get_postgres_settings, load_config, validate_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _raw(cfg):
    return {k: copy.deepcopy(v) for k, v in cfg.items() if not k.startswith("_")}


def test_loads_and_resolves_paths(cfg):
    assert cfg["seeds"] == [42, 13, 2024]
    assert cfg["privacy"]["k_min"] == 50
    assert cfg["_paths"]["models"].is_absolute()
    assert cfg["_paths"]["models"] == (cfg["_root"] / "trained_models").resolve()
    assert (cfg["_root"] / "code" / "config.yaml").is_file()


def test_inference_defaults_safe_for_pascal(cfg):
    assert cfg["inference"]["half_precision"] is False
    assert cfg["inference"]["max_vram_gb"] <= 4.0


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("privacy"),
    lambda c: c["privacy"].__setitem__("k_min", 0),
    lambda c: c.__setitem__("seeds", []),
    lambda c: c["chunking"].__setitem__("stride", 999),
    lambda c: c["change_detection"]["ewma"].__setitem__("lambda", 1.5),
    lambda c: c["spark"].__setitem__("master", "yarn"),
    lambda c: c["postgres"].__setitem__("password_env", "hunter2"),
    lambda c: c["models"]["stress"].__setitem__("max_length", 4096),
])
def test_invalid_configs_rejected(cfg, mutate):
    raw = _raw(cfg)
    mutate(raw)
    with pytest.raises(ConfigError):
        validate_config(raw)


def test_postgres_reads_env_only(cfg, monkeypatch):
    monkeypatch.setenv("PG_HOST", "example-host")
    monkeypatch.delenv("PG_PASSWORD", raising=False)
    pg = get_postgres_settings(cfg)
    assert pg["host"] == "example-host"
    assert pg["password"] is None
