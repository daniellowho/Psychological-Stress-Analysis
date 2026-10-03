"""Configuration loading and validation for the stress_signals package."""

from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path
from typing import Any

import yaml

# Layout: <project root>/code/config.yaml; every relative path in the config resolves against the project root.
CONFIG_RELPATH = Path("code") / "config.yaml"

REQUIRED_SECTIONS = (
    "project", "paths", "storage", "seeds", "privacy", "sources", "models",
    "inference", "chunking", "temporal", "change_detection", "spark", "postgres",
)
PATH_KEYS = (
    "data_root", "raw", "interim", "processed", "lake", "gold", "splits",
    "artifacts", "models", "reports", "logs", "manifests", "taxonomy",
)


class ConfigError(ValueError):
    """Raised when config.yaml is missing required keys or holds invalid values."""


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk upwards from `start` (default: cwd) until a directory containing code/config.yaml is found (= project root)."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / CONFIG_RELPATH).is_file():
            return candidate
    raise ConfigError(f"Could not find {CONFIG_RELPATH} above {here}")


def _require(cond: bool, msg: str, errors: list[str]) -> None:
    if not cond:
        errors.append(msg)


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Validate a raw config dict. Returns a list of warnings; raises ConfigError on hard errors."""
    errors: list[str] = []
    warns: list[str] = []

    for section in REQUIRED_SECTIONS:
        _require(section in cfg, f"missing section: {section}", errors)
    if errors:
        raise ConfigError("; ".join(errors))

    paths = cfg["paths"]
    for key in PATH_KEYS:
        _require(isinstance(paths.get(key), str) and paths.get(key), f"paths.{key} must be a non-empty string", errors)

    seeds = cfg["seeds"]
    _require(isinstance(seeds, list) and len(seeds) > 0 and all(isinstance(s, int) for s in seeds),
             "seeds must be a non-empty list of ints", errors)
    _require(cfg.get("primary_seed") in (seeds or []), "primary_seed must be one of seeds", errors)

    k_min = cfg["privacy"].get("k_min")
    _require(isinstance(k_min, int) and k_min >= 1, "privacy.k_min must be an int >= 1", errors)
    if isinstance(k_min, int) and k_min < 50:
        warns.append(f"privacy.k_min={k_min} is below the project default of 50")

    for comp in ("stress", "emotion", "stressor"):
        m = cfg["models"].get(comp)
        _require(isinstance(m, dict), f"models.{comp} missing", errors)
        if isinstance(m, dict):
            _require(isinstance(m.get("base_model"), str) and m["base_model"], f"models.{comp}.base_model required", errors)
            ml = m.get("max_length")
            _require(isinstance(ml, int) and 8 <= ml <= 512, f"models.{comp}.max_length must be int in [8, 512]", errors)

    inf = cfg["inference"]
    _require(inf.get("device") in ("auto", "cpu", "cuda"), "inference.device must be auto|cpu|cuda", errors)
    _require(isinstance(inf.get("batch_size"), int) and inf["batch_size"] >= 1, "inference.batch_size must be int >= 1", errors)
    _require(isinstance(inf.get("half_precision"), bool), "inference.half_precision must be bool", errors)
    if inf.get("half_precision"):
        warns.append("inference.half_precision=true: fp16 inference is slow on older GPUs (e.g. Pascal sm_61)")

    ch = cfg["chunking"]
    _require(isinstance(ch.get("max_tokens"), int) and ch["max_tokens"] > 0, "chunking.max_tokens must be int > 0", errors)
    _require(isinstance(ch.get("stride"), int) and 0 <= ch.get("stride", -1) < ch.get("max_tokens", 0),
             "chunking.stride must satisfy 0 <= stride < max_tokens", errors)
    _require(ch.get("aggregation") in ("mean", "weighted_mean", "max"), "chunking.aggregation must be mean|weighted_mean|max", errors)

    _require(cfg["temporal"].get("granularity") in ("day", "week", "month"),
             "temporal.granularity must be day|week|month", errors)

    cd = cfg["change_detection"]
    lam = cd.get("ewma", {}).get("lambda")
    _require(isinstance(lam, (int, float)) and 0 < lam <= 1, "change_detection.ewma.lambda must be in (0, 1]", errors)
    _require(isinstance(cd.get("ewma", {}).get("L"), (int, float)) and cd["ewma"]["L"] > 0, "change_detection.ewma.L must be > 0", errors)
    _require(isinstance(cd.get("cusum", {}).get("k"), (int, float)) and cd["cusum"]["k"] >= 0, "change_detection.cusum.k must be >= 0", errors)
    _require(isinstance(cd.get("cusum", {}).get("h"), (int, float)) and cd["cusum"]["h"] > 0, "change_detection.cusum.h must be > 0", errors)
    _require(isinstance(cd.get("persistence"), int) and cd["persistence"] >= 1, "change_detection.persistence must be int >= 1", errors)
    _require(isinstance(cd.get("baseline_window"), int) and isinstance(cd.get("min_periods"), int)
             and 1 <= cd.get("min_periods", 0) <= cd.get("baseline_window", 0),
             "change_detection: need 1 <= min_periods <= baseline_window", errors)

    sp = cfg["spark"]
    _require(isinstance(sp.get("master"), str) and sp["master"].startswith("local"),
             "spark.master must be a local[...] master for this project", errors)
    _require(isinstance(sp.get("shuffle_partitions"), int) and sp["shuffle_partitions"] >= 1,
             "spark.shuffle_partitions must be int >= 1", errors)

    pg = cfg["postgres"]
    for key in ("host_env", "port_env", "db_env", "user_env", "password_env"):
        _require(isinstance(pg.get(key), str) and pg[key].isupper(),
                 f"postgres.{key} must be an UPPERCASE env-var name (never a literal value)", errors)

    st = cfg.get("storage", {})
    models = Path(str(paths.get("models", "")))
    _require(not models.is_absolute() and "drive" not in str(models).lower(),
             "paths.models must be a project-relative local folder (no Google Drive)", errors)
    for key in ("checkpoint_root",):
        v = Path(str(st.get(key, "")))
        if "drive" in v.as_posix().lower():
            errors.append(f"storage.{key} must NOT be on Google Drive")
        if models.parts and v.parts[:len(models.parts)] == models.parts:
            errors.append(f"storage.{key} must be outside paths.models (the Trainer deletes old checkpoints)")

    if cfg["sources"].get("main_corpus", {}).get("source") in (None, "UNSET"):
        warns.append("sources.main_corpus.source is UNSET (needed by Step 8)")

    if errors:
        raise ConfigError("Invalid config:\n  - " + "\n  - ".join(errors))
    return warns


def resolve_paths(cfg: dict[str, Any], root: Path) -> dict[str, Path]:
    """Return absolute Paths for every entry of cfg['paths'] (relative entries are joined to root)."""
    out: dict[str, Path] = {}
    for key, value in cfg["paths"].items():
        p = Path(value)
        out[key] = p if p.is_absolute() else (root / p).resolve()
    return out


def load_config(path: str | Path | None = None, validate: bool = True) -> dict[str, Any]:
    """Load config.yaml, validate it, and attach `_root` and `_paths` (absolute Paths).

    `path` may be the YAML file itself or None (auto-discover from cwd; env STRESS_SIGNALS_CONFIG overrides).
    """
    path = path or os.environ.get("STRESS_SIGNALS_CONFIG")
    cfg_file = Path(path).resolve() if path else find_project_root() / CONFIG_RELPATH
    if not cfg_file.is_file():
        raise ConfigError(f"Config file not found: {cfg_file}")
    with cfg_file.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config did not parse to a mapping: {cfg_file}")
    if validate:
        for w in validate_config(cfg):
            warnings.warn(w, stacklevel=2)
    cfg = copy.deepcopy(cfg)
    root = cfg_file.parent.parent             # code/config.yaml -> project root
    cfg["_root"] = root
    cfg["_config_file"] = cfg_file
    cfg["_paths"] = resolve_paths(cfg, root)
    return cfg


def get_postgres_settings(cfg: dict[str, Any]) -> dict[str, str | None]:
    """Read Postgres connection values from the env vars named in config (values are never stored in YAML)."""
    pg = cfg["postgres"]
    return {
        "host": os.environ.get(pg["host_env"]),
        "port": os.environ.get(pg["port_env"]),
        "dbname": os.environ.get(pg["db_env"]),
        "user": os.environ.get(pg["user_env"]),
        "password": os.environ.get(pg["password_env"]),
        "sslmode": pg.get("sslmode", "prefer"),
    }
