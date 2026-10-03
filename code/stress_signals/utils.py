"""Shared utilities: seeding, logging, hashing, manifests, environment detection."""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

TRACKED_LIBRARIES = (
    "numpy", "pandas", "pyarrow", "yaml", "sklearn", "torch", "transformers",
    "datasets", "tokenizers", "accelerate", "pyspark", "openpyxl", "streamlit", "psycopg",
)
MIN_TRAINING_VRAM_GB = 6.0


# ---------------------------------------------------------------- logging
def get_logger(name: str = "stress_signals", level: str | int = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Return a configured logger (idempotent: handlers are not duplicated on re-run)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve() for h in logger.handlers):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------- seeding
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy and (if installed) torch. Optional deterministic cuDNN/algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass


# ---------------------------------------------------------------- hashing / manifests
def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, streamed in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: str | Path) -> dict[str, str]:
    """SHA-256 of every file under a directory, keyed by POSIX relative path (sorted, deterministic)."""
    root = Path(path)
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def library_versions(libs: Iterable[str] = TRACKED_LIBRARIES) -> dict[str, str | None]:
    """Installed versions of tracked libraries (None if not installed)."""
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in libs:
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - any import failure means "not usable"
            out[name] = None
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def write_json(path: str | Path, obj: Any) -> Path:
    """Write JSON atomically (temp file + replace) so interrupted runs never leave half-written files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_manifest(
    path: str | Path,
    artifacts: Iterable[str | Path] = (),
    seeds: Iterable[int] | None = None,
    dataset_versions: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a reproducibility manifest: timestamps, env, library versions, seeds, dataset versions, artifact hashes.

    Directories in `artifacts` are hashed file-by-file. Paths are stored relative to `root` when given.
    """
    root = Path(root).resolve() if root else None
    hashes: dict[str, Any] = {}
    for a in artifacts:
        p = Path(a).resolve()
        key = p.relative_to(root).as_posix() if root and p.is_relative_to(root) else p.as_posix()
        if p.is_dir():
            hashes[key] = {"type": "dir", "files": sha256_dir(p)}
        elif p.is_file():
            hashes[key] = {"type": "file", "sha256": sha256_file(p), "bytes": p.stat().st_size}
        else:
            hashes[key] = {"type": "missing"}
    manifest = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "device": device_label(),
        "library_versions": library_versions(),
        "seeds": list(seeds) if seeds is not None else None,
        "dataset_versions": dataset_versions or {},
        "artifacts": hashes,
        "extra": extra or {},
    }
    write_json(path, manifest)
    return manifest


# ---------------------------------------------------------------- storage
def storage_root(cfg: dict[str, Any]) -> Path:
    """Root that paths.models is joined to: the project root on whatever laptop runs the notebook."""
    return Path(cfg["_root"])


def models_root(cfg: dict[str, Any]) -> Path:
    """<project root>/trained_models (paths.models): every model artefact and the handoff log."""
    return storage_root(cfg) / cfg["paths"]["models"]


def checkpoint_root(cfg: dict[str, Any]) -> Path:
    """Where Trainer checkpoints go: outside trained_models, because the Trainer deletes (rotates) old checkpoints."""
    p = Path(cfg.get("storage", {}).get("checkpoint_root", "data/outputs/checkpoints"))
    return p if p.is_absolute() else Path(cfg["_root"]) / p


# User-downloaded raw datasets (Step 1 audits / Step 2 builds need them; Step 3 needs only Dreaddit from Hugging Face).
RAW_USER_DOWNLOADS = {"sad": ("sad", "local_path"), "mendeley": ("mendeley_stress_indicators", "local_path"),
                      "senticnet": ("senticnet", "local_dir"), "zenodo_pilot": ("zenodo_pilot", "local_dir")}


def raw_data_status(cfg: dict[str, Any]) -> dict[str, bool]:
    """Which manually downloaded raw datasets are on this laptop (a folder counts only if it holds files)."""
    root = Path(cfg["_root"])
    out = {}
    for name, (src, key) in RAW_USER_DOWNLOADS.items():
        p = root / cfg["sources"][src][key]
        out[name] = p.is_file() or (p.is_dir() and any(f.is_file() for f in p.rglob("*")))
    return out


def code_fingerprint(root: str | Path, parts: Iterable[str] = ("code/stress_signals", "code/tests", "code/config.yaml")) -> str:
    """sha256 over the package, tests and config (so the test cell re-runs only after the code changed)."""
    root = Path(root)
    h = hashlib.sha256()
    for part in parts:
        base = root / part
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
        for p in files:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- compute device
def select_device(cfg: dict[str, Any] | None = None) -> str:
    """Device on THIS laptop: the NVIDIA GPU ("cuda") if torch can use one, else "cpu".
    config inference.device (auto|cuda|cpu) can force one; "cuda" without a usable GPU falls back to "cpu"."""
    want = (cfg or {}).get("inference", {}).get("device", "auto")
    try:
        import torch
    except ImportError:
        return "cpu"
    if want == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if want == "cuda":
        _warn_once("inference.device='cuda' but torch sees no NVIDIA GPU on this laptop; using the CPU")
    return "cpu"


def device_label(device: str | None = None) -> str:
    """Human-readable device, e.g. "Quadro P1000 (cuda)" or "cpu". Never raises."""
    try:
        device = device or select_device()
        if device == "cuda":
            import torch
            return f"{torch.cuda.get_device_name(0)} (cuda)"
    except Exception:  # noqa: BLE001 - a broken GPU setup must not break logging
        pass
    return "cpu"


_WARNED: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        get_logger("stress_signals.utils").warning(msg)


def java_version() -> str | None:
    """First line of `java -version` (stderr), or None if Java is not on PATH / JAVA_HOME."""
    java = shutil.which("java")
    if java is None and os.environ.get("JAVA_HOME"):
        cand = Path(os.environ["JAVA_HOME"], "bin", "java.exe" if os.name == "nt" else "java")
        java = str(cand) if cand.exists() else None
    if java is None:
        return None
    try:
        res = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=20)
        text = (res.stderr or res.stdout).strip()
        return text.splitlines()[0] if text else None
    except (OSError, subprocess.SubprocessError):
        return None


def check_hf_datasets_not_shadowed(project_root: str | Path | None = None) -> str | None:
    """Warn-string if `import datasets` resolves to a local folder named datasets/ (on sys.path) instead of the HF library."""
    try:
        mod = importlib.import_module("datasets")
    except ImportError:
        return None
    origin = getattr(mod, "__file__", None)
    if origin is None or not hasattr(mod, "load_dataset"):
        loc = list(getattr(mod, "__path__", [])) or origin
        return (f"`import datasets` resolved to a local folder ({loc}), shadowing the Hugging Face library. "
                "Remove that folder from sys.path or rename it (project data lives under data/raw/).")
    return None


def gpu_supported_by_build(major: int, minor: int, arch_list: Iterable[str]) -> bool:
    """CUDA compatibility rules: sm_XY code runs on any GPU of the same major version with minor >= Y (sm_86 runs on an
    RTX 40 / sm_89); PTX (compute_XY) is JIT-compiled for any GPU with capability >= X.Y."""
    for a in arch_list:
        m = re.fullmatch(r"(sm|compute)_(\d+)(\d)a?", a)
        if not m:
            continue
        kind, amaj, amin = m.group(1), int(m.group(2)), int(m.group(3))
        if kind == "sm" and amaj == major and amin <= minor:
            return True
        if kind == "compute" and (amaj, amin) <= (major, minor):
            return True
    return False


def detect_environment() -> dict[str, Any]:
    """Collect Python/torch/CUDA/GPU/Java facts and hardware warnings. Never raises if torch is missing."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": None,
        "device": "cpu",
        "torch_cuda_build": None,
        "cuda_available": False,
        "arch_list": [],
        "gpu_name": None,
        "gpu_capability": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "java": java_version(),
        "warnings": [],
    }
    w: list[str] = info["warnings"]
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["arch_list"] = list(torch.cuda.get_arch_list())
            info["gpu_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info["gpu_capability"] = f"sm_{major}{minor}"
            free, total = torch.cuda.mem_get_info(0)
            info["vram_free_gb"] = round(free / 1024**3, 2)
            info["vram_total_gb"] = round(total / 1024**3, 2)
            if not gpu_supported_by_build(major, minor, info["arch_list"]):
                w.append(f"{info['gpu_capability']} is NOT in torch.cuda.get_arch_list() -> this torch build "
                         "cannot run kernels on this GPU. Install a build that lists it (or use CPU).")
            if info["gpu_capability"] == "sm_61" and "sm_61" not in info["arch_list"]:
                w.append("sm_61 (Pascal) missing from arch list.")
            if info["vram_total_gb"] < MIN_TRAINING_VRAM_GB:
                w.append(f"GPU VRAM {info['vram_total_gb']} GB < {MIN_TRAINING_VRAM_GB} GB: training works but is slower; "
                         "cell 3.0c uses a small per-step batch (same effective batch).")
        info["device"] = select_device()
        if info["device"] == "cpu":
            w.append("No usable NVIDIA GPU: training runs on the CPU and takes hours per seed (cell 0.0 installs the GPU build "
                     "of torch when the laptop has an NVIDIA GPU).")
    except ImportError:
        w.append("torch not installed: CPU-only utilities only; inference steps need torch.")
    if info["java"] is None:
        w.append("Java not found: PySpark (Steps 9-11) needs a JDK (17 recommended for Spark 3.5) and JAVA_HOME.")
    shadow = check_hf_datasets_not_shadowed()
    if shadow:
        w.append(shadow)
    return info


def model_version_dir(cfg: dict[str, Any], component: str, date: _dt.date | None = None) -> Path:
    """<project root>/trained_models/<component>/v<YYYYMMDD>/."""
    stamp = (date or _dt.date.today()).strftime("%Y%m%d")
    return models_root(cfg) / component / f"v{stamp}"
