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
        "is_colab": is_colab(),
        "library_versions": library_versions(),
        "seeds": list(seeds) if seeds is not None else None,
        "dataset_versions": dataset_versions or {},
        "artifacts": hashes,
        "extra": extra or {},
    }
    write_json(path, manifest)
    return manifest


# ---------------------------------------------------------------- environment
def is_colab() -> bool:
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def storage_root(cfg: dict[str, Any]) -> Path:
    """Root that paths.models is joined to: the project root on BOTH kernels (no Google Drive). Locally that is the PC
    (E:/.../TA-BDA); on Colab it is /content/TA-BDA on the server's own disk, which is temporary, so cell 3G packs
    trained_models/ into colab_results.zip for the PC (stress_signals.handoff.pack_colab_results)."""
    return Path(cfg["_root"])


def models_root(cfg: dict[str, Any]) -> Path:
    """<project root>/trained_models (paths.models): every model artefact and the handoff log."""
    return storage_root(cfg) / cfg["paths"]["models"]


class LocalKernelOnly(RuntimeError):
    """Raised by notebook cells that must run on the local kernel (they need data that stays on the PC)."""


def require_local_kernel(what: str) -> None:
    """Stop a notebook cell with a clear message when it is run on a Colab kernel. Steps 0-2 (raw data, audits,
    preprocessing) and building the Colab bundle run locally; the Colab kernel only trains (Step 3)."""
    if is_colab():
        raise LocalKernelOnly(f"{what} runs on the LOCAL kernel only (its data stays on your PC). On the Colab kernel run "
                              "only 0.0 -> 0.1 -> 3.0a -> 3.1 -> 3A.1 -> 3B.1 (optional 3E.2) -> 3G, then switch back to local.")


COLAB_BUNDLE_NAME = "colab_bundle.zip"
COLAB_BUNDLE_DATA = ("data/processed/dreaddit.parquet", "data/processed/splits/dreaddit_split_v1.json")
# The local handoff log travels in the bundle under this name (NOT at trained_models/handoff.json, so unpacking a bundle
# never overwrites the Colab server's own log); stress_signals.handoff merges it, so Colab skips steps done locally.
BUNDLE_HANDOFF_ARCNAME = "_handoff_import/handoff.json"


def build_colab_bundle(cfg: dict[str, Any], extra_files: Iterable[str] = ()) -> Path:
    """LOCAL: zip code/ + the files Step 3 training needs + the local handoff log into data/outputs/colab_bundle.zip
    (~1 MB). Upload that one file to the Colab server (VS Code: right-click > Upload to Colab); cell 0.0 unpacks it on
    the server's own disk. Nothing goes to Google Drive. Rebuild after every code change and after local training."""
    import zipfile
    root = Path(cfg["_root"])
    skip = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    files = [p for p in sorted((root / "code").rglob("*")) if p.is_file() and not skip & set(p.parts) and p.suffix != ".pyc"]
    for rel in (*COLAB_BUNDLE_DATA, *extra_files):
        if not (root / rel).is_file():
            raise FileNotFoundError(f"{rel} missing; run Step 2 locally first")
        files.append(root / rel)
    out = Path(cfg["_paths"]["artifacts"]) / COLAB_BUNDLE_NAME
    tmp = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(root).as_posix())
        log = models_root(cfg) / "handoff.json"
        if log.is_file():
            z.write(log, BUNDLE_HANDOFF_ARCNAME)
    os.replace(tmp, out)
    return out


def checkpoint_root(cfg: dict[str, Any]) -> Path:
    """Where Trainer checkpoints go: outside trained_models, because the Trainer deletes (rotates) old checkpoints and
    they would bloat colab_results.zip. Colab: the server's own disk (lost if the server is recycled). Local:
    data/outputs/checkpoints under the project root."""
    st = cfg.get("storage", {})
    if is_colab():
        return Path(st.get("colab_checkpoint_root", "/content/stress_signals_checkpoints"))
    p = Path(st.get("local_checkpoint_root", "data/outputs/checkpoints"))
    return p if p.is_absolute() else Path(cfg["_root"]) / p


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


def detect_environment() -> dict[str, Any]:
    """Collect Python/torch/CUDA/GPU/Java facts and hardware warnings. Never raises if torch is missing."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "is_colab": is_colab(),
        "torch": None,
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
            if info["gpu_capability"] not in info["arch_list"] and f"compute_{major}{minor}" not in info["arch_list"]:
                w.append(f"{info['gpu_capability']} is NOT in torch.cuda.get_arch_list() -> this torch build "
                         "cannot run kernels on this GPU. Install a build that lists it (or use CPU).")
            if info["gpu_capability"] == "sm_61" and "sm_61" not in info["arch_list"]:
                w.append("sm_61 (Pascal) missing from arch list.")
            if info["vram_total_gb"] < MIN_TRAINING_VRAM_GB:
                w.append(f"GPU VRAM {info['vram_total_gb']} GB < {MIN_TRAINING_VRAM_GB} GB: transformer training here is slow and "
                         "may run out of memory (notebook_local.ipynb L0.4 lowers the per-step batch); Colab is faster.")
        else:
            w.append("CUDA not available: CPU-only fallback will be used. Transformer training is very slow on CPU; prefer Colab.")
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
