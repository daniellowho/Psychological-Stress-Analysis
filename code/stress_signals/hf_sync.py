"""Pull the datasets and the trained stress model from the Hugging Face repos named in config.yaml (`huggingface:`).

Only files that are missing locally are downloaded; existing files are never overwritten or deleted. If the repos are private
and this laptop is not logged in (`hf auth login`), or the repo is unreachable, every function returns a note instead of raising,
so the notebook still runs (Step 3 can fall back to downloading Dreaddit and training).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _pull(repo_id: str, repo_type: str, target_root: Path, prefix: str, wanted: tuple[str, ...]) -> list[str]:
    """Download repo files under `prefix` (and matching `wanted` top-level folders/names) that are missing in target_root."""
    from huggingface_hub import HfApi, hf_hub_download

    got = []
    for f in HfApi().list_repo_files(repo_id, repo_type=repo_type):
        if f == ".gitattributes" or (wanted and not any(f == w or f.startswith(w.rstrip("/") + "/") for w in wanted)):
            continue
        rel = f[len(prefix):] if prefix and f.startswith(prefix) else f
        if (target_root / rel).is_file():
            continue
        hf_hub_download(repo_id, f, repo_type=repo_type, local_dir=str(target_root))
        got.append(f)
    return got


def pull_missing(cfg: dict[str, Any], what: tuple[str, ...] = ("data", "model")) -> dict[str, str]:
    """Fetch missing datasets ("data") and/or the trained stress bundle ("model"). Returns {what: human-readable note}."""
    hf = cfg.get("huggingface") or {}
    if not hf.get("enabled", False):
        return {"all": "huggingface.enabled is false: nothing pulled"}
    root = Path(cfg["_root"])
    notes: dict[str, str] = {}
    jobs = []
    if "data" in what and hf.get("data_repo"):
        jobs.append(("data", hf["data_repo"], "dataset", root, "", tuple(hf.get("data_paths", ["data/raw", "data/processed"]))))
    if "model" in what and hf.get("model_repo"):
        bundle = root / hf.get("model_dir", "trained_models/stress/v20261004")
        jobs.append(("model", hf["model_repo"], "model", bundle, "", ()))
    for name, repo, rtype, target, prefix, wanted in jobs:
        try:
            got = _pull(repo, rtype, target, prefix, wanted)
            notes[name] = f"downloaded {len(got)} missing file(s) from {repo}" if got else f"nothing missing (checked {repo})"
        except Exception as e:  # not logged in, no access, offline, repo not found ...
            notes[name] = f"SKIPPED ({type(e).__name__}): could not reach {repo}. Private repo? Run `hf auth login` or see README."
    return notes
