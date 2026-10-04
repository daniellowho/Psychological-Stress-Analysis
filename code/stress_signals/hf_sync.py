"""Hugging Face sync for the datasets and the three model bundles named in config.yaml (`huggingface:`).

PULL (`pull_missing`): download only the files that are missing locally; existing files are never overwritten or deleted. If the repos are
private and this laptop is not logged in (`hf auth login`), or a repo is unreachable, every function returns a note instead of raising,
so the notebook still runs (Step 3 can fall back to downloading Dreaddit and training).

PUSH (`push_bundle`): upload one saved bundle (trained_models/<component>/v<date>/) to its PRIVATE model repo, byte for byte, so the
SHA-256 manifest inside the bundle still verifies after a download, plus a README.md model card (YAML header + the bundle's model_card.md).
Command line (needs `hf auth login` with write access to the organisation):
    python -m stress_signals.hf_sync push emotion       # or: stressor | stress
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# component -> (config key of the repo, config key of the local folder)
BUNDLES = {"model": ("model_repo", "model_dir"), "emotion": ("emotion_repo", "emotion_dir"), "stressor": ("stressor_repo", "stressor_dir")}
COMPONENT_OF = {"stress": "model", "emotion": "emotion", "stressor": "stressor"}   # push_bundle names -> BUNDLES keys

# YAML headers of the model-repo README.md (license kept "other": the data licences are not all verified, as for the stress repo)
README_HEADERS = {
    "stress": """language: en
library_name: transformers
pipeline_tag: text-classification
base_model: roberta-base
license: other
license_name: unverified-dreaddit-terms
tags:
  - stress
  - reddit
  - roberta
  - text-classification
datasets:
  - andreagasparini/dreaddit""",
    "emotion": """language: en
library_name: transformers
pipeline_tag: text-classification
base_model: roberta-base
license: other
license_name: unverified-see-model-card
tags:
  - emotion
  - goemotions
  - reddit
  - roberta
  - multi-label
datasets:
  - google-research-datasets/go_emotions""",
    "stressor": """language: en
library_name: transformers
pipeline_tag: text-classification
base_model: distilroberta-base
license: other
license_name: unverified-see-model-card
tags:
  - stressors
  - reddit
  - distilroberta
  - multi-label""",
}

README_NOTES = {
    "stress": "",
    "emotion": "",
    "stressor": """
## Notes added when this bundle was published

- **The weights alone are not the tagger.** The tagger also needs `stressor_config.json`, `thresholds.json`, `prototypes.npy` (reference vectors for
  the two zero-shot categories) and the embedding model `sentence-transformers/all-MiniLM-L6-v2` (downloaded separately from the Hub). Use
  `stress_signals.inference.StressSignalPipeline`, which loads all of it and re-verifies every file against `manifest.json`.
- **The evaluation set is machine-labelled**, not hand-labelled (the line in the card above that says "hand-labelled" is wrong; the
  `model_card.md` file is kept byte-for-byte because `manifest.json` records its SHA-256). Every Reddit score is agreement with a zero-shot labeller.
- **Thresholds.** The labeller marks most categories on most posts, so a category that is positive on more than 60% of the tuning posts is not tuned:
  probability categories use the 0.5 default, and the two zero-shot-only categories (uncertainty, societal_problems) are never flagged (their raw
  scores are still output). In the project the tags are used only for texts that the stress model flags.
- Provisional: use for aggregate signals only, never for any person.
""",
}


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


def pull_missing(cfg: dict[str, Any], what: tuple[str, ...] = ("data", "model", "emotion", "stressor")) -> dict[str, str]:
    """Fetch missing datasets ("data") and/or the trained bundles ("model" = stress, "emotion", "stressor"). Returns {what: note}.
    A bundle whose folder already exists locally with all its files is left alone (nothing is downloaded)."""
    hf = cfg.get("huggingface") or {}
    if not hf.get("enabled", False):
        return {"all": "huggingface.enabled is false: nothing pulled"}
    root = Path(cfg["_root"])
    notes: dict[str, str] = {}
    jobs = []
    if "data" in what and hf.get("data_repo"):
        jobs.append(("data", hf["data_repo"], "dataset", root, "", tuple(hf.get("data_paths", ["data/raw", "data/processed"]))))
    for name, (repo_key, dir_key) in BUNDLES.items():
        if name in what and hf.get(repo_key):
            default_dir = {"model": "stress", "emotion": "emotion", "stressor": "stressor"}[name]
            bundle = root / hf.get(dir_key, f"trained_models/{default_dir}/v20261004")
            jobs.append((name, hf[repo_key], "model", bundle, "", ()))
    for name, repo, rtype, target, prefix, wanted in jobs:
        try:
            got = _pull(repo, rtype, target, prefix, wanted)
            notes[name] = f"downloaded {len(got)} missing file(s) from {repo}" if got else f"nothing missing (checked {repo})"
        except Exception as e:  # not logged in, no access, offline, repo not found ...
            notes[name] = f"SKIPPED ({type(e).__name__}): could not reach {repo}. Private repo? Run `hf auth login` or see README."
    return notes


def bundle_readme(component: str, model_card: str) -> str:
    """README.md for a model repo: YAML header + the bundle's own model_card.md + publication notes (if any)."""
    return f"---\n{README_HEADERS[component]}\n---\n\n{model_card.rstrip()}\n{README_NOTES[component]}"


def push_bundle(cfg: dict[str, Any], component: str, private: bool = True, dry_run: bool = False) -> dict[str, Any]:
    """Upload trained_models/<component>/<version>/ to its model repo (config huggingface.<...>_repo / <...>_dir), PRIVATE by default.
    Files go up byte for byte; README.md is added. Returns what was (or would be) uploaded. `dry_run=True` only lists it.
    Refuses if the bundle's own manifest does not verify (a corrupted bundle must not be published)."""
    if component not in COMPONENT_OF:
        raise KeyError(f"component must be one of {sorted(COMPONENT_OF)}")
    hf = cfg.get("huggingface") or {}
    repo_key, dir_key = BUNDLES[COMPONENT_OF[component]]
    repo, folder = hf.get(repo_key), Path(cfg["_root"]) / hf.get(dir_key, "")
    if not repo or not folder.is_dir():
        raise FileNotFoundError(f"config huggingface.{repo_key} / {dir_key} missing, or {folder} is not a folder")
    from .inference import verify_bundle
    check = verify_bundle(folder)                                    # raises BundleIntegrityError if any file differs from its manifest
    files = sorted(p.name for p in folder.iterdir() if p.is_file())
    card = (folder / "model_card.md").read_text(encoding="utf-8")
    readme = bundle_readme(component, card)
    out = {"repo": repo, "private": private, "folder": folder.as_posix(), "files": files + ["README.md"], "verified_files": check["n_files"],
           "bytes": sum((folder / f).stat().st_size for f in files)}
    if dry_run:
        return out
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type="model", commit_message=f"Upload {component} bundle {folder.name}")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md", repo_id=repo, repo_type="model",
                    commit_message="Add model card (README.md)")
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    from .config import load_config
    ap = argparse.ArgumentParser(description="Hugging Face sync")
    ap.add_argument("cmd", choices=["push"])
    ap.add_argument("component", choices=sorted(COMPONENT_OF))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--public", action="store_true", help="make the repo public (default: private)")
    a = ap.parse_args(argv)
    print(json.dumps(push_bundle(load_config(), a.component, private=not a.public, dry_run=a.dry_run), indent=2))


if __name__ == "__main__":
    main()
