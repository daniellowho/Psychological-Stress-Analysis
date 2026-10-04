# Early Detection of Collective Psychological Pressure

> **Aggregate, population-level language signals only. No diagnosis. No individual-level analysis.**

This project measures stress-related *language* in public Reddit text. It looks at emotions, stressors, and changes over time,
and reports only aggregate counts. Any aggregate cell with n < `privacy.k_min` (50) is suppressed. Results are phrased as
"stress-related language in the sampled communities shifted". They never say anything about a person or a population's mental health.

## Layout

```
code/
  notebook.ipynb        the whole pipeline in one notebook: open it and Run All
  config.yaml           single configuration; every path in it is relative to the project root
  stress_signals/       package (audit, preprocess, privacy, config, utils, stress_model, metrics, handoff)
  tests/                pytest suite
  pyproject.toml        package + pytest settings
  _deferred/            placeholders for the Postgres step (docker-compose.yml, .env.example)
data/
  raw/                  downloads (SAD_v1, mendeley, sentic, zenodo, goemotions_official, sad, tensistrength, _metadata)
  processed/            unified-schema parquet files, splits/ (fixed split indices), _cache/ (audits, hand-check, per-file caches)
  outputs/              manifests/, reports/, logs/, checkpoints/ (later: gold/, taxonomy/)
trained_models/         every model artefact (runs, predictions, bundles) + HANDOFF.md / handoff.json (what ran, where, when)
requirements.txt
```

The project root is the folder that contains `code/` and `data/`. `stress_signals.config.load_config()` finds it by walking up from the
current directory until it reaches `code/config.yaml`.

## Run it on any Windows laptop

1. Install **Python 3.11** (python.org) and VS Code with the Python + Jupyter extensions.
2. Get the project (`git clone https://github.com/daniellowho/Psychological-Stress-Analysis`) and create a virtual environment in it:
   ```
   python -m venv .venv
   ```
3. Open `code/notebook.ipynb` in VS Code → *Select Kernel* → *Python Environments* → `.venv`.
4. **Run All.**
   - Cell **0.0** installs every library. If the laptop has an NVIDIA GPU it installs the matching CUDA build of torch
     (12.1, 11.8 for old drivers, 12.8 for RTX 50-series) and proves the GPU works. Without an NVIDIA GPU it uses the CPU.
     If it asks you to restart the kernel, do so and Run All again.
   - Training starts automatically: the baseline, then 3 seeds of roberta-base. Cell 3.0c sizes the batch to the GPU's memory and
     keeps the effective batch at 16, so results are comparable between laptops. A recent NVIDIA GPU takes minutes per epoch,
     a 4 GB GPU much longer, and the CPU hours per seed.

**Nothing runs twice.** Every step that computes or writes something runs once and is logged with its time, laptop and device in
`trained_models/HANDOFF.md`. Later Run Alls print `SKIP <step>: already done on <laptop> at <time>` and load the saved result. An
interrupted training resumes from its last finished epoch. The test suite (2.5) re-runs only after the code changed. To redo a step on purpose,
add its number to `FORCE_RERUN` in cell 0.1, e.g. `{"3C.1"}`. To retrain a seed, move its folder out of `trained_models/stress/runs/`.

**Raw datasets are optional for training.** Steps 1–2 audit and build all datasets only when the manual downloads (SAD_v1, Mendeley,
SenticNet, Zenodo) are in `data/raw/`. Without them those cells print `NOT RUN …`. Step 3 needs only Dreaddit, which is downloaded from
Hugging Face, and the fixed split file, which is in git.

**Optional:** leave-one-subreddit-out (3E.2, about 10 extra trainings) runs only with `RUN_LOSO = True` in that cell.

## Command line

- Tests: `cd code` and run `..\.venv\Scripts\python.exe -m pytest -q`.
- Step 1 audit: `.venv\Scripts\python.exe -m stress_signals.audit`.
- Step 2 preprocessing: `.venv\Scripts\python.exe -m stress_signals.preprocess`.
- Both commands skip outputs that already exist. Saved splits are never regenerated unless you pass `--force-splits`.

## Reproducibility

Every step writes a manifest to `data/outputs/manifests/`. It records library versions, seeds, the device, dataset versions and SHA-256
hashes of the artifacts. `relocation_map.json` in the same folder records the move from the old layout to this one. Nothing in this project
deletes files; a test enforces that the package has no delete calls. Epoch checkpoints, which the Trainer rotates, live in
`data/outputs/checkpoints/`, outside `trained_models/`.

## What is on GitHub vs Hugging Face

GitHub holds the code, config, tests, taxonomy, fixed split indices, manifests/reports and the small run records
(`train_summary.json`, evaluation/calibration JSONs, logs, predictions without text). It does **not** hold anything large or
anything containing Reddit text, so these must be fetched or rebuilt:

| Missing from a fresh clone | Where it comes from |
|---|---|
| Trained stress model (`model.safetensors`, tokenizer, config; ~480 MB) | Hugging Face: [`dannyyyyellooo/stress-roberta-base`](https://huggingface.co/dannyyyyellooo/stress-roberta-base) (currently private; ask the owner for access or to make it public) |
| Per-seed `best_model/` weights and checkpoints (`trained_models/*/runs/*/seed*/best_model/`, `data/outputs/checkpoints/`) | Not published. Retrain by running the notebook (Step 3). |
| Emotion model weights | Not published. Retrain by running the notebook. |
| `data/raw/` and `data/processed/*.parquet` | Private Hugging Face dataset [`dannyyyyellooo/stress-signals-data`](https://huggingface.co/datasets/dannyyyyellooo/stress-signals-data) (see below), or rebuilt by Steps 1-2. Dreaddit and GoEmotions download automatically; SAD, Mendeley, SenticNet and Zenodo are manual downloads. |

### Using the published stress model

The Hugging Face repo is a standard `transformers` checkpoint (roberta-base, binary stress / no-stress), so it downloads and caches
itself on first use. While the repo is private, log in first with `hf auth login` (a read token is enough).

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "dannyyyyellooo/stress-roberta-base"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)
```

Or download the whole folder to `trained_models/stress/v20261004/`, where the project code expects it:

```
.venv\Scripts\hf download dannyyyyellooo/stress-roberta-base --local-dir trained_models/stress/v20261004
```

The repo also contains `stress_config.json` (temperature-scaling calibration and the decision threshold), `metrics.json`,
`manifest.json` and the model card (shown as the repo's README). Apply the calibration from `stress_config.json` before using raw
probabilities. The model is for aggregate, population-level signals only, never for diagnosing or scoring individuals.

### Datasets (private)

`dannyyyyellooo/stress-signals-data` is a **private** Hugging Face dataset repo holding `data/raw/` and `data/processed/`
(including `_cache/` and the fixed splits), laid out under the same `data/` paths as the project. It is private on purpose:
the files contain real Reddit text, the raw Zenodo files still have `author`, and some source licences (TensiStrength,
Dreaddit) are restricted or unverified. Do not make it public or redistribute it. To restore it into a fresh clone (needs
access to the repo and `hf auth login`):

```
.venv\Scripts\hf download dannyyyyellooo/stress-signals-data --repo-type dataset --local-dir .
```

This recreates `data/raw/` and `data/processed/` in place, so Steps 1-2 are then skipped. Compare file hashes against
`data/outputs/manifests/` to confirm they match.
