# Early Detection of Collective Psychological Pressure

> **Aggregate, population-level language signals only. No diagnosis. No individual-level analysis.**

This project measures stress-related *language* in public Reddit text. It looks at emotions, stressors, and changes over time,
and reports only aggregate counts. Any aggregate cell with n < `privacy.k_min` (50) is suppressed. Results are phrased as
"stress-related language in the sampled communities shifted". They never say anything about a person or a population's mental health.

## Layout

```
code/
  notebook.ipynb        one section per step; imports the package below (Step 3 training cells run on Colab)
  notebook_local.ipynb  local-kernel twin of Step 3: trains/evaluates the same models on the same data on this PC
  config.yaml           single configuration; every path in it is relative to the project root
  stress_signals/       package (audit, preprocess, privacy, config, utils, stress_model, metrics, handoff)
  tests/                pytest suite
  pyproject.toml        package + pytest settings
  _deferred/            placeholders for the Postgres step (docker-compose.yml, .env.example)
data/
  raw/                  downloads (SAD_v1, mendeley, sentic, zenodo, goemotions_official, sad, tensistrength, _metadata)
  processed/            unified-schema parquet files, splits/ (fixed split indices), _cache/ (audits, hand-check, per-file caches)
  outputs/              manifests/, reports/, logs/, checkpoints/ (later: gold/, taxonomy/)
trained_models/         every model artefact (runs, predictions, bundles) + HANDOFF.md / handoff.json (what ran where, when)
requirements.txt
```

The project root is the folder that contains `code/` and `data/`. `stress_signals.config.load_config()` finds it by walking up from the
current directory until it reaches `code/config.yaml`.

## Setup (local, Windows)

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e code --no-deps
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.get_arch_list())"   # must list sm_61 for the Quadro P1000
```

Training normally runs on a Colab GPU (`notebook.ipynb`). It can also run on this PC with `notebook_local.ipynb`, which is slow on the
4 GB Quadro P1000; its cell L0.4 lowers the per-step batch. Local inference uses fp32 and small batches, with a CPU fallback.

## Run

- Notebook: open `code/notebook.ipynb`, run cell 0.1, then the cells for the current step.
- Local Step 3 training: open `code/notebook_local.ipynb` on the local kernel and run L0.1 → L0.3, then the Step 3 cells.
- Tests: `cd code` and run `..\.venv\Scripts\python.exe -m pytest -q`.
- Step 1 audit: `.venv\Scripts\python.exe -m stress_signals.audit`.
- Step 2 preprocessing: `.venv\Scripts\python.exe -m stress_signals.preprocess`.
- Both commands skip outputs that already exist. Saved splits are never regenerated unless you pass `--force-splits`.

## Reproducibility

Every step writes a manifest to `data/outputs/manifests/`. It records library versions, seeds, dataset versions, and SHA-256 hashes of the artifacts.
`relocation_map.json` in the same folder records the move from the old layout to this one.

## Colab GPU, trained_models/ and the handoff log

- **Everything lives on this PC. Google Drive is not used.** Model artefacts go to `trained_models/` in the project root on whichever kernel ran the step.
  On Colab that is `/content/TA-BDA/trained_models`, on the server's **temporary** disk.
- **To Colab:** build `data/outputs/colab_bundle.zip` with cell 3.0c (or L9 in `notebook_local.ipynb`). It holds the code, the training data and the PC's handoff log.
  In the VS Code Explorer, right-click it → **Upload to Colab**, then run cell **0.0** on the Colab kernel. Rebuild and re-upload after any code change and after local training.
- **Back from Colab:** cell **3G** packs `trained_models/` into `/content/colab_results.zip`. Download it into the project root, `data/outputs/` or Downloads.
  Then run **3.0d** (or **L0.3**) on the local kernel. The import never overwrites existing files and never deletes anything, and it merges Colab's log into the local one.
- **Run once, across both kernels.** Each training unit (baseline, each seed, each LOSO subreddit, the bundle) is logged in `trained_models/HANDOFF.md` with UTC timestamps,
  the kernel and a fingerprint of the training data. A unit that is logged as done, on either kernel, is skipped by both notebooks, and the cell says where and when it ran.
- **Run All is safe.** Training cells need an explicit opt-in: in `notebook.ipynb` they train only on the Colab kernel, and in `notebook_local.ipynb` only what you
  switch on in `TRAIN_HERE` (cell L0.5). Every other training or evaluation cell prints `NOT RUN …` with the reason and Run All carries on.
- **Nothing in this project deletes files.** A test enforces that the package has no delete calls. Epoch checkpoints, which the Trainer rotates, live outside `trained_models/`:
  in `data/outputs/checkpoints/` locally and in `/content/stress_signals_checkpoints` on Colab.
