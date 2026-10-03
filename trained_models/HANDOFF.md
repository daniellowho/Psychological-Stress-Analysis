# Handoff log: what has been done, where and when

_Regenerated 2026-10-03T08:09:02+00:00 UTC by the local kernel. Times are UTC. Source of truth: `handoff.json`._

## How this works (read first)

**Run All is safe. Training never starts by accident.**

- **`notebook.ipynb`**: training cells (3A.1, 3B.1, 3E.2) train **only on the Colab kernel**. On the local kernel they print `NOT RUN …: notebook.ipynb trains on the Colab kernel only…`.
- **`notebook_local.ipynb`**: nothing trains until you switch it on in cell **L0.5**:
  `TRAIN_HERE = {"baseline": False, "seeds": [], "loso": False}` (e.g. `"seeds": [42]`). That cell also previews what Run All would do now ("done on colab at … → skip" / "WILL TRAIN HERE" / "not switched on → skip").
- **Done always wins.** A step marked **done** below, on EITHER kernel, is skipped by both notebooks even if it is switched on. The cell says where and when it ran.
- **Missing Colab results:** if Colab trained something that is not imported yet, the message says to import `colab_results.zip` instead of retraining.
- **Evaluation cells** (3A.2, 3C.\*, 3D.\*, 3E.1, 3E.3, 3F) print e.g. `NOT RUN 3C.1: … missing [13, 2024]` instead of failing, so Run All carries on. They wait for **every** seed, because 3D.2 writes the final test report only once and must not pick a final seed from a partial set.

**Moving results between kernels:** PC → Colab: rebuild `colab_bundle.zip` (cell 3.0c / L9), upload it, run 0.0 (it carries this log). Colab → PC: cell 3G writes `/content/colab_results.zip`; download it to the project root, `data/outputs/` or Downloads, and run 3.0d / L0.3. The import never overwrites or deletes files, and it merges the logs.

## Current state per step

| step | what | state | kernel | when (UTC) | duration | note |
|---|---|---|---|---|---|---|

## Full history (oldest first)

| when (UTC) | kernel | step | event | note |
|---|---|---|---|---|
