# Handoff log: what has been done, where and when

_Regenerated 2026-10-03T09:37:41+00:00 UTC on DESKTOP-GH0P1UM. Times are UTC. Source of truth: `handoff.json`._

## How this works (read first)

**Run All does everything that is not done yet, and nothing twice.**

- Every step that computes or writes something runs **once**. On later Run Alls it prints `SKIP <step>: already done on <machine> at <time>` and loads its saved result instead. Cells that only load or show things re-run (seconds).
- **Training** (baseline 3A.1, seeds 3B.1) starts automatically for whatever is not done, on the laptop's NVIDIA GPU (CUDA) if it has one, else on the CPU. An interrupted seed resumes from its last epoch on the next Run All.
- **Optional LOSO** (3E.2, ~10 extra trainings) runs only with `RUN_LOSO = True` in that cell.
- **Evaluation** (3C–3F) waits until every seed is trained (`NOT RUN … missing [...]` until then), because the final test report is written only once and must not pick a final seed from a partial set.
- **The test suite** (2.5) re-runs only after the code changed. **Re-run anything on purpose:** add its step number to `FORCE_RERUN` in cell 0.1, e.g. `FORCE_RERUN = {"3C.1"}`.
- A step whose saved output was deleted counts as not done and runs again.

## Current state per step

| step | what | state | machine (device) | when (UTC) | duration | note |
|---|---|---|---|---|---|---|
| `1.2/audits` | Audit all datasets | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T09:36:11+00:00 |  | output found on disk (finished before it was logged) |
| `1.5/handcheck/cb810d15f241` | Score the SenticNet hand-check | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T09:36:26+00:00 | 14s |  |
| `2.2/processed/all` | Build processed datasets + splits | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T09:36:27+00:00 |  | output found on disk (finished before it was logged) |
| `2.4/chunking_check` | Chunking check on the Zenodo pilot | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T09:36:33+00:00 | 6s |  |
| `2.5/tests/e901d8ff11e302c0` | Unit tests for this code version | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T09:37:41+00:00 | 68s |  |

## Full history (oldest first)

| when (UTC) | machine (device) | step | event | note |
|---|---|---|---|---|
| 2026-10-03T09:36:11+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `1.2/audits` | done | output found on disk (finished before it was logged) |
| 2026-10-03T09:36:12+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `1.5/handcheck/cb810d15f241` | started |  |
| 2026-10-03T09:36:26+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `1.5/handcheck/cb810d15f241` | done |  |
| 2026-10-03T09:36:27+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `2.2/processed/all` | done | output found on disk (finished before it was logged) |
| 2026-10-03T09:36:27+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `2.4/chunking_check` | started |  |
| 2026-10-03T09:36:33+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `2.4/chunking_check` | done |  |
| 2026-10-03T09:36:33+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `2.5/tests/e901d8ff11e302c0` | started |  |
| 2026-10-03T09:37:41+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `2.5/tests/e901d8ff11e302c0` | done |  |
