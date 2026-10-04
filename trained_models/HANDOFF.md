# Handoff log: what has been done, where and when

_Regenerated 2026-10-04T06:42:48+00:00 UTC on DESKTOP-GH0P1UM. Times are UTC. Source of truth: `handoff.json`._

## How this works (read first)

**Run All does everything that is not done yet, and nothing twice.**

- Every step that computes or writes something runs **once**. On later Run Alls it prints `SKIP <step>: already done on <machine> at <time>` and loads its saved result instead. Cells that only load or show things re-run (seconds).
- **Training** (baseline 3A.1, seeds 3B.1) starts automatically for whatever is not done, on the laptop's NVIDIA GPU (CUDA) if it has one, else on the CPU. An interrupted seed resumes from its last epoch on the next Run All.
- **Emotion model (Step 4)**: its seeds (4B.1) train the same way. Its evaluation (4C-4G) waits until every emotion seed is trained; the Dreaddit domain-shift check (4F) is inference only.
- **Stressor model (Step 5)**: the SAD fits (5C.3 logistic regression, 5C.4 fine-tune seeds) train the same way. Everything that uses the Reddit gold set (5C.6 onward) waits until annotator 1's sheet is complete.
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
| `3A.1/baseline_tfidf_lr` | Fit TF-IDF + logistic regression baseline | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T11:11:03+00:00 | 17s |  |
| `3A.2/baseline_eval` | Evaluate the baseline | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T11:11:14+00:00 | 11s |  |
| `3B.1/roberta-base/seed42` | Fine-tune roberta-base, seed 42 | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T11:32:12+00:00 | 1258s |  |
| `3B.1/roberta-base/seed13` | Fine-tune roberta-base, seed 13 | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T19:53:46+00:00 | 27379s |  |
| `3B.1/roberta-base/seed2024` | Fine-tune roberta-base, seed 2024 | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T20:15:49+00:00 | 1323s |  |
| `3C.1/evaluate_seeds/roberta-base` | Evaluate all seeds (uncalibrated) | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T20:16:28+00:00 | 39s |  |
| `3D.1/calibrate/roberta-base/seed2024` | Calibrate the final seed on validation | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T20:16:39+00:00 | 11s |  |
| `3D.2/final_test_report/roberta-base/seed2024` | Final calibrated test report | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T20:16:51+00:00 | 11s |  |
| `3E.1/topic_masking/roberta-base/seed2024` | Topic-masking shortcut check | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T20:20:00+00:00 | 190s |  |
| `3F/stress_bundle` | Save stress model bundle (roberta-base, seed 2024) | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T20:20:04+00:00 | 3s |  |
| `2.5/tests/3d32290f41966c2a` | Unit tests for this code version | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T22:36:03+00:00 | 72s |  |
| `4B.1/emotion/roberta-base/seed42` | Fine-tune emotion roberta-base, seed 42 | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-04T03:16:02+00:00 | 14531s |  |
| `2.5/tests/74d86a3c84313baa` | Unit tests for this code version | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-03T23:13:49+00:00 | 74s |  |
| `4B.1/emotion/roberta-base/seed13` | Fine-tune emotion roberta-base, seed 13 | **done** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-04T06:19:02+00:00 | 5494s |  |
| `4B.1/emotion/roberta-base/seed2024` | Fine-tune emotion roberta-base, seed 2024 | **failed** | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | 2026-10-04T06:42:48+00:00 | 164s | KeyboardInterrupt:  |

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
| 2026-10-03T11:10:46+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3A.1/baseline_tfidf_lr` | started |  |
| 2026-10-03T11:11:03+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3A.1/baseline_tfidf_lr` | done |  |
| 2026-10-03T11:11:03+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3A.2/baseline_eval` | started |  |
| 2026-10-03T11:11:14+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3A.2/baseline_eval` | done |  |
| 2026-10-03T11:11:14+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3B.1/roberta-base/seed42` | started |  |
| 2026-10-03T11:32:12+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3B.1/roberta-base/seed42` | done |  |
| 2026-10-03T11:32:12+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3B.1/roberta-base/seed13` | started |  |
| 2026-10-03T11:37:20+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3B.1/roberta-base/seed13` | failed | KeyboardInterrupt:  |
| 2026-10-03T12:17:27+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3B.1/roberta-base/seed13` | started |  |
| 2026-10-03T19:53:46+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3B.1/roberta-base/seed13` | done |  |
| 2026-10-03T19:53:46+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3B.1/roberta-base/seed2024` | started |  |
| 2026-10-03T20:15:49+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3B.1/roberta-base/seed2024` | done |  |
| 2026-10-03T20:15:49+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3C.1/evaluate_seeds/roberta-base` | started |  |
| 2026-10-03T20:16:28+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3C.1/evaluate_seeds/roberta-base` | done |  |
| 2026-10-03T20:16:28+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3D.1/calibrate/roberta-base/seed2024` | started |  |
| 2026-10-03T20:16:39+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3D.1/calibrate/roberta-base/seed2024` | done |  |
| 2026-10-03T20:16:39+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3D.2/final_test_report/roberta-base/seed2024` | started |  |
| 2026-10-03T20:16:51+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3D.2/final_test_report/roberta-base/seed2024` | done |  |
| 2026-10-03T20:16:51+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3E.1/topic_masking/roberta-base/seed2024` | started |  |
| 2026-10-03T20:20:00+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3E.1/topic_masking/roberta-base/seed2024` | done |  |
| 2026-10-03T20:20:01+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3F/stress_bundle` | started |  |
| 2026-10-03T20:20:04+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `3F/stress_bundle` | done |  |
| 2026-10-03T22:34:51+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `2.5/tests/3d32290f41966c2a` | started |  |
| 2026-10-03T22:36:03+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `2.5/tests/3d32290f41966c2a` | done |  |
| 2026-10-03T22:36:07+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed42` | started |  |
| 2026-10-03T22:52:05+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed42` | failed | KeyboardInterrupt:  |
| 2026-10-03T23:12:35+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `2.5/tests/74d86a3c84313baa` | started |  |
| 2026-10-03T23:13:49+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `2.5/tests/74d86a3c84313baa` | done |  |
| 2026-10-03T23:13:51+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed42` | started |  |
| 2026-10-04T03:16:02+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed42` | done |  |
| 2026-10-04T03:16:02+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed13` | started |  |
| 2026-10-04T03:34:19+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed13` | failed | KeyboardInterrupt:  |
| 2026-10-04T03:34:46+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed13` | started |  |
| 2026-10-04T03:35:52+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed13` | failed | KeyboardInterrupt:  |
| 2026-10-04T03:43:05+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed13` | started |  |
| 2026-10-04T03:48:08+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed13` | failed | KeyboardInterrupt:  |
| 2026-10-04T04:47:28+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed13` | started |  |
| 2026-10-04T06:19:02+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed13` | done |  |
| 2026-10-04T06:19:02+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed2024` | started |  |
| 2026-10-04T06:39:45+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed2024` | failed | KeyboardInterrupt:  |
| 2026-10-04T06:40:04+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed2024` | started |  |
| 2026-10-04T06:42:48+00:00 | DESKTOP-GH0P1UM (Quadro P1000 (cuda)) | `4B.1/emotion/roberta-base/seed2024` | failed | KeyboardInterrupt:  |
