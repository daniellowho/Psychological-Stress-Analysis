# Handoff log: what has been done, where and when

_Regenerated 2026-10-04T17:20:18+00:00 UTC on madhava_laptop. Times are UTC. Source of truth: `handoff.json`._

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
| `4B.1/emotion/roberta-base/seed13` | Fine-tune emotion roberta-base, seed 13 | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:40:16+00:00 | 757s |  |
| `4B.1/emotion/roberta-base/seed2024` | Fine-tune emotion roberta-base, seed 2024 | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:22:50+00:00 | 882s |  |
| `2.2/processed/dreaddit` | Build processed datasets + splits | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T08:29:07+00:00 |  | output found on disk (finished before it was logged) |
| `2.5/tests/4ef0afd941f00d1c` | Unit tests for this code version | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:08:05+00:00 | 34s |  |
| `4C.1/emotion_evaluate_seeds/roberta-base` | Evaluate all emotion seeds (uncalibrated) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:22:50+00:00 | 0s |  |
| `4D.1/emotion_calibrate/roberta-base/seed13` | Calibrate the emotion model per label | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:40:17+00:00 | 1s |  |
| `4C.3/emotion_thresholds/roberta-base/seed13` | Tune per-label emotion thresholds | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:40:17+00:00 | 0s |  |
| `4C.4/emotion_final_test_report/roberta-base/seed13` | Final emotion test report | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:40:32+00:00 | 15s |  |
| `4F.1/emotion_domain_shift/roberta-base/seed13` | Emotion domain-shift check on Dreaddit | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:40:48+00:00 | 16s |  |
| `4F.2/emotion_handcheck_sheet/roberta-base/seed13` | Write the emotion hand-check sheet | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:40:49+00:00 | 0s |  |
| `4G/emotion_bundle` | Save emotion model bundle (roberta-base, seed 13) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:40:50+00:00 | 1s |  |
| `5A.1/taxonomy_report` | Write the stressor taxonomy report | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:40:50+00:00 | 0s |  |
| `5A.2/sad_split` | Create the fixed SAD split | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:40:50+00:00 | 0s |  |
| `5C.0/embedding_check` | Verify the embedding models load | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:41:45+00:00 | 55s |  |
| `5B.1/gold_sample` | Sample the stressor gold set + annotation sheets | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:41:47+00:00 | 0s |  |
| `5C.3/stressor_embed_lr` | Fit embedding + logistic regression (all-MiniLM-L6-v2) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:41:49+00:00 | 3s |  |
| `5C.4/stressor/distilroberta-base/seed42` | Fine-tune stressor distilroberta-base, seed 42 | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:44:03+00:00 | 133s |  |
| `5C.4/stressor/distilroberta-base/seed13` | Fine-tune stressor distilroberta-base, seed 13 | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:44:50+00:00 | 47s |  |
| `5C.4/stressor/distilroberta-base/seed2024` | Fine-tune stressor distilroberta-base, seed 2024 | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:45:36+00:00 | 46s |  |
| `5C.5/sad_report/embed_lr__all-MiniLM-L6-v2` | SAD calibration + test (embed_lr) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:45:36+00:00 | 0s |  |
| `5C.5/sad_report/finetune__distilroberta-base/seed2024` | SAD calibration + test (finetune) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T09:45:36+00:00 | 0s |  |
| `2.5/tests/ad861f36908a1cb4` | Unit tests for this code version | **failed** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:06:08+00:00 | 44s | AssertionError: unit tests failed (see the output above) |
| `2.5/tests/9751651770fba51f` | Unit tests for this code version | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:18:13+00:00 | 44s |  |
| `6A.1/tensistrength_sample` | TensiStrength on Dreaddit test + SenticNet sample | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:18:23+00:00 | 2s |  |
| `6B.1/senticnet_scores/v20261004` | Score SenticNet with the Step 3 model (zero-shot) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:19:12+00:00 | 49s |  |
| `6B.3/senticnet_zero_shot/v20261004` | Zero-shot SenticNet agreement (bootstrap CIs) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:20:26+00:00 | 72s |  |
| `2.5/tests/3bcaec51fc14fd4a` | Unit tests for this code version | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:27:26+00:00 | 47s |  |
| `5B.1b/machine_gold_labels` | Machine-label the stressor gold sheet | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:07+00:00 | 557s |  |
| `5C.6/score_gold/keyword__taxonomy-v1` | Score the gold set (keyword__taxonomy-v1) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:08+00:00 | 0s |  |
| `5C.6/tune_dev/keyword__taxonomy-v1` | Tune thresholds on gold-dev (keyword__taxonomy-v1) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:08+00:00 | 0s |  |
| `5C.6/score_gold/zero_shot__all-MiniLM-L6-v2` | Score the gold set (zero_shot__all-MiniLM-L6-v2) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:09+00:00 | 1s |  |
| `5C.6/tune_dev/zero_shot__all-MiniLM-L6-v2` | Tune thresholds on gold-dev (zero_shot__all-MiniLM-L6-v2) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:09+00:00 | 0s |  |
| `5C.6/score_gold/embed_lr__all-MiniLM-L6-v2` | Score the gold set (embed_lr__all-MiniLM-L6-v2) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:10+00:00 | 0s |  |
| `5C.6/tune_dev/embed_lr__all-MiniLM-L6-v2` | Tune thresholds on gold-dev (embed_lr__all-MiniLM-L6-v2) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:10+00:00 | 0s |  |
| `5C.6/score_gold/finetune__distilroberta-base` | Score the gold set (finetune__distilroberta-base) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:12+00:00 | 2s |  |
| `5C.6/tune_dev/finetune__distilroberta-base` | Tune thresholds on gold-dev (finetune__distilroberta-base) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:12+00:00 | 0s |  |
| `5C.7/gold_test_report` | Final stressor gold-test report | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:14+00:00 | 2s |  |
| `5C.8/crosscheck/zero_shot__all-MiniLM-L6-v2` | Dreaddit subreddit cross-check | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:16+00:00 | 2s |  |
| `5D/stressor_bundle` | Save stressor bundle (zero_shot) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:18+00:00 | 2s |  |
| `6B.1b/machine_handcheck` | Machine-label the SenticNet hand-check | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T10:42:25+00:00 | 4s |  |
| `2.5/tests/e66cf390db7491d4` | Unit tests for this code version | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T16:48:47+00:00 | 54s |  |
| `2.5/tests/dfaeea6f7f5d3b47` | Unit tests for this code version | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T17:02:21+00:00 | 66s |  |
| `5C.8/crosscheck/finetune__distilroberta-base` | Dreaddit subreddit cross-check | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T17:02:29+00:00 |  | output found on disk (finished before it was logged) |
| `8.2/ingest/pilot` | Ingest the pilot corpus into the Parquet lake | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T17:20:18+00:00 |  | output found on disk (finished before it was logged) |
| `8.4/data_quality/pilot` | Data quality report (pilot) | **done** | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | 2026-10-04T17:20:18+00:00 |  | output found on disk (finished before it was logged) |

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
| 2026-10-04T08:29:07+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.2/processed/dreaddit` | done | output found on disk (finished before it was logged) |
| 2026-10-04T08:29:08+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/4ef0afd941f00d1c` | started |  |
| 2026-10-04T08:29:46+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/4ef0afd941f00d1c` | failed | AssertionError: unit tests failed (see the output above) |
| 2026-10-04T09:07:30+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/4ef0afd941f00d1c` | started |  |
| 2026-10-04T09:08:05+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/4ef0afd941f00d1c` | done |  |
| 2026-10-04T09:08:07+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4B.1/emotion/roberta-base/seed2024` | started |  |
| 2026-10-04T09:22:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4B.1/emotion/roberta-base/seed2024` | done |  |
| 2026-10-04T09:22:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.1/emotion_evaluate_seeds/roberta-base` | started |  |
| 2026-10-04T09:22:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.1/emotion_evaluate_seeds/roberta-base` | done |  |
| 2026-10-04T09:22:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4D.1/emotion_calibrate/roberta-base/seed13` | started |  |
| 2026-10-04T09:22:51+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4D.1/emotion_calibrate/roberta-base/seed13` | done |  |
| 2026-10-04T09:22:51+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.3/emotion_thresholds/roberta-base/seed13` | started |  |
| 2026-10-04T09:22:51+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.3/emotion_thresholds/roberta-base/seed13` | done |  |
| 2026-10-04T09:22:51+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.4/emotion_final_test_report/roberta-base/seed13` | started |  |
| 2026-10-04T09:23:06+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.4/emotion_final_test_report/roberta-base/seed13` | done |  |
| 2026-10-04T09:23:06+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4F.1/emotion_domain_shift/roberta-base/seed13` | started |  |
| 2026-10-04T09:23:06+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4F.1/emotion_domain_shift/roberta-base/seed13` | failed | OSError: Incorrect path_or_model_id: 'C:\Visual_studio\TA-BDA Project\Psychological-Stress-Analysis\trained_models\emotion\runs\roberta-base\seed13\best_model'. Please provide either the path to a local folder or the repo_id of a model on the Hub. |
| 2026-10-04T09:27:39+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4B.1/emotion/roberta-base/seed13` | started |  |
| 2026-10-04T09:40:16+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4B.1/emotion/roberta-base/seed13` | done |  |
| 2026-10-04T09:40:16+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4D.1/emotion_calibrate/roberta-base/seed13` | started |  |
| 2026-10-04T09:40:17+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4D.1/emotion_calibrate/roberta-base/seed13` | done |  |
| 2026-10-04T09:40:17+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.3/emotion_thresholds/roberta-base/seed13` | started |  |
| 2026-10-04T09:40:17+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.3/emotion_thresholds/roberta-base/seed13` | done |  |
| 2026-10-04T09:40:17+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.4/emotion_final_test_report/roberta-base/seed13` | started |  |
| 2026-10-04T09:40:32+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4C.4/emotion_final_test_report/roberta-base/seed13` | done |  |
| 2026-10-04T09:40:32+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4F.1/emotion_domain_shift/roberta-base/seed13` | started |  |
| 2026-10-04T09:40:48+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4F.1/emotion_domain_shift/roberta-base/seed13` | done |  |
| 2026-10-04T09:40:48+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4F.2/emotion_handcheck_sheet/roberta-base/seed13` | started |  |
| 2026-10-04T09:40:49+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4F.2/emotion_handcheck_sheet/roberta-base/seed13` | done |  |
| 2026-10-04T09:40:49+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4G/emotion_bundle` | started |  |
| 2026-10-04T09:40:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `4G/emotion_bundle` | done |  |
| 2026-10-04T09:40:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5A.1/taxonomy_report` | started |  |
| 2026-10-04T09:40:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5A.1/taxonomy_report` | done |  |
| 2026-10-04T09:40:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5A.2/sad_split` | started |  |
| 2026-10-04T09:40:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5A.2/sad_split` | done |  |
| 2026-10-04T09:40:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.0/embedding_check` | started |  |
| 2026-10-04T09:41:45+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.0/embedding_check` | done |  |
| 2026-10-04T09:41:46+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5B.1/gold_sample` | started |  |
| 2026-10-04T09:41:47+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5B.1/gold_sample` | done |  |
| 2026-10-04T09:41:47+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.3/stressor_embed_lr` | started |  |
| 2026-10-04T09:41:49+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.3/stressor_embed_lr` | done |  |
| 2026-10-04T09:41:49+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.4/stressor/distilroberta-base/seed42` | started |  |
| 2026-10-04T09:44:03+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.4/stressor/distilroberta-base/seed42` | done |  |
| 2026-10-04T09:44:03+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.4/stressor/distilroberta-base/seed13` | started |  |
| 2026-10-04T09:44:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.4/stressor/distilroberta-base/seed13` | done |  |
| 2026-10-04T09:44:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.4/stressor/distilroberta-base/seed2024` | started |  |
| 2026-10-04T09:45:36+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.4/stressor/distilroberta-base/seed2024` | done |  |
| 2026-10-04T09:45:36+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.5/sad_report/embed_lr__all-MiniLM-L6-v2` | started |  |
| 2026-10-04T09:45:36+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.5/sad_report/embed_lr__all-MiniLM-L6-v2` | done |  |
| 2026-10-04T09:45:36+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.5/sad_report/finetune__distilroberta-base/seed2024` | started |  |
| 2026-10-04T09:45:36+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.5/sad_report/finetune__distilroberta-base/seed2024` | done |  |
| 2026-10-04T10:04:19+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/ad861f36908a1cb4` | started |  |
| 2026-10-04T10:05:08+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/ad861f36908a1cb4` | failed | AssertionError: unit tests failed (see the output above) |
| 2026-10-04T10:05:24+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/ad861f36908a1cb4` | started |  |
| 2026-10-04T10:06:08+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/ad861f36908a1cb4` | failed | AssertionError: unit tests failed (see the output above) |
| 2026-10-04T10:17:30+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/9751651770fba51f` | started |  |
| 2026-10-04T10:18:13+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/9751651770fba51f` | done |  |
| 2026-10-04T10:18:21+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `6A.1/tensistrength_sample` | started |  |
| 2026-10-04T10:18:23+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `6A.1/tensistrength_sample` | done |  |
| 2026-10-04T10:18:23+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `6B.1/senticnet_scores/v20261004` | started |  |
| 2026-10-04T10:19:12+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `6B.1/senticnet_scores/v20261004` | done |  |
| 2026-10-04T10:19:14+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `6B.3/senticnet_zero_shot/v20261004` | started |  |
| 2026-10-04T10:20:26+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `6B.3/senticnet_zero_shot/v20261004` | done |  |
| 2026-10-04T10:26:39+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/3bcaec51fc14fd4a` | started |  |
| 2026-10-04T10:27:26+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/3bcaec51fc14fd4a` | done |  |
| 2026-10-04T10:32:50+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5B.1b/machine_gold_labels` | started |  |
| 2026-10-04T10:42:07+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5B.1b/machine_gold_labels` | done |  |
| 2026-10-04T10:42:08+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/score_gold/keyword__taxonomy-v1` | started |  |
| 2026-10-04T10:42:08+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/score_gold/keyword__taxonomy-v1` | done |  |
| 2026-10-04T10:42:08+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/tune_dev/keyword__taxonomy-v1` | started |  |
| 2026-10-04T10:42:08+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/tune_dev/keyword__taxonomy-v1` | done |  |
| 2026-10-04T10:42:08+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/score_gold/zero_shot__all-MiniLM-L6-v2` | started |  |
| 2026-10-04T10:42:09+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/score_gold/zero_shot__all-MiniLM-L6-v2` | done |  |
| 2026-10-04T10:42:09+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/tune_dev/zero_shot__all-MiniLM-L6-v2` | started |  |
| 2026-10-04T10:42:09+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/tune_dev/zero_shot__all-MiniLM-L6-v2` | done |  |
| 2026-10-04T10:42:09+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/score_gold/embed_lr__all-MiniLM-L6-v2` | started |  |
| 2026-10-04T10:42:10+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/score_gold/embed_lr__all-MiniLM-L6-v2` | done |  |
| 2026-10-04T10:42:10+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/tune_dev/embed_lr__all-MiniLM-L6-v2` | started |  |
| 2026-10-04T10:42:10+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/tune_dev/embed_lr__all-MiniLM-L6-v2` | done |  |
| 2026-10-04T10:42:10+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/score_gold/finetune__distilroberta-base` | started |  |
| 2026-10-04T10:42:12+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/score_gold/finetune__distilroberta-base` | done |  |
| 2026-10-04T10:42:12+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/tune_dev/finetune__distilroberta-base` | started |  |
| 2026-10-04T10:42:12+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.6/tune_dev/finetune__distilroberta-base` | done |  |
| 2026-10-04T10:42:13+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.7/gold_test_report` | started |  |
| 2026-10-04T10:42:14+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.7/gold_test_report` | done |  |
| 2026-10-04T10:42:14+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.8/crosscheck/zero_shot__all-MiniLM-L6-v2` | started |  |
| 2026-10-04T10:42:16+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.8/crosscheck/zero_shot__all-MiniLM-L6-v2` | done |  |
| 2026-10-04T10:42:16+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5D/stressor_bundle` | started |  |
| 2026-10-04T10:42:18+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5D/stressor_bundle` | done |  |
| 2026-10-04T10:42:21+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `6B.1b/machine_handcheck` | started |  |
| 2026-10-04T10:42:25+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `6B.1b/machine_handcheck` | done |  |
| 2026-10-04T16:47:53+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/e66cf390db7491d4` | started |  |
| 2026-10-04T16:48:47+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/e66cf390db7491d4` | done |  |
| 2026-10-04T17:01:14+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/dfaeea6f7f5d3b47` | started |  |
| 2026-10-04T17:02:21+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `2.5/tests/dfaeea6f7f5d3b47` | done |  |
| 2026-10-04T17:02:29+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `5C.8/crosscheck/finetune__distilroberta-base` | done | output found on disk (finished before it was logged) |
| 2026-10-04T17:20:18+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `8.2/ingest/pilot` | done | output found on disk (finished before it was logged) |
| 2026-10-04T17:20:18+00:00 | madhava_laptop (NVIDIA GeForce RTX 4060 Laptop GPU (cuda)) | `8.4/data_quality/pilot` | done | output found on disk (finished before it was logged) |
