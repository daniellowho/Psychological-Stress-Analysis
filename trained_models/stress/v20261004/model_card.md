# Model card: stress-language classifier (roberta-base, seed 2024)

**What it is.** A binary classifier of *stress-related language* in short English Reddit-style text, fine-tuned from
`roberta-base` on Dreaddit text only. Its outputs feed **aggregate, population-level** signals.

**Not for diagnosis.** It does not detect mental illness or assess any individual. Do not use it for screening,
profiling, ranking, moderation or any decision about a person. Report results only as aggregates, with cells of n < K_MIN suppressed.

**Data.** Dreaddit (Turcan & McKeown, 2019): 2017–2018 posts from 10 subreddits in 5 domains (abuse, anxiety,
financial, PTSD, social). The labels are crowd-annotated segments. The HF card states no licence (UNVERIFIED). Splits: official test kept as the final
test; train/validation grouped by post_id. Label map: {0: 'not_stress', 1: 'stress'} (from the Dreaddit paper; the meaning is not in the HF card).

**Final test (calibrated, threshold 0.50 chosen on validation, T = 1.372).**
F1(stress) 0.824 [95% CI 0.790–0.855],
macro-F1 0.812, ROC-AUC 0.905, PR-AUC 0.916, accuracy 0.813,
ECE 0.055, n = 715. Across seeds (uncalibrated, threshold 0.5): F1(stress) 0.819 ± 0.014 (n=3 runs).
Threshold: searched 0.40-0.60 on validation (tuned value
0.47); applied = 0.5 (gain 0.0028 not clear (min_gain 0.01, CI [-0.0037, 0.0110]); keeping 0.5).
Test F1(stress) at 0.5: 0.824; at the tuned threshold: 0.833.
TF-IDF + LR baseline on the same test set (threshold 0.5): F1(stress) 0.752, ROC-AUC 0.832.

**Inference.** Input is `text_clean` from Step 2 cleaning; max 256 tokens. Longer posts are split into sentence-aligned chunks (stride
64), and the post score is the mean of calibrated chunk probabilities. fp32 by default.

**Limitations.**
- Domain shift: Dreaddit segments are short (median ~99 tokens) and come from 5 domains in 2017–2018. Target communities, years
  (2019–2022, COVID) and post lengths differ, so calibration and accuracy may not transfer. Re-check on a labelled sample.
- Topic shortcuts: see the masking and leave-one-subreddit-out results in metrics.json. The model may partly key on topic words
  (money, abuse) rather than stress expression.
- The labels are annotator judgements of whether a segment expresses stress. They are not clinical ground truth. Some segments have low annotator agreement.
- English only. No fairness audit across demographic groups was possible (no such attributes are available, by design).
