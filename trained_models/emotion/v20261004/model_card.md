# Model card: emotion-language classifier (roberta-base, seed 13)

**What it is.** A multi-label classifier (28 independent sigmoid outputs: 27 emotions + neutral) of the emotions
*expressed in* short English Reddit-style text, fine-tuned from `roberta-base` on GoEmotions. Outputs feed
**aggregate, population-level** signals only.

**Not for diagnosis, and not a stress detector.** It does not assess any individual. A negative emotion is not stress,
and emotion language is not an emotional state. Do not use it for screening, profiling, ranking or moderation.

**Data.** GoEmotions (Demszky et al., 2020), official google-research files (Apache-2.0), Reddit comments, 2019.
Official splits {'test': 5427, 'train': 43410, 'validation': 5426}. Labels are rater judgements; many emotions are rare.

**Final test (calibrated: platt per label; per-label thresholds tuned on validation).**
macro-F1 0.480 [95% CI 0.464–0.494],
micro-F1 0.603 [0.593–0.613], macro ROC-AUC 0.923,
macro AP 0.495, n = 5427. Same calibrated probabilities at a global 0.5: macro-F1
0.445. Pooled ECE uncalibrated 0.005 →
calibrated 0.002. Across seeds (uncalibrated, threshold 0.5): macro-F1 0.445 ± 0.010, macro AP 0.493 ± 0.004 (n=3 runs).
Weakest labels (test F1): grief 0.00 (n=6), nervousness 0.00 (n=23), relief 0.00 (n=11), realization 0.22 (n=145), pride 0.30 (n=16). Low-support labels (threshold noisy): grief, nervousness, pride, relief.
Without the 48 test rows whose text also occurs in train/validation: macro-F1 0.479.

**Ekman view (dashboard).** group probability = max of member calibrated probabilities (lower bound of P(any member), no independence assumption); group predicted = any member above its own threshold; group gold = any member labelled. Test macro-F1 over the 7 groups 0.631, macro ROC-AUC
0.915.

**Domain shift.** Dreaddit check (inference only, n = 1850 stress / 1678 not-stress segments): largest differences in mean calibrated Ekman probability (stress − not stress): joy -0.161 [-0.178, -0.144]; fear +0.156 [+0.143, +0.172]; neutral -0.141 [-0.154, -0.127]. Plausibility hand-check: see the local sheet.
GoEmotions is short comments; the target corpora are longer posts (sentence-aligned chunks of ≤ 128 tokens,
mean of calibrated chunk probabilities). Calibration was fitted on GoEmotions validation and is not guaranteed on posts.

**Limitations.**
- Rare labels (e.g. grief, pride, relief, nervousness) have few examples; their F1 and thresholds are unstable (see CIs).
- Rater disagreement in GoEmotions is high for several emotions; labels are not ground truth about the writer.
- English only; 2019 Reddit comments; no demographic fairness audit is possible (no such attributes, by design).
- The Ekman group probability is a rule over member probabilities, not a separately calibrated model.
