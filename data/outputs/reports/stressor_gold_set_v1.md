# Stressor gold set v1

_Stressor labels are a RESEARCH taxonomy of what a text names as a cause of pressure. They are not a clinical assessment, say nothing about any person, and feed aggregate signals only._

**Annotation: MACHINE-LABELLED by a zero-shot NLI model (not a human annotation); no inter-annotator agreement.** The 'gold' labels were produced by a zero-shot NLI model (machine_labeler), NOT by a person. Every gold metric is AGREEMENT WITH THAT LABELLER, not correctness, and inherits its errors; it cannot show that the stressor classifier is right on Reddit. Treat all stressor results as provisional until a human annotates the sheet.

**Sampling.** Dreaddit HUMAN label 1 (stress), confidence >= min_confidence; one segment per post; stratified by subreddit (floor, rest proportional); dev/test stratified by subreddit; test never used for tuning. Settings: `{"annotators": 1, "dev_fraction": 0.5, "dreaddit_splits": ["train", "validation"], "floor_per_subreddit": 20, "min_confidence": 0.6, "n_total": 400, "one_segment_per_post": true}`, seed 42.

**Leakage.** Gold texts come ONLY from Dreaddit's train and validation splits (stress label 1). Dreaddit's official test split, on which the Step 3 stress model was evaluated once (3D.2, written before this gold set existed), is never sampled, so that evaluation is untouched. Gold labels never train or tune the stress model. The stressor models never train on Dreaddit; gold-dev only tunes stressor thresholds and picks the method; gold-test is scored once. Caveat: the Step 3 model was TRAINED on these segments, so its stress probabilities on gold texts are in-sample and must not be evaluated on the gold set (dreaddit_split is recorded per gold row).

**Size.** 400 segments: {'dev': 200, 'test': 200}.

## Segments per subreddit and split

| community | dev | test |
|---|---|---|
| almosthomeless | 12 | 12 |
| anxiety | 37 | 37 |
| assistance | 17 | 18 |
| domesticviolence | 25 | 24 |
| food_pantry | 5 | 6 |
| homeless | 13 | 13 |
| ptsd | 36 | 35 |
| relationships | 28 | 28 |
| stress | 11 | 11 |
| survivorsofabuse | 16 | 16 |

## Annotation progress and support

Annotated 400 of 400 rows (0 problems).

| column | positives_dev | positives_test | coverage |
|---|---|---|---|
| workplace_pressure | 193 | 187 | sad |
| academic_workload | 30 | 22 | sad |
| financial_concerns | 77 | 64 | sad |
| relationship_issues | 190 | 188 | sad |
| health_concerns | 134 | 134 | sad |
| uncertainty | 198 | 197 | zero_shot |
| overload | 129 | 121 | sad |
| societal_problems | 144 | 144 | zero_shot |
| other_unclear | 0 | 0 | sad |
| none_unclear | 0 | 0 | gold-only column |
