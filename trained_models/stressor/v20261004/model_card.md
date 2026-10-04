# Model card: stressor-language classifier (finetune__distilroberta-base)

**What it is.** A multi-label tagger of the stressor categories a text *names as a cause of pressure*
(taxonomy v1, 9 categories). Outputs feed **aggregate, population-level**
signals only.

**Stressor labels are a research taxonomy, not a clinical assessment.** They describe language in sampled text. They do
not diagnose, assess or describe any person, and must not be used for screening, profiling, ranking or moderation.

**Method.** finetune. Posts are split into sentence windows; each window is scored; post score = max
over windows; per-category thresholds tuned on the Reddit gold-dev set; no category above threshold -> none_unclear.
Categories without SAD training labels (uncertainty, societal_problems) are scored by zero-shot embedding similarity and are
**lower confidence**. Embedding model: sentence-transformers/all-MiniLM-L6-v2.

**Data.** Training: SAD v1 (MIT; crowd-written and LiveJournal sentences, no Reddit). Evaluation: hand-labelled gold set
of Dreaddit stress-positive segments (train/validation splits only), split 50/50 into dev (tuning) and test (scored once).

**Gold-test (n = 200).** macro-F1 over categories with test positives 0.387
[95% CI 0.360–0.413]; micro-F1 0.359;
SAD-covered categories 0.516; zero-shot-only categories 0.000;
none_unclear F1 0.000. Per category: workplace_pressure 0.33 (test n=187, dev n=193), academic_workload 0.77 (test n=22, dev n=30), financial_concerns 0.73 (test n=64, dev n=77), relationship_issues 0.79 (test n=188, dev n=190), health_concerns 0.44 (test n=134, dev n=134), uncertainty 0.00 (test n=197, dev n=198), overload 0.03 (test n=121, dev n=129), societal_problems 0.00 (test n=144, dev n=144), other_unclear 0.00 (test n=0, dev n=0).
Low-support categories (< 10 dev positives; default threshold, or one global threshold for cosine scores):
other_unclear.

**Domain shift.** SAD-test category macro-F1 0.725 vs Reddit gold-test 0.516
(drop 0.209; approximate: different units and decision rules).

**Weak sanity check (not accuracy).** Predicted rates in mapped vs other Dreaddit subreddits: financial_concerns: mapped 0.703 vs other 0.089; relationship_issues: mapped 0.904 vs other 0.466.

**Gold labels: MACHINE-LABELLED by a zero-shot NLI model (not a human annotation); no inter-annotator agreement.**

**Limitations.**
- The 'gold' labels were produced by a zero-shot NLI model (machine_labeler), NOT by a person. Every gold metric is AGREEMENT WITH THAT LABELLER, not correctness, and inherits its errors; it cannot show that the stressor classifier is right on Reddit. Treat all stressor results as provisional until a human annotates the sheet.
- SAD is short, mostly non-Reddit, 94.5% stressor-positive text; the gold set is small (~200 test posts) and comes from
  10 Dreaddit subreddits (2017-2018), so per-category F1 has wide CIs and may not transfer to other communities or years.
- Max aggregation favours long posts (more windows, more chances to pass a threshold).
- 'Everyday Decision Making' was mapped to overload (user decision), so overload also absorbs some uncertainty language.
- Seed keywords and paraphrase descriptions are author-chosen and unvalidated.
- English only; no demographic fairness audit is possible (no such attributes, by design).
