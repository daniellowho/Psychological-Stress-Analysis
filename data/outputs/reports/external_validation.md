# External validation (Step 6)

> Aggregate, population-level language signals only. No diagnosis. No individual-level analysis.

Generated 2026-10-04T17:02:33+00:00 for stress bundle `v20261004`. SenticNet labels were produced by an AUTOMATED DNN-based strategy (SenticNet README), not by human annotators. All numbers are AGREEMENT with those labels, not accuracy.

## Verdict (proposed rules, see config)

**SHIP WITH CAVEATS** (PROVISIONAL)

- Provisional because the hand-check was labelled by a machine, not a human; 'ship' cannot be reached until a human has done it.
- Rules: proposed by the assistant, not from the literature; edit config.yaml before reading results.

| id | level | condition | observed | met |
|---|---|---|---|---|
| R1 | retrain | reddit ROC-AUC CI upper bound < 0.7 | AUC 0.954 [0.950, 0.958] | no |
| R2 | retrain | ceiling-relative skill CI upper bound < 0.5 | label-noise ceiling not available | unknown |
| S1 | ship | reddit ROC-AUC CI lower bound >= 0.8 | lower bound 0.950 | yes |
| S2 | ship | label-noise ceiling known and skill >= 0.75 | label-noise ceiling not available (hand-check pending) | unknown |
| S3 | ship | reddit ECE <= 0.1 (else read probabilities as relative, or recalibrate) | ECE 0.149 | no |
| S4 | ship | twitter ROC-AUC point >= 0.7 | AUC 0.788 | yes |
| S5 | ship | the hand-check behind the ceiling was labelled by a human | label source: machine | no |
| C1 | caveat | drop in ROC-AUC vs Dreaddit test is not clearly larger than 0.1 | delta 0.050 [0.028, 0.074] | yes |

Caveats triggered:
- S2: label-noise ceiling known and skill >= 0.75 -> label-noise ceiling not available (hand-check pending)
- S3: reddit ECE <= 0.1 (else read probabilities as relative, or recalibrate) -> ECE 0.149
- S5: the hand-check behind the ceiling was labelled by a human -> label source: machine

### What each verdict means

| verdict | evidence required |
|---|---|
| ship | no retrain condition met AND every ship condition (S1-S5) and caveat check (C1) met, with a completed hand-check |
| ship with caveats | no retrain condition met, but at least one ship condition not met or unknown; the caveats are listed above |
| retrain | any retrain condition (R1, R2) met: the model carries too little rank signal on the primary platform |

## 6B. Zero-shot agreement with SenticNet labels (no fine-tuning)

Model: roberta-base seed 2024, T = 1.372, threshold 0.500. Rows evaluated: 16786 of 17036 (excluded non-English: 250, duplicates: 0).

| set | n | prevalence of label 1 | F1 (stress) | ROC-AUC | PR-AUC | ECE |
|---|---|---|---|---|---|---|
| Dreaddit test (reference) | 715 | 0.516 | 0.824 [0.790, 0.855] | 0.905 [0.880, 0.925] | 0.916 [0.891, 0.936] | 0.055 [0.047, 0.090] |
| SenticNet reddit | 8511 | 0.626 | 0.833 [0.825, 0.841] | 0.954 [0.950, 0.958] | 0.970 [0.967, 0.974] | 0.149 [0.142, 0.157] |
| SenticNet twitter | 8275 | 0.511 | 0.187 [0.172, 0.202] | 0.788 [0.778, 0.798] | 0.816 [0.804, 0.828] | 0.395 [0.385, 0.405] |
| Reddit_Combi | 3118 | 0.879 | 0.940 [0.934, 0.947] | 0.942 [0.929, 0.954] | 0.990 [0.987, 0.993] | 0.110 [0.101, 0.120] |
| Reddit_Title | 5393 | 0.481 | 0.695 [0.680, 0.711] | 0.942 [0.936, 0.948] | 0.933 [0.925, 0.941] | 0.172 [0.162, 0.181] |
| Twitter_Full | 8275 | 0.511 | 0.187 [0.172, 0.202] | 0.788 [0.778, 0.798] | 0.816 [0.804, 0.828] | 0.395 [0.385, 0.405] |
| Twitter_Full (Non-Advert rows only) | 1935 | 0.616 | 0.077 [0.058, 0.096] | 0.773 [0.752, 0.793] | 0.867 [0.850, 0.883] | 0.519 [0.497, 0.541] |

Difference to Dreaddit test (SenticNet minus Dreaddit; negative = lower on SenticNet), 95% CI:

| set | delta F1 | delta ROC-AUC | delta PR-AUC | delta ECE |
|---|---|---|---|---|
| reddit | 0.009 [-0.022, 0.045] | 0.050 [0.028, 0.074] | 0.055 [0.034, 0.080] | 0.094 [0.060, 0.103] |
| twitter | -0.637 [-0.672, -0.600] | -0.116 [-0.141, -0.090] | -0.099 [-0.123, -0.073] | 0.340 [0.301, 0.350] |
| Reddit_Combi | 0.116 [0.085, 0.151] | 0.037 [0.013, 0.064] | 0.074 [0.054, 0.099] | 0.055 [0.018, 0.064] |
| Reddit_Title | -0.129 [-0.164, -0.091] | 0.037 [0.015, 0.062] | 0.017 [-0.005, 0.042] | 0.117 [0.081, 0.126] |
| Twitter_Full | -0.637 [-0.672, -0.600] | -0.116 [-0.141, -0.090] | -0.099 [-0.123, -0.073] | 0.340 [0.301, 0.350] |
| Twitter_Full (Non-Advert rows only) | -0.747 [-0.785, -0.707] | -0.131 [-0.159, -0.101] | -0.048 [-0.073, -0.018] | 0.464 [0.419, 0.483] |

Differences mix domain shift, label noise, prevalence and length; F1 and PR-AUC move with prevalence. 

## 6B. Label noise and the ceiling on agreement

> **The hand-check was labelled by a zero-shot NLI model (machine_labeler), NOT by a person. The 'label noise' and the ceiling below are therefore agreement between two automated systems, not an estimate of true label error.**

| platform | n | human-auto agreement | est. label noise | ceiling ROC-AUC | ceiling accuracy | ceiling F1 |
|---|---|---|---|---|---|---|
| reddit | 24 | 0.625 [0.427, 0.788] | 0.375 | 0.625 [0.500, 0.750] | 0.720 [0.626, 0.813] | 0.817 [0.770, 0.870] |
| twitter | 24 | 0.667 [0.467, 0.820] | 0.333 | 0.667 [0.500, 0.833] | 0.670 [0.502, 0.837] | 0.721 [0.559, 0.863] |

No classifier, however good, should be expected to exceed the ceiling on agreement with these labels; a result at the ceiling is the best possible. The ceiling assumes the human label is the truth, which one annotator cannot guarantee.

## 6A. TensiStrength auxiliary signal

| set | n | Spearman(stress strength, P(stress)) | Spearman(relaxation, P(stress)) | share with any stress term |
|---|---|---|---|---|
| dreaddit_test | 715 | 0.522 [0.468, 0.577] | -0.175 [-0.248, -0.099] | 0.899 |
| senticnet_reddit | 1000 | 0.599 [0.558, 0.637] | -0.114 [-0.176, -0.053] | 0.673 |
| senticnet_twitter | 1000 | 0.472 [0.423, 0.521] | -0.347 [-0.405, -0.295] | 0.573 |

Spearman between two scorers = agreement. A lexicon with no labelled validation data cannot confirm the model. Many texts get stress_strength 1 (ties), which lowers rho.

Speed on this machine: dreaddit_test: 715 texts in 0.37 s (1947.5/s) in one Java process; senticnet: 2000 texts in 0.51 s (3948.6/s) in one Java process; one Java launch per text: 0.15 s each.

### Why not run TensiStrength on every record

- Java dependency: every machine that runs the pipeline needs a compatible JRE, and Spark workers would each need one too.
- Process startup: a JVM launch per record (or per Spark task) costs far more than the lexicon lookup itself. The probe cell measures seconds per single launch against texts per second in one batch on THIS machine; those two numbers (in external_validation.json) are the evidence, not an assumption.
- Licence: free for academic use only (GBP 1000 commercial), so it should not be a hard dependency of the pipeline.
- No labelled validation data ships with it, so its scores cannot be verified, only compared to other signals.
- A lexicon tool adds a second, differently-biased signal at the cost of a second runtime; for millions of records the Step 3 model and, if wanted, a Python re-implementation of a lexicon feature would be the scalable route.

## Optional: cross-training

NOT RUN (config external_validation.cross_training.enabled is false, or no results yet).
Dreaddit -> SenticNet is the zero-shot result above.


## Limitations

- SenticNet labels are automated. Agreement with them measures similarity to another model, not correctness.
- The label-noise estimate rests on a small hand-check by one annotator; its confidence intervals are wide.
- F1 and PR-AUC depend on label prevalence, which differs between Dreaddit and each SenticNet file. ROC-AUC is the least prevalence-sensitive number here.
- Differences from Dreaddit test mix domain shift, label noise, prevalence and text-length differences; this analysis cannot separate them.
- Calibration (ECE) against noisy labels is inflated even for a perfect model; read it as 'probabilities may need recalibration on this domain', not as a measured error rate.
- TensiStrength is a lexicon tool with no labelled validation data in its repository. A correlation shows the two signals move together, not that either measures psychological stress.
- Nothing here assesses any individual. All results are over sets of texts.
