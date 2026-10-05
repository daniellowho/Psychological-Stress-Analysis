# Stressor taxonomy v1

_Stressor labels are a RESEARCH taxonomy of what a text names as a cause of pressure. They are not a clinical assessment, say nothing about any person, and feed aggregate signals only._

Source of truth: `code/taxonomy/stressors_v1.yaml` (definitions, include/exclude rules, seed keywords, 3 paraphrase descriptions per category). Keywords and paraphrases are author-chosen and unvalidated.

## Categories

| id | name | coverage | how it is scored | SAD labels | Mendeley columns | Dreaddit subreddits (weak check) | n keywords |
|---|---|---|---|---|---|---|---|
| workplace_pressure | Workplace pressure | sad | SAD-supervised (or keyword/zero-shot baselines) | Work | work_environment | - | 23 |
| academic_workload | Academic workload | sad | SAD-supervised (or keyword/zero-shot baselines) | School | academic_overload, peer_competition, professor_difficulties, academic_conflicts | - | 28 |
| financial_concerns | Financial concerns | sad | SAD-supervised (or keyword/zero-shot baselines) | Financial Problem | - | almosthomeless, assistance, food_pantry, homeless | 24 |
| relationship_issues | Relationship issues | sad | SAD-supervised (or keyword/zero-shot baselines) | Family Issues, Social Relationships | relationship_stress, home_environment | relationships, domesticviolence, survivorsofabuse | 29 |
| health_concerns | Health concerns | sad | SAD-supervised (or keyword/zero-shot baselines) | Health, Fatigue, or Physical Pain | health_issues | - | 26 |
| uncertainty | Uncertainty | zero_shot | zero-shot embedding similarity ONLY (lower confidence) | - | - | - | 12 |
| overload | Overload | sad | SAD-supervised (or keyword/zero-shot baselines) | Everyday Decision Making | lack_relaxation_time | - | 16 |
| societal_problems | Societal / social problems | zero_shot | zero-shot embedding similarity ONLY (lower confidence) | - | - | - | 19 |
| other_unclear | Other / unclear stressor | sad | SAD-supervised (or keyword/zero-shot baselines) | Other | - | - | 17 |

## SAD categories -> taxonomy

| SAD label | n (top_label) | classifier class | taxonomy category | sub-category | stressor output |
|---|---|---|---|---|---|
| Work | 1339 | workplace_pressure | workplace_pressure |  | yes |
| School | 742 | academic_workload | academic_workload |  | yes |
| Financial Problem | 634 | financial_concerns | financial_concerns |  | yes |
| Family Issues | 742 | relationship_issues/family | relationship_issues | family | yes |
| Social Relationships | 629 | relationship_issues/social | relationship_issues | social | yes |
| Health, Fatigue, or Physical Pain | 780 | health_concerns | health_concerns |  | yes |
| Everyday Decision Making | 337 | overload | overload |  | yes |
| Emotional Turmoil | 667 | emotional_turmoil | (none: not a stressor) |  | no (discarded at post level) |
| Other | 973 | other_unclear | other_unclear |  | yes |

## Decisions

- **everyday_decision_making**: SAD 'Everyday Decision Making' -> overload (USER DECISION, Step 5). Inspection of SAD sentences showed mostly planning/logistics load (packing for a trip, planning meals, sorting out a schedule) plus some general 'too much on my mind' text; uncertainty-about-the-future sentences also occur in it, so the overload head will absorb some uncertainty language (documented limitation). Uncertainty itself has no SAD label.
- **emotional_turmoil**: SAD 'Emotional Turmoil' is an emotional RESPONSE, not a cause, so it is NEVER a stressor output. It is KEPT as a separate non-stressor class in the supervised classifiers (not dropped). Why: Reddit stress posts are full of feelings-only sentences ('I just had a breakdown'); a classifier trained without that class must spread every such sentence over the stressor classes (softmax sums to 1), producing false stressors. With the class kept, that probability mass goes to a non-stressor output that is discarded at post level. Dropping its 667 rows would also remove SAD's best 'no identifiable cause' examples. Emotion language is measured by the Step 4 model instead.
- **sleep_problems**: NOT split from health_concerns. SAD has no sleep label: a sleep sub-label would have to be created by keyword matching (105 of 780 'Health, Fatigue, or Physical Pain' sentences match sleep|slept|insomnia|awake|nightmare*, Step 5 inspection), i.e. a keyword-derived label rather than a human one, and Mendeley's authors list sleep_problems as a SYMPTOM, not a stressor. Kept combined; sleep language is part of health_concerns.
- **sad_non_stressor_rows**: SAD's 373 rows with is_stressor = 0 (365 of them top_label 'Other') keep their rater category; they are few and the 'Other' class is mapped to other_unclear, which is a low-information class anyway.
- **severity**: SAD avg/median severity is NOT used as a label (optional auxiliary information only, unused in v1).

## Categories SAD does not cover

uncertainty, societal_problems: no SAD label exists, so only the keyword baseline and zero-shot embedding similarity (window embedding vs the mean embedding of the category definition + 3 paraphrases) can score them. In the SAD-trained methods these columns come from the zero-shot scorer (hybrid). They are flagged lower-confidence everywhere, and their thresholds are tuned on the small gold-dev set only.

## Mendeley columns: cause vs response (structure only, no statistics)

| column | authors' group | role | category | reason |
|---|---|---|---|---|
| gender | demographics | not_used |  | demographic attribute |
| age | demographics | not_used |  | demographic attribute |
| stress_type | target | not_used |  | outcome label (distress / eustress / mixed), not a cause |
| stress_experience | symptoms_emotional_responses | symptom_response |  | self-rated stress level: an outcome |
| heartbeat_palpitations | symptoms_emotional_responses | symptom_response |  | bodily stress reaction |
| anxiety_tension | symptoms_emotional_responses | symptom_response |  | emotional reaction |
| sleep_problems | symptoms_emotional_responses | symptom_response |  | reaction; supports keeping sleep inside health_concerns rather than a stressor of its own |
| restlessness | symptoms_emotional_responses | symptom_response |  | behavioural reaction |
| irritability | symptoms_emotional_responses | symptom_response |  | emotional reaction |
| sadness_low_mood | symptoms_emotional_responses | symptom_response |  | emotional reaction |
| loneliness_isolation | symptoms_emotional_responses | symptom_response |  | a feeling; social isolation stated as a SITUATION is covered by relationship_issues in text |
| concentration_problems | symptoms_emotional_responses | symptom_response |  | cognitive reaction |
| headaches | health_indicators | symptom_response |  | bodily reaction |
| weight_changes | health_indicators | symptom_response |  | bodily reaction |
| health_issues | health_indicators | stressor | health_concerns | a health problem is a situation that causes pressure (AMBIGUOUS: can also be a consequence) |
| academic_overload | stressors_academic_environment | stressor | academic_workload | external academic demand |
| peer_competition | stressors_academic_environment | stressor | academic_workload | external academic pressure |
| professor_difficulties | stressors_academic_environment | stressor | academic_workload | external academic relationship problem |
| academic_conflicts | stressors_academic_environment | stressor | academic_workload | external academic conflict |
| work_environment | stressors_academic_environment | stressor | workplace_pressure | external work condition |
| home_environment | stressors_academic_environment | stressor | relationship_issues | home/family conditions (AMBIGUOUS: could also be housing quality -> financial_concerns) |
| relationship_stress | stressors_social_relationship | stressor | relationship_issues | external relationship problem |
| lack_relaxation_time | stressors_social_relationship | stressor | overload | too many demands for the time available |
| low_academic_confidence | stressors_academic_environment | appraisal_not_stressor |  | DISAGREES with the authors' grouping: a self-appraisal (how the student judges themself), i.e. a cognitive response, not an external cause |
| subject_confidence | stressors_academic_environment | appraisal_not_stressor |  | DISAGREES with the authors' grouping: self-appraisal, as above |
| class_attendance | stressors_academic_environment | behaviour_not_stressor |  | DISAGREES with the authors' grouping: a behaviour that can be a consequence of stress, not its cause |
