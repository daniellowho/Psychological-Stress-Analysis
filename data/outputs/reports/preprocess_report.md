# Step 2: preprocessing report

Counts only; every number comes from `data/outputs/manifests/step2_preprocess_manifest.json`. No text is included.

## dreaddit

Final rows: **3528**

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 3553 |  |
| drop:empty_after_cleaning | 0 | 3553 |  |
| drop:train_exact_duplicate_text | 22 | 3531 | {"conflicting_label_groups": 3, "conflicting_label_rows_dropped": 8, "duplicate_rows_dropped": 14} |
| drop:train_text_also_in_test | 3 | 3528 |  |
| official_test_kept_unchanged |  |  | {"test_duplicate_rows": 0} |
| lang_computed_not_filtered |  |  | {"counts": {"en": 3527, "eu": 1}} |

Split counts: `{"test": 715, "train": 2398, "validation": 415}`

Label by split: `{"test": {"0": 346, "1": 369}, "train": {"0": 1151, "1": 1247}, "validation": {"0": 181, "1": 234}}`

post_id overlap between splits (must be 0): `{"test&train": 0, "test&validation": 0, "train&validation": 0}`

Cleaning edits (counts of substitutions): `{"html_unescape": 55, "md_bullet": 22, "md_em": 87, "md_escape": 137, "md_heading": 1, "md_link": 2, "md_quote": 4, "md_strong": 210, "md_superscript": 1, "mention": 8, "phone_removed": 4, "url": 1, "url_placeholder": 344, "zero_width": 48}`

## goemotions

Final rows: **54263**

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 54263 |  |
| drop:empty_after_cleaning | 0 | 54263 |  |
| lang_computed_not_filtered |  |  | {"counts": {"af": 81, "cy": 91, "en": 52289, "fr": 80, "it": 59, "la": 142, "nl": 114, "st": 112, "tl": 192, "tn": 84}} |
| official_splits_kept_unchanged |  |  | {"cross_split_shared_texts": {"train&test": 37, "train&validation": 47, "validation&test": 11}, "within_train_duplicate_rows": 223} |

Split counts: `{"test": 5427, "train": 43410, "validation": 5426}`

Cleaning edits (counts of substitutions): `{"html_unescape": 2, "md_bullet": 14, "md_code": 2, "md_em": 694, "md_escape": 1, "md_heading": 8, "md_link": 2, "md_quote": 639, "md_strong": 203, "md_superscript": 347, "mention": 77, "phone_removed": 2, "removed_marker": 1, "url": 6, "url_placeholder": 8, "zero_width": 62}`

## sad

Final rows: **6843**

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 6850 |  |
| drop:empty_after_cleaning | 0 | 6850 |  |
| drop:exact_duplicate_text | 7 | 6843 | {"conflicting_label_groups": 0, "conflicting_label_rows_dropped": 0, "duplicate_rows_dropped": 7} |
| lang_computed_not_filtered |  |  | {"counts": {"cy": 3, "en": 6761, "fr": 8, "it": 2, "la": 19, "nl": 3, "ro": 3, "sn": 3, "st": 8, "tl": 6}} |

Cleaning edits (counts of substitutions): `{"md_code": 1, "md_em": 1, "url": 1, "url_placeholder": 1}`

## zenodo

Final rows: **170199**

### COVID19_support_post_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 981 |  |
| drop:unparseable_date | 0 | 981 |  |
| drop:empty_or_removed_after_cleaning | 0 | 981 |  |
| drop:bot_text | 0 | 981 |  |
| drop:shorter_than_5_tokens | 1 | 980 |  |
| drop:non_english | 0 | 980 | {"top_detected": {"en": 980}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### anxiety_post_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 15896 |  |
| drop:unparseable_date | 0 | 15896 |  |
| drop:empty_or_removed_after_cleaning | 0 | 15896 |  |
| drop:bot_text | 0 | 15896 |  |
| drop:shorter_than_5_tokens | 0 | 15896 |  |
| drop:non_english | 1 | 15895 | {"top_detected": {"en": 15895, "tl": 1}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### anxiety_pre_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 19976 |  |
| drop:unparseable_date | 0 | 19976 |  |
| drop:empty_or_removed_after_cleaning | 0 | 19976 |  |
| drop:bot_text | 0 | 19976 |  |
| drop:shorter_than_5_tokens | 0 | 19976 |  |
| drop:non_english | 3 | 19973 | {"top_detected": {"en": 19973, "nl": 1, "st": 1, "yo": 1}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### divorce_post_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 2747 |  |
| drop:unparseable_date | 0 | 2747 |  |
| drop:empty_or_removed_after_cleaning | 0 | 2747 |  |
| drop:bot_text | 0 | 2747 |  |
| drop:shorter_than_5_tokens | 1 | 2746 |  |
| drop:non_english | 0 | 2746 | {"top_detected": {"en": 2746}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### divorce_pre_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 6419 |  |
| drop:unparseable_date | 0 | 6419 |  |
| drop:empty_or_removed_after_cleaning | 0 | 6419 |  |
| drop:bot_text | 0 | 6419 |  |
| drop:shorter_than_5_tokens | 0 | 6419 |  |
| drop:non_english | 0 | 6419 | {"top_detected": {"en": 6419}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### parenting_post_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 7045 |  |
| drop:unparseable_date | 0 | 7045 |  |
| drop:empty_or_removed_after_cleaning | 0 | 7045 |  |
| drop:bot_text | 0 | 7045 |  |
| drop:shorter_than_5_tokens | 0 | 7045 |  |
| drop:non_english | 0 | 7045 | {"top_detected": {"en": 7045}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### parenting_pre_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 17371 |  |
| drop:unparseable_date | 0 | 17371 |  |
| drop:empty_or_removed_after_cleaning | 0 | 17371 |  |
| drop:bot_text | 0 | 17371 |  |
| drop:shorter_than_5_tokens | 0 | 17371 |  |
| drop:non_english | 0 | 17371 | {"top_detected": {"en": 17371}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### personalfinance_post_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 37165 |  |
| drop:unparseable_date | 0 | 37165 |  |
| drop:empty_or_removed_after_cleaning | 0 | 37165 |  |
| drop:bot_text | 0 | 37165 |  |
| drop:shorter_than_5_tokens | 0 | 37165 |  |
| drop:non_english | 1 | 37164 | {"top_detected": {"en": 37164, "la": 1}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### personalfinance_pre_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 23440 |  |
| drop:unparseable_date | 0 | 23440 |  |
| drop:empty_or_removed_after_cleaning | 0 | 23440 |  |
| drop:bot_text | 0 | 23440 |  |
| drop:shorter_than_5_tokens | 0 | 23440 |  |
| drop:non_english | 0 | 23440 | {"top_detected": {"en": 23440}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### relationships_post_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 20767 |  |
| drop:unparseable_date | 0 | 20767 |  |
| drop:empty_or_removed_after_cleaning | 0 | 20767 |  |
| drop:bot_text | 0 | 20767 |  |
| drop:shorter_than_5_tokens | 0 | 20767 |  |
| drop:non_english | 0 | 20767 | {"top_detected": {"en": 20767}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### relationships_pre_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 15037 |  |
| drop:unparseable_date | 0 | 15037 |  |
| drop:empty_or_removed_after_cleaning | 0 | 15037 |  |
| drop:bot_text | 0 | 15037 |  |
| drop:shorter_than_5_tokens | 0 | 15037 |  |
| drop:non_english | 0 | 15037 | {"top_detected": {"en": 15037}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### teaching_post_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 1063 |  |
| drop:unparseable_date | 0 | 1063 |  |
| drop:empty_or_removed_after_cleaning | 0 | 1063 |  |
| drop:bot_text | 0 | 1063 |  |
| drop:shorter_than_5_tokens | 0 | 1063 |  |
| drop:non_english | 0 | 1063 | {"top_detected": {"en": 1063}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### teaching_pre_features_tfidf_256

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 2350 |  |
| drop:unparseable_date | 0 | 2350 |  |
| drop:empty_or_removed_after_cleaning | 0 | 2350 |  |
| drop:bot_text | 0 | 2350 |  |
| drop:shorter_than_5_tokens | 0 | 2350 |  |
| drop:non_english | 1 | 2349 | {"top_detected": {"en": 2349, "eo": 1}} |
| outside_observed_window_kept |  |  | {"n": 0} |

### all files

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 170249 |  |
| drop:exact_duplicate_text_global | 50 | 170199 | {"conflicting_label_groups": 0, "conflicting_label_rows_dropped": 0, "duplicate_rows_dropped": 50} |

Rows by community × window:

| community | window | n |
|---|---|---|
| anxiety | post | 15892 |
| anxiety | pre | 19969 |
| covid19_support | post | 977 |
| divorce | post | 2744 |
| divorce | pre | 6418 |
| parenting | post | 7043 |
| parenting | pre | 17365 |
| personalfinance | post | 37159 |
| personalfinance | pre | 23438 |
| relationships | post | 20758 |
| relationships | pre | 15025 |
| teaching | post | 1063 |
| teaching | pre | 2348 |

Cleaning edits (counts of substitutions): `{"email_removed": 142, "html_unescape": 28848, "md_bullet": 18126, "md_code": 77, "md_em": 10739, "md_escape": 19339, "md_heading": 369, "md_link": 5897, "md_quote": 1265, "md_rule": 863, "md_strong": 12839, "md_superscript": 485, "mention": 189, "phone_removed": 111, "url": 5146, "url_placeholder": 11237, "zero_width": 29206}`

## senticnet

Final rows: **17036**

### Reddit_Title

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 5556 |  |
| drop:empty_text_or_invalid_label | 0 | 5556 |  |
| drop:within_file_duplicates | 53 | 5503 | {"duplicate_rows_dropped": 33, "conflicting_label_groups": 10, "conflicting_label_rows_dropped": 20} |
| drop:exact_dreaddit_text | 0 | 5503 |  |
| drop:contains_dreaddit_segment_ge_80% | 0 | 5503 |  |

### Reddit_Combi

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 3123 |  |
| drop:empty_text_or_invalid_label | 0 | 3123 |  |
| drop:within_file_duplicates | 0 | 3123 | {"duplicate_rows_dropped": 0, "conflicting_label_groups": 0, "conflicting_label_rows_dropped": 0} |
| drop:exact_dreaddit_text | 0 | 3123 |  |
| drop:contains_dreaddit_segment_ge_80% | 5 | 3118 |  |

### Twitter_Full

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 8900 |  |
| drop:empty_text_or_invalid_label | 0 | 8900 |  |
| drop:within_file_duplicates | 485 | 8415 | {"duplicate_rows_dropped": 485, "conflicting_label_groups": 0, "conflicting_label_rows_dropped": 0} |
| drop:exact_dreaddit_text | 0 | 8415 |  |
| is_non_advert_flag_added |  |  | {"rows_flagged": 1967, "non_advert_texts": 1967, "non_advert_texts_without_match_in_full": 0} |

### Twitter_Non-Advert

| step | dropped | rows after | details |
|---|---|---|---|
| loaded |  | 2051 |  |
| drop:empty_text_or_invalid_label | 0 | 2051 |  |
| drop:within_file_duplicates | 84 | 1967 | {"duplicate_rows_dropped": 84, "conflicting_label_groups": 0, "conflicting_label_rows_dropped": 0} |
| drop:exact_dreaddit_text | 0 | 1967 |  |
| drop:not_kept_as_separate_set | 1967 | 0 |  |

Automated label counts by file: `{"Reddit_Combi": {"0": 378, "1": 2740}, "Reddit_Title": {"0": 2801, "1": 2702}, "Twitter_Full": {"0": 4117, "1": 4298}}`

Cleaning edits (counts of substitutions): `{"html_unescape": 2, "md_bullet": 3, "md_em": 150, "md_escape": 39, "md_quote": 2, "md_strong": 97, "md_superscript": 11, "mention": 8, "phone_removed": 154, "url": 63, "url_placeholder": 63, "zero_width": 248}`
