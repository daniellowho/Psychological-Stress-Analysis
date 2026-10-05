# Data quality report: corpus `pilot`

Aggregate counts only. No text, no identifiers. Cells with n < k_min are suppressed in the CSVs (k_min = 50); the tables below show only communities and months at or above k_min.

- rows in (raw): **170,257** | rows out (lake): **170,199** | row accounting balanced: **True**
- missing / unparseable timestamps dropped: **0**
- exact duplicate rate (duplicates dropped / rows reaching the dedupe step): **0.03%** (near-duplicates are NOT removed)
- days with data: 477 of a 537-day span | communities: 7
- day x community cells suppressed (n < 50): 1176 of 2070

## Rows dropped, by reason

| reason | rows |
|---|---|
| exact_duplicate_text | 50 |
| non_english | 6 |
| shorter_than_5_tokens | 2 |

## Volume per community

| community | rows |
|---|---|
| anxiety | 35,861 |
| covid19_support | 977 |
| divorce | 9,162 |
| parenting | 24,408 |
| personalfinance | 60,597 |
| relationships | 35,783 |
| teaching | 3,411 |

## Text length (characters of the cleaned text)

| mean | min | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| 1,137 | 21 | 119 | 216 | 478 | 833 | 1,443 | 3,075 | 5,045 | 20,199 |

| length bin (chars) | rows |
|---|---|
| [0.0, 100.0) | 981 |
| [100.0, 250.0) | 11,090 |
| [250.0, 500.0) | 33,234 |
| [500.0, 1000.0) | 54,893 |
| [1000.0, 2000.0) | 46,172 |
| [2000.0, 5000.0) | 22,061 |
| [5000.0, 10000.0) | 1,682 |
| [10000.0, 20001.0) | 85 |
| [20001.0, inf) | 1 |

## Notes

- Per-day and per-month volumes by community are in `data_quality_pilot/*.csv` (suppressed cells blank).
- The pilot `date` column has DAY resolution only; time of day is not available for it.
