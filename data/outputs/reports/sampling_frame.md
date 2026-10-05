# Sampling frame and selection bias

*Generated from config.yaml by `stress_signals.ingestion.write_sampling_frame`. Counts come from the ingestion manifests only.*

**Aggregate, population-level language signals only. No diagnosis. No individual-level analysis.**

## What is in the frame

| corpus | source | communities | period |
|---|---|---|---|
| pilot | Zenodo Reddit Mental Health Dataset, record 3941387 (files you downloaded) | r/anxiety, r/divorce, r/parenting, r/personalfinance, r/relationships, r/teaching, r/COVID19_support | `pre` 2018-11-01 to 2019-11-01; `post` 2020-01-01 to 2020-04-20 (observed on disk, not the record's description) |
| main | Reddit SUBMISSIONS (posts) from per-subreddit dumps / Arctic Shift (files you downloaded) | r/anxiety, r/personalfinance, r/relationships, r/parenting, r/divorce, r/teaching, r/legaladvice, r/unemployment | 2019-01 to 2022-12 |

Ingestion status:

- `pilot`: 170,257 rows read, 170,199 kept (manifest step8_ingestion_pilot.json, 2026-10-04T17:18:57+00:00)
- `main`: not ingested yet (no manifest)

Why these communities (user decisions, Step 1): the pilot communities are kept in the main corpus so the pilot is comparable, plus r/legaladvice
and r/unemployment as communities where stressors such as legal and job problems are discussed. Other everyday-stress communities could
have been chosen; this is a convenience-and-comparability sample, not a probability sample of Reddit or of any population.

## What is excluded, and why

- **Suicide-related communities (r/suicidewatch)**: excluded by design. They hold posts from people who may be in acute crisis.
  Even aggregate monitoring of them invites misuse (targeting, outreach to individuals, re-identification of small groups), the
  project must not produce individual-level output, and small communities make low-count cells hard to protect. The project is a
  language-signal study of everyday pressure, not a crisis-detection system. **Consequence:** the most severe end of distress is
  under-represented; results must not be read as covering it.
- Posts removed or deleted (`[removed]`, `[deleted]`, moderator-removed), empty selftext (link-only posts), bot / moderator text.
- Non-English text (language filter on, `ingestion.language_filter`); texts under 5 tokens.
- Exact duplicate texts after cleaning (the first seen is kept). Near-duplicates are not removed.
- Comments (main corpus is submissions only), images, links, and every author / user field (dropped at ingestion).
- Outside DATE_RANGE (main) and communities not listed above.

## Selection bias and what it means for the results

- **Who posts:** only people who choose to post publicly in these communities. Reddit users are not a random sample of any population
  (demographics not measured here; UNVERIFIED), and each community has its own norms and rules.
- **Survivorship and moderation:** removed or deleted posts are missing, so what was removed (often the most acute or the rule-breaking)
  is not seen. Dump timing decides which posts had been removed when the archive was taken (UNVERIFIED for the files you hold).
- **Community mix drives totals:** pooled numbers are dominated by the largest communities; compare within a community over time.
- **Pilot windows are not contiguous** (2019-11-01 to 2020-01-01 is missing) and the post window
  starts at the onset of the pandemic, so a pre/post difference mixes season, pandemic and community change. The pilot `date` has day resolution only.
- **Text-only signal:** language is not a measurement of stress or mental health of anyone. Findings are stated as "stress-related language in the
  sampled communities shifted", never as the population or any person becoming more stressed.
- **Machine labelling:** models were validated partly against machine-made labels (see the model cards); that limits what the numbers mean.

## Before using real data

Check Reddit's terms of use and API/data policy, the dataset licence (Zenodo record 3941387; the Arctic Shift terms), and your
institution's ethics requirements. This code reads local files only and never scrapes.
