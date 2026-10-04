# START HERE: what is done and what you do next

Written 2026-10-05 for Madhava. Nothing is running right now. Everything you need to continue is in this file.

## In one paragraph

The project reads Reddit posts and measures stress-related language: is a post stressed, which emotions it shows, which stressor it names.
The models are trained and packaged. The pilot data (170,199 posts from 7 subreddits) is cleaned and stored. The big scoring job over the
pilot is **more than half done: 252 of 477 days**. After it finishes, the remaining work is turning the scores into trends (Steps 10 to 13)
and writing the final report.

## What is finished (you do not need to touch these)

- Steps 1 to 6: data cleaning, the stress model, the emotion model, the stressor tagger, external checks.
- Step 7: one package that runs all three models together (`code/stress_signals/inference.py`).
- Step 8: the data store for the pilot (Parquet files under `data/processed/lake/pilot/`).
- Step 9: the Spark scoring job. It works and was checked against the plain pipeline (differences below 0.000003).
- Step 10: the analytics code is written and its tests pass. Its first tables were made from only a small slice of the data.
- Everything is pushed to GitHub on the branch `steps-5-10-inference-spark-analytics` (nothing is merged into `main`).

## Your to-do list, in order

### 1. Finish the scoring job (about 1 hour)

Close other heavy programs first (browsers with many tabs, games). This laptop has little free memory and two big jobs at once make it crash.
Plug the laptop in. Then open PowerShell and run:

```
cd "D:\Visual_studio\TA-BDA Project\Psychological-Stress-Analysis\code"
..\..\.venv\Scripts\python.exe -m stress_signals.spark_jobs run --corpus pilot
```

- It continues from day 253. You can stop it any time (close the window or end the python and java processes). Nothing is lost; run the same command again to continue.
- It is finished when the last lines show `"dates_run"` and an empty `"errors": {}`, and the time stops printing. To double check, run it once more: it should say `"dates_run": 0`.

### 2. Decide how topic modelling gets its text (one decision)

Topic modelling (Step 10, cells 10.3 and 10.4) needs the post text. The scoring job deliberately does not keep text, so those two cells currently fail with
`FileNotFoundError`. Pick one:

- **Option A:** turn on `keep_text_for_topics` in `code/config.yaml` and score everything again (about 1.7 hours more).
- **Option B (recommended):** I write a small script that copies the text from the data store, no re-scoring. Ask me: "write the text exporter".

Either way the text folder stays on your laptop only (it is git-ignored). Never put it in the database, the dashboard or GitHub.

### 3. Redo the analytics on the full data

In VS Code open `code/notebook.ipynb`, choose the kernel `.venv` (top right), and run cells **10.1 and 10.2** only. The tables currently in `data/gold` come from
just 14,339 posts, so they should be replaced. (Do not use Run All for this.)

### 4. Fit the topic model

Install three packages once, then run cell 10.3, 10.4 and 10.5 in the notebook:

```
"D:\Visual_studio\TA-BDA Project\.venv\Scripts\python.exe" -m pip install bertopic umap-learn hdbscan
```

Afterwards pin the installed versions in `requirements.txt` (tell me and I will do it).

### 5. Build the remaining steps

Paste the prompts for **Step 11** (baseline and change detection), **Step 13** (the dashboard) and **Step 14** (final report) to me one at a time.
Step 12 (PostgreSQL) is optional and can be skipped: the dashboard can read the Parquet tables directly.

### 6. Update the LaTeX report

The report (`Downloads\main (4).tex`) still says "195 of 477 dates". Update the numbers after step 1 (status table, the Step 9 section, Results, limitations, conclusion).
Tell me "update the report" when the run has finished and I will do it.

### 7. Much later: the big Reddit download

Only after the dashboard works on the pilot. Not needed now.

## Questions that are waiting for you

- Open a pull request on GitHub to merge the branch into `main`? Link: https://github.com/daniellowho/Psychological-Stress-Analysis/pull/new/steps-5-10-inference-spark-analytics
- Delete `.venv-1` in the folder above this one? It is an unused old environment.
- Move the Hugging Face model cache off `C:`? It lives in `C:\Users\medam\.cache\huggingface` and the project still uses it. Leave it unless C: is full.

## If something goes wrong

| What you see | What to do |
|---|---|
| `memory allocation failed`, `inference child exited`, a crash | Close other programs, run the command again. It resumes by itself. |
| Command not found / path error | Put the whole path in quotes. The folder name has a space ("TA-BDA Project"). |
| A test fails | Run `..\..\.venv\Scripts\python.exe -m pytest tests -q` from `code/` and send me the output. All 252 passed at the last full run. |
| Notebook asks to train a model | Do not. `ALLOW_TRAINING = False` in cell 0.1 stops training; leave it that way. |
| You want to edit a file inside `trained_models\...\v20261004\` | Don't. Those files are checked with hashes and the loader rejects a changed bundle. |

## Words used in this project

- **Pilot:** the 170,199 posts already downloaded (a trial of the whole pipeline). The big 2019 to 2022 Reddit corpus comes later.
- **Bundle:** the three trained models plus the exact cleaning code, packed together and fingerprinted.
- **Stage 1 / stage 2:** stage 1 scores every post for stress. Stage 2 (emotions, stressors) runs only on stressed posts plus a 15% random sample of the others, to save time. The column `sample_weight` corrects for that.
- **Enriched data:** the scores per post (no text), in `data/processed/enriched/pilot/`.
- **Provisional (stressor tagger):** its test labels were made by a machine, so treat its results as a first look, not proof.

## Where things are

- Code: `code/stress_signals/` and the notebook `code/notebook.ipynb`; settings: `code/config.yaml`.
- Scores: `data/processed/enriched/pilot/`. Cleaned posts: `data/processed/lake/pilot/`. Aggregate tables: `data/gold/`.
- Reports about the data: `data/outputs/reports/`. Models and the automatic run log: `trained_models/` (`HANDOFF.md` lists every step that ran).
- Python environment: `D:\Visual_studio\TA-BDA Project\.venv` (Python 3.11).
