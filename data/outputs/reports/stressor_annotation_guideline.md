# Stressor annotation guideline (gold set v1)

**Task.** For each text, mark every **cause of pressure that the text names**. You are labelling *language*, not the
writer: never guess at a diagnosis, a person's condition, or anything the text does not say.

**File.** `data/outputs/gold/stressor_gold_v1_annotator1.csv` (all ~400 rows). Open it in Excel, type `1` in
every column that applies, leave the others blank, and save it as CSV (UTF-8). Do not edit `gold_id` or `text`, and do
not re-sort the rows. Every row needs at least one `1`.

**Categories** (full definitions, include/exclude rules: `code/taxonomy/stressors_v1.yaml`):

| column | mark 1 when the text names... |
|---|---|
| workplace_pressure | paid work or job seeking: workload, boss/co-workers, shifts, job loss, unemployment |
| academic_workload | school/university: exams, grades, assignments, teachers, failing |
| financial_concerns | money: rent, bills, debt, income, affording food/housing, eviction, homelessness as a money problem |
| relationship_issues | family, partner, friends, break-up, social conflict, abuse by someone close, isolation as a *situation* |
| health_concerns | illness, injury, pain, fatigue, sleep problems, medical tests/treatment (own or of someone close) |
| uncertainty | not knowing what will happen: waiting for an outcome, an unclear future |
| overload | too much at once: many tasks/decisions, no time to rest, planning and logistics piling up |
| societal_problems | public events/conditions: pandemic, politics, war, discrimination, crime, news, economy |
| other_unclear | a cause is named but fits none of the above, or is vague ("everything is going wrong", car trouble, a court case) |
| none_unclear | **no cause is named at all**: only feelings/symptoms, or the text is unintelligible. Never combine with another column |

**Rules.**
1. **Multi-label.** Mark all causes that are stated. "I lost my job and can't pay rent" -> workplace_pressure + financial_concerns.
2. **Cause, not feeling.** "I feel anxious and can't breathe" names no cause -> none_unclear. Emotions and symptoms are
   never a stressor by themselves (an anxiety *disorder* the writer is dealing with as a health problem -> health_concerns).
3. **Stated, not inferred.** Mark what the text says. Do not use the subreddit or your guess about the writer's life.
   Rhetorical or past mentions count only if presented as a current source of pressure.
4. **Abuse/violence** by a partner or relative -> relationship_issues (+ health_concerns if injuries/health are the point).
5. **Pandemic.** As a public event (lockdowns, news) -> societal_problems. Own COVID illness -> health_concerns. Job lost
   because of the pandemic -> workplace_pressure (+ societal_problems only if the public event itself is discussed).
6. **Homelessness/housing** -> financial_concerns when framed as money/housing; + relationship_issues if a family
   member threw the writer out.
7. **Uncertainty vs a domain.** "Waiting for my test results" -> health_concerns + uncertainty. Mark uncertainty only
   when not-knowing is itself presented as the pressure.
8. **Overload vs a domain.** Load from one domain only -> that domain. Overload only for many demands across life areas.
9. **Unsure?** Prefer the more specific category; use `notes` for real doubts. Use other_unclear only when a cause is
   present; none_unclear only when none is.

**Edge cases.** Sarcasm: label the literal cause. Quoted or hypothetical situations ("what if I lose my job"):
uncertainty (+ the domain). A stressor affecting someone else that the writer worries about (a friend's illness) ->
the same category (health_concerns). Texts that are mostly advice or questions to others with no own stressor ->
none_unclear.

**Agreement.** Single annotator, no inter-annotator agreement: there is no second annotator, so label reliability
is not measured (and is never simulated). This is recorded in the gold-set report, the model card and the final report.
Label in one pass where possible and use `notes` for doubtful rows.

**Privacy.** The sheets contain Reddit text: keep them on this laptop (under `data/`, never committed or shared). Do
not copy texts into notes, chats or reports.
