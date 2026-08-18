# Live grounding calibration

DB: `/home/sfx/.cache/living-memory-harness/snapshot.sqlite3`

- 72023 result/trace pairs over 9035 consumed events (9069 labeled of 54757 usable, 8623 corpus documents)
- Reference (whole-corpus IDF at 0.25): 11073 positives (0.1537 of pairs)
- Pearson correlation live vs corpus containment: 0.9844

## Threshold sweep (live per-event IDF vs the corpus-IDF reference)

| threshold | positives | pos.rate | precision | recall | F1 | agreement |
|---|---|---|---|---|---|---|
| 0.050 | 54376 | 0.7550 | 0.2036 | 1.0000 | 0.3384 | 0.3988 |
| 0.075 | 43545 | 0.6046 | 0.2543 | 1.0000 | 0.4055 | 0.5491 |
| 0.100 | 34484 | 0.4788 | 0.3211 | 1.0000 | 0.4861 | 0.6750 |
| 0.125 | 27278 | 0.3787 | 0.4059 | 0.9999 | 0.5774 | 0.7750 |
| 0.150 | 21662 | 0.3008 | 0.5103 | 0.9984 | 0.6754 | 0.8525 |
| 0.175 | 17159 | 0.2382 | 0.6395 | 0.9911 | 0.7774 | 0.9127 |
| 0.200 | 13685 | 0.1900 | 0.7837 | 0.9686 | 0.8664 | 0.9541 |
| 0.225 | 10849 | 0.1506 | 0.9149 | 0.8964 | 0.9056 | 0.9713 |
| 0.250 | 8698 | 0.1208 | 0.9776 | 0.7679 | 0.8601 | 0.9616 |
| 0.275 | 6976 | 0.0969 | 0.9923 | 0.6251 | 0.7670 | 0.9416 |
| 0.300 | 5634 | 0.0782 | 0.9996 | 0.5086 | 0.6742 | 0.9244 |
| 0.325 | 4511 | 0.0626 | 0.9998 | 0.4073 | 0.5788 | 0.9089 |
| 0.350 | 3657 | 0.0508 | 1.0000 | 0.3303 | 0.4965 | 0.8970 |
| 0.375 | 2987 | 0.0415 | 1.0000 | 0.2698 | 0.4249 | 0.8877 |
| 0.400 | 2477 | 0.0344 | 1.0000 | 0.2237 | 0.3656 | 0.8806 |

F1 optimum: 0.225 (F1 0.9056, agreement 0.9713).
Adopted: **0.25**.

Held at the replay default. The live view at 0.25 is the precision-favouring point (precision 0.978, recall 0.768); F1 peaks at 0.225 (precision 0.915, recall 0.896). For a credit rule a false positive is the defect being fixed — noise re-entering the learned signal — while a false negative only withholds a signal that recurs on the next consumption, so the asymmetry favours precision and no recalibration was adopted.

Both columns are proxies for "the agent used this result"; the sweep measures how far the live view can drift from the offline label, not accuracy against truth.
