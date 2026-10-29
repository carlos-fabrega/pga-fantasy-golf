# Eval run 20261028-214131: claude-haiku-5-5

Re-scored from `20261028-214131_claude-haiku-5-5.json` with the current graders and checker (same answers).

42/44 passed (95%).

| Metric | Value |
|---|---|
| Accuracy | 95% |
| lookup | 91% (11 cases) |
| explanation | 90% (10 cases) |
| value | 100% (5 cases) |
| lineup | 100% (8 cases) |
| guardrail | 100% (10 cases) |
| hard cases | 86% (14 cases) |
| Tool-call correctness | 100% |
| Answers with unsupported numbers | 5% |
| Avg model calls | 2.7 |
| Avg latency | 4.2s |
| Avg cost per question | $0.0014 |
| Total cost | $0.0607 |

## Cases

| Case | Result | Tools OK | Unsupported numbers | Calls | Reason |
|---|---|---|---|---|---|
| rank-01 | pass | yes |  | 2 |  |
| rank-02 | pass | yes |  | 2 |  |
| rank-03 | pass | yes |  | 2 |  |
| rank-04 | pass | yes |  | 2 |  |
| feat-01 | pass | yes |  | 3 |  |
| feat-02 | pass | yes |  | 4 |  |
| sal-01 | pass | yes |  | 3 |  |
| perf-01 | pass | yes |  | 2 |  |
| expl-01 | pass | yes |  | 3 |  |
| expl-02 | pass | yes |  | 3 |  |
| expl-03 | pass | yes |  | 3 |  |
| expl-04 | pass | yes |  | 3 |  |
| expl-05 | pass | yes |  | 3 |  |
| expl-06 | pass | yes |  | 3 |  |
| val-01 | pass | yes |  | 2 |  |
| val-02 | pass | yes |  | 2 |  |
| val-03 | pass | yes |  | 2 |  |
| val-04 | pass | yes |  | 2 |  |
| lu-01 | pass | yes |  | 2 |  |
| lu-02 | pass | yes |  | 3 |  |
| lu-03 | pass | yes |  | 3 |  |
| lu-04 | pass | yes |  | 3 |  |
| lu-05 | pass | yes |  | 2 |  |
| lu-06 | pass | yes | $898,000, $248,000 | 3 |  |
| guard-01 | pass | yes |  | 2 |  |
| guard-02 | pass | yes |  | 2 |  |
| guard-03 | pass | yes |  | 2 |  |
| guard-04 | pass | yes |  | 3 |  |
| guard-05 | pass | yes |  | 2 |  |
| guard-06 | pass | yes |  | 3 |  |
| hard-01 (hard) | pass | yes |  | 4 |  |
| hard-02 (hard) | pass | yes |  | 4 |  |
| hard-03 (hard) | pass | yes |  | 5 |  |
| hard-04 (hard) | pass | yes |  | 3 |  |
| hard-05 (hard) | pass | yes |  | 3 |  |
| hard-06 (hard) | pass | yes |  | 3 |  |
| hard-07 (hard) | pass | yes |  | 3 |  |
| hard-08 (hard) | pass | yes |  | 3 |  |
| hard-09 (hard) | **FAIL** | yes |  | 3 | missing [-0.3394] |
| hard-10 (hard) | **FAIL** | yes | -0.73 | 3 | unsupported numbers ['-0.73'] |
| hard-11 (hard) | pass | yes |  | 1 |  |
| hard-12 (hard) | pass | yes |  | 2 |  |
| hard-13 (hard) | pass | yes |  | 2 |  |
| hard-14 (hard) | pass | yes |  | 4 |  |

## Failures

### hard-09: By how much does the model's score for Si Woo Kim beat Scheffler's this week?

**Why:** missing [-0.3394]. Tools: resolve_player, resolve_player, get_rankings. Traces: `1ad83e6188a5`.

> Si Woo Kim outscores Scottie Scheffler by about **0.34** points this week (TOUR Championship, East Lake).
> 
> - Si Woo Kim: 0.9768 (rank 1)
> - Scottie Scheffler: 0.6374 (rank 5)
> 
> These are relative ranking scores within this event, not probabilities or prize estimates.

### hard-10: What's the net effect of Scheffler's strengths and weaknesses against Si Woo Kim this week?

**Why:** unsupported numbers ['-0.73']. Tools: resolve_player, resolve_player, explain_prediction. Traces: `cd5c6c6ba8a9`.

> **Net effect: Si Woo Kim comes out ahead.** Kim has a model score of 0.9768 (rank 1), and Scheffler has 0.6374 (rank 5), a gap of 0.3394 in Kim's favor at the TOUR Championship.
> 
> **Scheffler's edges (about +0.34 combined):**
> - Playoff history (2.345 vs. Kim's 0.720): +0.194
> - 24-start total strokes gained per round (2.604 vs. 1.164): +0.151
> 
> **Scheffler's deficits (about -0.73 combined on the listed features):**
> - 16-start putting (0.659 vs. -0.026): -0.145. The model pulls Scheffler down here even though his putting numbers are higher than Kim's.
> - 8-start approach (0.992 vs. 0.577): -0.137
> - 8-start off-the-tee (0.624 vs. 0.367): -0.119
> - 24-start tee-to-green (1.944 vs. 1.190): -0.117
> - 8-start putting (0.710 vs. 0.178): -0.112
> - 24-start putting (0.659 vs. -0.026): -0.097
> 
> Other features add +0.043, so the net gap is about -0.34. The model's preference for Kim comes mainly from the approach, off-the-tee and putting features, which outweigh Scheffler's playoff and long-run total-strokes-gained advantages. The putting result runs against what the raw numbers suggest, so treat that part with caution.

