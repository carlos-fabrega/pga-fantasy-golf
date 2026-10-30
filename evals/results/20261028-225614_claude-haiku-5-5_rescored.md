# Eval run 20261028-225614: claude-haiku-5-5

Re-scored from `20261028-225614_claude-haiku-5-5.json` with the current graders and checker (same answers).

43/44 passed (98%).

| Metric | Value |
|---|---|
| Accuracy | 98% |
| lookup | 91% (11 cases) |
| explanation | 100% (10 cases) |
| value | 100% (5 cases) |
| lineup | 100% (8 cases) |
| guardrail | 100% (10 cases) |
| hard cases | 93% (14 cases) |
| Tool-call correctness | 100% |
| Answers with unsupported numbers | 7% |
| Avg model calls | 2.7 |
| Avg latency | 4.5s |
| Avg cost per question | $0.0014 |
| Total cost | $0.0611 |

## Cases

| Case | Result | Tools OK | Unsupported numbers | Calls | Reason |
|---|---|---|---|---|---|
| rank-01 | pass | yes |  | 2 |  |
| rank-02 | pass | yes |  | 2 |  |
| rank-03 | pass | yes |  | 2 |  |
| rank-04 | pass | yes |  | 2 |  |
| feat-01 | pass | yes |  | 3 |  |
| feat-02 | pass | yes |  | 3 |  |
| sal-01 | pass | yes |  | 3 |  |
| perf-01 | pass | yes |  | 2 |  |
| expl-01 | pass | yes |  | 3 |  |
| expl-02 | pass | yes |  | 3 |  |
| expl-03 | pass | yes |  | 3 |  |
| expl-04 | pass | yes |  | 3 |  |
| expl-05 | pass | yes |  | 3 |  |
| expl-06 | pass | yes | +0.21 | 3 |  |
| val-01 | pass | yes |  | 2 |  |
| val-02 | pass | yes |  | 2 |  |
| val-03 | pass | yes |  | 2 |  |
| val-04 | pass | yes |  | 2 |  |
| lu-01 | pass | yes |  | 2 |  |
| lu-02 | pass | yes |  | 3 |  |
| lu-03 | pass | yes |  | 3 |  |
| lu-04 | pass | yes |  | 3 |  |
| lu-05 | pass | yes |  | 2 |  |
| lu-06 | pass | yes | $898,000 | 3 |  |
| guard-01 | pass | yes |  | 3 |  |
| guard-02 | pass | yes |  | 2 |  |
| guard-03 | pass | yes |  | 2 |  |
| guard-04 | pass | yes |  | 4 |  |
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
| hard-09 (hard) | **FAIL** | yes | 0.3394 | 3 | unsupported numbers ['0.3394'] |
| hard-10 (hard) | pass | yes |  | 3 |  |
| hard-11 (hard) | pass | yes |  | 1 |  |
| hard-12 (hard) | pass | yes |  | 2 |  |
| hard-13 (hard) | pass | yes |  | 2 |  |
| hard-14 (hard) | pass | yes |  | 4 |  |

## Failures

### hard-09: By how much does the model's score for Si Woo Kim beat Scheffler's this week?

**Why:** unsupported numbers ['0.3394']. Tools: resolve_player, resolve_player, get_rankings. Traces: `5179cd8709a3`.

> Si Woo Kim's model score is **0.9768** and Scottie Scheffler's is **0.6374**, so Kim leads by **0.3394** points at this week's TOUR Championship (East Lake).
> 
> Kim ranks 1st and Scheffler 5th of the 30-player field. These are relative ranking scores from the no-cut model, so the gap shows ordering strength within this event, not a win probability.

