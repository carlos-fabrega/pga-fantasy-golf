# Eval run 20261028-225933: claude-haiku-5-5

Re-scored from `20261028-225933_claude-haiku-5-5.json` with the current graders and checker (same answers).

42/44 passed (95%).

| Metric | Value |
|---|---|
| Accuracy | 95% |
| lookup | 91% (11 cases) |
| explanation | 100% (10 cases) |
| value | 100% (5 cases) |
| lineup | 88% (8 cases) |
| guardrail | 100% (10 cases) |
| hard cases | 86% (14 cases) |
| Tool-call correctness | 98% |
| Answers with unsupported numbers | 5% |
| Avg model calls | 2.7 |
| Avg latency | 4.5s |
| Avg cost per question | $0.0014 |
| Total cost | $0.0605 |

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
| expl-06 | pass | yes |  | 3 |  |
| val-01 | pass | yes |  | 2 |  |
| val-02 | pass | yes |  | 2 |  |
| val-03 | pass | yes |  | 2 |  |
| val-04 | pass | yes |  | 3 |  |
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
| hard-03 (hard) | pass | yes |  | 4 |  |
| hard-04 (hard) | pass | yes |  | 3 |  |
| hard-05 (hard) | pass | yes |  | 2 |  |
| hard-06 (hard) | pass | yes |  | 3 |  |
| hard-07 (hard) | pass | yes |  | 3 |  |
| hard-08 (hard) | pass | yes |  | 3 |  |
| hard-09 (hard) | **FAIL** | yes | 0.3394 | 3 | unsupported numbers ['0.3394'] |
| hard-10 (hard) | pass | yes |  | 3 |  |
| hard-11 (hard) | pass | yes |  | 3 |  |
| hard-12 (hard) | pass | yes |  | 2 |  |
| hard-13 (hard) | pass | yes |  | 3 |  |
| hard-14 (hard) | **FAIL** | no |  | 2 | no successful propose_lineups call |

## Failures

### hard-09: By how much does the model's score for Si Woo Kim beat Scheffler's this week?

**Why:** unsupported numbers ['0.3394']. Tools: resolve_player, resolve_player, get_rankings. Traces: `986ca323d5fe`.

> Si Woo Kim's score is 0.9768 and Scheffler's is 0.6374, so Kim is ahead by **0.3394** points.
> 
> This is the TOUR Championship at East Lake (nocut model, 30-player field). Kim ranks 1st and Scheffler 5th. The gap is in raw ranking-score units, so it shows relative order, not a probability or prize difference.

### hard-14: Build 3 lineups for this week that include both Kims in this week's field.

**Why:** no successful propose_lineups call. Tools: resolve_player. Traces: `641b7bfd6638`.

> "Kim" matches several golfers, so I need to know which two you mean. The candidates are:
> 
> 1. Si Woo Kim
> 2. Tom Kim
> 3. Michael Kim
> 4. Baekjun Kim
> 5. Chan Kim
> 6. Hanbyeol Kim
> 7. Whee Kim
> 
> Which two should I include? Once you confirm, I'll check that they're in this week's field and build the three lineups.

