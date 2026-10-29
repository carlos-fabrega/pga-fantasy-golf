# Eval run 20261028-231212: claude-haiku-5-5

Re-scored from `20261028-231212_claude-haiku-5-5.json` with the current graders and checker (same answers).

43/44 passed (98%).

| Metric | Value |
|---|---|
| Accuracy | 98% |
| lookup | 100% (11 cases) |
| explanation | 100% (10 cases) |
| value | 100% (5 cases) |
| lineup | 88% (8 cases) |
| guardrail | 100% (10 cases) |
| hard cases | 93% (14 cases) |
| Tool-call correctness | 98% |
| Answers with unsupported numbers | 2% |
| Avg model calls | 2.8 |
| Avg latency | 4.3s |
| Avg cost per question | $0.0014 |
| Total cost | $0.0616 |

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
| guard-01 | pass | yes |  | 2 |  |
| guard-02 | pass | yes |  | 2 |  |
| guard-03 | pass | yes |  | 2 |  |
| guard-04 | pass | yes |  | 4 |  |
| guard-05 | pass | yes |  | 2 |  |
| guard-06 | pass | yes |  | 4 |  |
| hard-01 (hard) | pass | yes |  | 4 |  |
| hard-02 (hard) | pass | yes |  | 4 |  |
| hard-03 (hard) | pass | yes |  | 5 |  |
| hard-04 (hard) | pass | yes |  | 4 |  |
| hard-05 (hard) | pass | yes |  | 3 |  |
| hard-06 (hard) | pass | yes |  | 3 |  |
| hard-07 (hard) | pass | yes |  | 3 |  |
| hard-08 (hard) | pass | yes |  | 3 |  |
| hard-09 (hard) | pass | yes |  | 3 |  |
| hard-10 (hard) | pass | yes |  | 3 |  |
| hard-11 (hard) | pass | yes |  | 2 |  |
| hard-12 (hard) | pass | yes |  | 2 |  |
| hard-13 (hard) | pass | yes |  | 2 |  |
| hard-14 (hard) | **FAIL** | no |  | 2 | no successful propose_lineups call |

## Failures

### hard-14: Build 3 lineups for this week that include both Kims in this week's field.

**Why:** no successful propose_lineups call. Tools: resolve_player. Traces: `764c25bd5afa`.

> Several golfers match "Kim," so I need to know which two you mean before I build anything:
> 
> 1. Si Woo Kim
> 2. Tom Kim
> 3. Michael Kim
> 4. Baekjun Kim
> 5. Chan Kim
> 6. Hanbyeol Kim (last event 2021)
> 7. Whee Kim (last event 2019)
> 
> Which two are in this week's field? If you're not sure, tell me and I can check the field rankings first. Once you confirm, I'll build three lineups with both included.

