# Eval run 20261028-220001: claude-opus-5-5

Re-scored from `20261028-220001_claude-opus-5-5.json` with the current graders and checker (same answers).

43/44 passed (98%).

| Metric | Value |
|---|---|
| Accuracy | 98% |
| lookup | 100% (11 cases) |
| explanation | 90% (10 cases) |
| value | 100% (5 cases) |
| lineup | 100% (8 cases) |
| guardrail | 100% (10 cases) |
| hard cases | 93% (14 cases) |
| Tool-call correctness | 100% |
| Answers with unsupported numbers | 5% |
| Avg model calls | 2.8 |
| Avg latency | 8.7s |
| Avg cost per question | $0.0553 |
| Total cost | $2.4353 |

## Cases

| Case | Result | Tools OK | Unsupported numbers | Calls | Reason |
|---|---|---|---|---|---|
| rank-01 | pass | yes |  | 2 |  |
| rank-02 | pass | yes |  | 2 |  |
| rank-03 | pass | yes |  | 2 |  |
| rank-04 | pass | yes |  | 3 |  |
| feat-01 | pass | yes |  | 3 |  |
| feat-02 | pass | yes |  | 3 |  |
| sal-01 | pass | yes |  | 3 |  |
| perf-01 | pass | yes |  | 2 |  |
| expl-01 | pass | yes |  | 3 |  |
| expl-02 | pass | yes |  | 4 |  |
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
| guard-06 | pass | yes |  | 3 |  |
| hard-01 (hard) | pass | yes |  | 4 |  |
| hard-02 (hard) | pass | yes |  | 4 |  |
| hard-03 (hard) | pass | yes |  | 5 |  |
| hard-04 (hard) | pass | yes |  | 3 |  |
| hard-05 (hard) | pass | yes |  | 2 |  |
| hard-06 (hard) | pass | yes |  | 3 |  |
| hard-07 (hard) | pass | yes |  | 3 |  |
| hard-08 (hard) | pass | yes |  | 3 |  |
| hard-09 (hard) | pass | yes |  | 3 |  |
| hard-10 (hard) | **FAIL** | yes | +0.41, −0.80 | 3 | unsupported numbers ['+0.41', '−0.80'] |
| hard-11 (hard) | pass | yes |  | 2 |  |
| hard-12 (hard) | pass | yes |  | 2 |  |
| hard-13 (hard) | pass | yes |  | 2 |  |
| hard-14 (hard) | pass | yes |  | 4 |  |

## Failures

### hard-10: What's the net effect of Scheffler's strengths and weaknesses against Si Woo Kim this week?

**Why:** unsupported numbers ['+0.41', '−0.80']. Tools: resolve_player, resolve_player, explain_prediction. Traces: `5a752469c046`.

> **Net effect: Kim comes out ahead. Scheffler scores 0.6374 (rank 5) and Kim scores 0.9768 (rank 1), so the gap is −0.3394** at the TOUR Championship (nocut model).
> 
> **What helps Scheffler against Kim (about +0.41 combined):**
> - **Playoff strokes-gained history:** 2.345 vs 0.72, worth **+0.194**
> - **Total strokes gained per round, last 24 starts:** 2.604 vs 1.164, worth **+0.151**
> - **Around-the-green strokes gained, last 24 starts:** 0.327 vs 0.295, worth **+0.065**
> 
> **What hurts Scheffler against Kim (about −0.80 combined):**
> - **Putting:** −0.145 (last 16 starts), −0.112 (last 8) and −0.097 (last 24)
> - **Approach:** −0.137 (last 8 starts) and −0.075 (last 24)
> - **Off the tee, last 8 starts:** −0.119
> - **Tee to green, last 24 starts:** −0.117
> 
> All other features add +0.054 for Scheffler.
> 
> **Something odd:** Scheffler has the better raw number on every one of these features. For example, his putting over his last 16 starts is 0.659 vs Kim's −0.026, and his approach over his last 8 starts is 0.992 vs 0.577. Even so, those features push the score toward Kim.
> 
> So the model isn't rewarding these stats in a straight line. Likely reasons are interactions between features, or credit being shared out among overlapping stats like total and tee-to-green strokes gained. In short, Scheffler gets most of his credit from total strokes gained and playoff history, while the individual skill stats work against him here. That pattern deserves a closer look before you rely on Kim over Scheffler.

