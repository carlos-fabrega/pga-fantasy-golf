# Eval changelog

One entry per change to the agent, prompt, tools, graders or cases, with before and after numbers
(PGA_AGENT_PLAN.md §9). Reports are in `evals/results/`.

## 2026-11-13: Backtest extended to 2018–2026; lineups optimise expected prize share

**Change.** The agent DB now holds the backfilled events, so `model_performance` covers 2018–2026 and
"this week" means the latest live weekly event rather than the latest scored one. `propose_lineups`
maximises expected prize share and returns `total_expected_prize_pct`. guard-06 asks about the 2027
Masters, since the 2026 Masters now has backtest scores. Cases regenerated.

**Result.** Not run.

## 2026-11-13: Cases regenerated after the 2023+ backfill and retrain

**Change.** The models were retrained on backfilled 2023–2026 data and the seven live weeks re-scored, so the
rankings behind the cases changed (Scheffler is now #1 at the TOUR Championship, not below Si Woo Kim). Five
cases had premises tied to the old rankings and now take them from the data:
- expl-01, expl-02, hard-09: the question names the two players in the order the model ranks them.
- guard-04: every live week's model now reproduces its scores, so the unexplainable event is a backtest one
  (2019 U.S. Open, `unsupported`) instead of a base-model week (`model_changed`).
- hard-04: the wrong "#1" is whoever is ranked 5th.
- hard-08: the pair and putting window are searched for in this week's top 12 (better putting number, negative
  contribution gap); the generator stops if none exists.

`model_performance` now grades the live weeks on out-of-fold scores, which changes perf-01 and hard-13.

**Result.** Not run: no API calls were made. Earlier results were measured on the old cases and rankings, so they
aren't comparable with runs on this set; re-run the baseline before measuring the next change.

## 2026-10-28: Iteration 1, "don't do arithmetic" prompt rule

**Change.** One system prompt rule: "Never add, subtract or average numbers yourself; report the figure a tool
returns. For the gap between two players, or the net effect of all their features, use explain_prediction's
score_gap. List contributions one by one rather than totalling them."

**Why.** It targets the only failure both models shared in the baseline: summing drivers for a "net effect"
(hard-10). It also targets Haiku subtracting two ranking scores instead of reporting the exact gap (hard-09).
`score_gap` already *is* the net effect of every feature, so the right answer was always in a tool output.

**Result.** Haiku 5.5, means over 3 runs each:

| Metric | Before | After |
|---|---|---|
| Accuracy | 95.5% | **97.7%** |
| Hard cases | 85.7% | **92.9%** |
| Answers with unsupported numbers | 6.8% | **0.8%** |
| Avg model calls | 2.71 | 2.74 |
| Avg latency | 4.4s | 4.3s |
| Avg cost per question | $0.0014 | $0.0014 |

**Cases whose pass count changed:**
- hard-09: 0/3 → 3/3
- hard-10: 2/3 → 3/3
- hard-14: 1/3 → 0/3

**Reading the answers:**
- hard-10 now gives `score_gap` (−0.3394) as the net effect and lists each contribution.
- lu-06 still explains the failure ("$1,898,000 combined, above the $1,000,000 budget") but mostly stops working out the overage itself. The one remaining flag across all 132 answers is that "$898,000", in one run.

**hard-14 isn't a regression from this rule.** It already failed 2/3 before. In every failing run Haiku asks
"which two Kims?" instead of checking the field, following the prompt's ask-when-ambiguous rule. That's a candidate for iteration 2.

**Grader fix made along the way.** guard-06 failed one "after" run on wording: "has no 2026 Masters
rankings … returned no matching event" wasn't in the unavailable-data cues. I added the cues, then re-scored *all six*
runs with `--rescore`, so before and after use the same grader. The table above is the re-scored one; the fix
changed only that one guard-06 answer.

Runs (re-scored reports are `*_rescored`): before `20261028-225614`, `225933`, `230252`; after `230902`, `231212`,
`231524`. Compare with `python -m evals.compare --before … --after …`.

## 2026-10-28: Measurement fixes, re-scored on the same answers

No agent change. These fixes change only how answers are graded, and both baselines were re-scored from their traces
(`run_evals --rescore`, no API calls), so the numbers move only because of the grading.

- **Numeric checker:**
  - Our own system prompt and tool descriptions now count as sources (the $1,650,000 default budget, "2018–2022").
  - "Top N" framing and Markdown table row numbers are skipped.
- **`drivers` grader:** "gains 1.4 strokes per round over his last 24 starts", with no other stat family named, counts as total strokes gained.

| | Haiku 5.5 before → after | Opus 5.5 before → after |
|---|---|---|
| Accuracy | 42/44 → 42/44 | 42/44 → **43/44** (hard-07 now passes) |
| Hard cases | 12/14 → 12/14 | 12/14 → 13/14 |
| Answers with unsupported numbers | 5% → 5% | 20% → **5%** |
| Report | `20261028-214131_claude-haiku-5-5_rescored` | `20261028-220001_claude-opus-5-5_rescored` |

**What's still flagged is genuine, for both models:**
- **hard-10:** sums the drivers when asked for a "net effect".
- **lu-06:** works out "$898,000 over" the cap itself.

**Remaining failures:**
- **hard-10, both models:** that summing.
- **hard-09, Haiku:** subtracts two ranking scores instead of reporting `explain_prediction`'s gap.

**Trade-off:** the tool descriptions contain small numbers ("1–5", "6 players"), so small integers now always pass. Opus's correct but derived "only rank 5 is missing" isn't flagged any more. The checker never tested small integers well, since nearly every one matches some rank, so it remains a check on precise numbers.

## 2026-10-28: Baseline, 44 cases (30 standard + 14 hard)

| | Haiku 5.5 | Opus 5.5 |
|---|---|---|
| Report | `20261028-214131_claude-haiku-5-5` | `20261028-220001_claude-opus-5-5` |
| Accuracy | 42/44 (95%) | 42/44 (95%) |
| Hard cases | 12/14 | 12/14 |
| Explanation | 9/10 | 8/10 |
| Lookup | 10/11 | 11/11 |
| Tool-call correctness | 100% | 100% |
| Answers with unsupported numbers | 5% (2) | 20% (9) |
| Avg model calls | 2.7 | 2.8 |
| Avg latency | 4.2s | 8.7s |
| Avg cost per question | $0.0014 | $0.0553 |
| Run cost | $0.06 | $2.44 |

**Failures.**
- **Both models fail hard-10:** asked for the "net effect", each sums the drivers itself (Haiku "−0.73", Opus "+0.41"/"−0.80").
- **Haiku fails hard-09:** it subtracts two ranking scores ("about 0.34"). Opus calls `explain_prediction` and reports the exact 0.3394 gap.
- **Opus "fails" hard-07, a grader miss:** "gains 1.402 strokes per round over his last 24 and last 16 starts" is total strokes gained, without the words "strokes gained".

**Unsupported numbers, read by hand.** Opus's 9 flagged answers are mostly framing, not data claims:
- "the rest of the top 10" / "in the top 5" (3);
- Markdown table row numbers (1);
- "2018–2022" and the "$1,650,000" default budget, which come from tool *descriptions*, not outputs (2);
- "only rank 5 is missing", a correct inference (2).

Genuine model-computed numbers: about 2 per model, hard-10 for both plus Haiku's hard-09 or Opus's "$898,000 over" in lu-06. Opus writes more framing text, so the checker's noise hits it harder.

**What the baseline says.**
- Opus isn't more accurate on this set. Both models make the same core mistake (doing arithmetic when asked for a total) and differ by one case each way.
- Opus costs about 40× more and takes 2× longer for the same accuracy.
- Single runs are noisy: one case flipped between two identical Haiku runs.

**Known measurement issues, not fixed here so the baseline stays as measured:**
- The numeric checker flags "top N" framing and table row numbers, and doesn't treat tool descriptions as a source.
- The `drivers` grader misses "strokes per round" as a way of saying total strokes gained.
