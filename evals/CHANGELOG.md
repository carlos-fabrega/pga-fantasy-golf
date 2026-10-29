# Eval changelog

One entry per change to the agent, prompt, tools, graders or cases, with before and after numbers
(PGA_AGENT_PLAN.md §9). Reports are in `evals/results/`.

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
