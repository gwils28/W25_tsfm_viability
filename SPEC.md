# SPEC: W25, zero-shot TSFM vs refitted specialist on hourly load

Output of the interview of 2026-08-02 and of the blind-spot pass. The plan is [`plan_w25.md`](plan_w25.md);
the numbers that decide the verdict are fixed in [`configs/protocol.yaml`](configs/protocol.yaml), committed
before any forecast was computed.

## Decisions from the interview

| Question | Decision |
|---|---|
| Main series | France, `FR_load_actual_entsoe_transparency`, OPSD time series package version 2020-10-06 (sha256 pinned) |
| Origins | Extended protocol: one origin per day over a full test year (365), plus a COVID stress window. The brief's "6 origins" run is kept as a minimal run, and its fragility is measured, not assumed |
| Challengers | Chronos-Bolt small and base (the brief), Chronos-2 as a newer-generation check |
| Block 3 | Structured reasoning as the brief asks, plus an empirical prototype of the router on the other OPSD load series |
| Deliverable | Tested package (`src/w25`), CLI, block READMEs, a Quarto report with the figures |
| Language | English (code, READMEs, report); the plan stays in French |
| Git | Commits by the agent, backdated over 2-16 August 2026, pushed to `gwils28/W25_tsfm_viability` (public) |

## Blind spots found before writing code

1. **Chronos-Bolt cannot emit a 90% interval.** It is trained on the quantiles 0.1 to 0.9 only. Asking it for
   0.05 and 0.95 returns 0.1 and 0.9 (`ChronosBoltPipeline.predict_quantiles` pads the grid with its own
   extreme quantiles, with a log warning). A "90%" interval from Bolt is an 80% interval by construction:
   its coverage is expected near 0.80 even if the model is perfectly calibrated. We report it as requested
   (the trap) and its native 80% coverage (the fair reading). Chronos-2 is trained on 0.01 to 0.99: no trap.
2. **The French series has a corrupted value**: 158,000 MW on 2020-07-07 17:00 UTC, while the national record
   is about 102 GW. Plus 9 gaps of 1 to 13 hours. A physical range check is needed before any metric.
3. **SeasonalNaive with season 24 forecasts Monday with Sunday.** The weekly naive (season 168) is added as a
   second naive. (Measured afterwards, see `IMPLEMENTATION_NOTES.md`: neither naive dominates; the daily one
   wins Tuesday to Friday and Sunday.)
4. **`utilsforecast.losses.smape` is half the textbook sMAPE**: |e| / (|y| + |f|), bounded by 1, not
   2|e| / (|y| + |f|) bounded by 2. Our tables say which one they show.
5. **Coverage alone rewards wide intervals.** It is reported with the mean interval width and the interval
   (Winkler) score, and with the scaled quantile loss for the full distribution.
6. **The 24 errors of one origin are not independent.** Confidence intervals resample whole origins, in
   blocks of 7 consecutive days (moving block bootstrap), never single hours.
7. **Contamination.** The pretraining corpus of Chronos may contain European load data overlapping 2015-2020.
   Checked at the source in block 1; if overlap is possible, the zero-shot result is an upper bound.
8. **2020 is not a normal year.** The French lockdown starts 2020-03-17. The test year is 2019; the
   lockdown period is a separate stress regime, not mixed with the main verdict.
9. **Time zone.** Timestamps stay in UTC (no DST duplicates), so the daily profile shifts by one hour twice a
   year in UTC terms. Regimes (weekday, holiday) are computed on Europe/Paris local dates.
10. **Six origins decide nothing.** With 6 origins the model ranking depends on which 6. We quantify it by
    drawing many 6-origin subsets from the 365.
11. **Router features must be ex-ante.** Anything computed from the test period's actuals (e.g. realised
    error) can explain the regimes afterwards but cannot route.
12. **Physical constraints and intermittency do not exist in national load.** The router prototype cannot
    test those two features of the paper. Said explicitly rather than faked.

## Scope

In: blocks 1 to 4 of the plan. Block 2 is the core: `w25 backtest` prints the metrics x models table.
Out: covariates (temperature, holiday flags fed to Chronos-2), fine-tuning, a gradient-boosting specialist.
They are named as next steps where they would change a conclusion.

## Success criteria

- `w25 backtest` reprints the same table on two runs (same seed, same numbers to the 4th decimal).
- One sentence stating on which horizon and regime the TSFM beats or loses to AutoETS, using coverage and
  not only mean MASE, with a confidence interval.
- Every figure in the report is produced by code in the repo from committed results.
- `uv run pytest` green; `uv run ruff check` clean.

## Status of the success criteria (2026-08-14)

| Criterion | Status |
|---|---|
| Table reprinted identically | Met: two GPU runs give bit-identical forecasts; CPU vs GPU differ by at most 0.016 MW. The models are deterministic, so the seed is not what makes it reproducible |
| One sentence, with coverage and a CI | Met: [`02_backtest_harness/README.md`](02_backtest_harness/README.md), first section. The holiday CI was degenerate in the first version and was corrected after the review |
| Every figure produced by code from committed results | Met: `make figures`, except the router figures, which need `w25 router` (the panel forecasts, 57 MB, are not committed) |
| Tests and lint | Met: 31 fast tests, 4 slow tests (real Chronos weights, real data), ruff clean, CI on every push |

Blind spot 3 was wrong as stated (see `IMPLEMENTATION_NOTES.md`); the other eleven held.
