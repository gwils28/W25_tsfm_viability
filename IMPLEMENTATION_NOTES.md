# Implementation notes

Deviations from the plan and from `SPEC.md`, conservative choices, and claims corrected by a measurement.
Newest last.

## Data

- **The "strawman" claim of SPEC blind spot 3 was wrong.** Over 2017-2018 (all hours, local weekday),
  SeasonalNaive(24) has a lower MAE than SeasonalNaive(168) on Tuesday to Friday and on Sunday; the weekly naive
  wins only on Monday and Saturday (`02_backtest_harness/figures/strawman_naive.png`). Both naives stay in the
  bench, as baselines of equal standing; the SPEC line is amended, not deleted.
- The corrupted hour (158 GW, 2020-07-07 17:00 UTC) is outside the test year and the stress window. It would
  only have touched inputs of the last origins; removed anyway by the physical range check.
- Long gaps are filled with the value one week earlier rather than interpolated: in a 13 h gap the
  interpolation crosses a full morning ramp.

## Block 2

- **DEVIATION: AutoETS forecasts a flat line at 572 of 608 origins.** statsforecast's AutoETS selects the model
  by AICc, a one-step-ahead likelihood. At hourly resolution the level-only ETS(A,N,N) has the best one-step fit
  (AICc 14945 vs 15180 for the best seasonal model on the window ending 2019-03-11), so it wins, and its 24-step
  forecast is the last observed value. The pre-registered reference `auto_ets` is kept and its verdict is reported
  as written; `auto_ets_seasonal` (same model, additive seasonality forced: `model="ZZA"`) is added after seeing
  the scores, and every verdict is also given against it, labelled as a deviation.
- Bolt's "90%" interval is its 80% one (blind spot 1): cov80 = cov90 = 0.858 over the test year. It lands inside
  the pre-registered band [0.85, 0.95] because its 80% interval over-covers; the CI of that coverage
  (0.833-0.884) crosses the band's lower edge. The rule says WINS; the reading says "by accident".
