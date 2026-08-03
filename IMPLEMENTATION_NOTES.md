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
