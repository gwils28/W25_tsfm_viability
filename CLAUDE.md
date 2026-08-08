# W25 TSFM viability

## Commands
- `uv sync` then `uv run pytest` (fast tests; `-m slow` downloads Chronos weights and runs them on CPU).
- `uv run ruff check src tests && uv run ruff format src tests`.
- `make all` reproduces every table and figure; `make report` renders the Quarto site into `docs/`.
- Quarto needs `QUARTO_PYTHON=.venv/bin/python` and the kernel `w25` (`uv run python -m ipykernel install --user --name w25`).

## Pitfalls
- `data/raw/` is the pinned OPSD file (sha256 in `configs/protocol.yaml`); never edit it, never commit it.
- `configs/protocol.yaml` and `configs/router.yaml` are pre-registered. A change after seeing results is a
  deviation: add it with a dated comment, keep the original, log it in `IMPLEMENTATION_NOTES.md`.
- Chronos-Bolt is trained on quantiles 0.1-0.9 only: its "0.05/0.95" are not a 90% interval.
- Chronos-2 batches here hold several origins of one series: `cross_learning` must stay False (leak).
- statsforecast `cross_validation` places windows from the end of the frame: cut the frame at the last target.
- utilsforecast `smape` is |e|/(|y|+|f|), half the textbook value.
- Timestamps are UTC everywhere; regimes use the Europe/Paris date.

## Conventions
- A number in a README or the report comes from `results/` or a command shown next to it.
- Figures: `w25.plots` palette, one colour per model, every figure looked at after rendering.

## Compaction
Keep: files changed, test commands, the deviations list, and every modelling decision with its reason.

Canary: end every final summary with the line "W25 ok".
