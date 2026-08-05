"""Command line: `w25 <command>`."""

from __future__ import annotations

from pathlib import Path

import typer

from w25.config import load_protocol

app = typer.Typer(add_completion=False, no_args_is_help=True)
BLOCK2 = Path("02_backtest_harness")


@app.command()
def download() -> None:
    """Download the pinned OPSD file and check its sha256."""
    from w25.data import download as dl

    d = load_protocol()["data"]
    print(dl(d["url"], d["sha256"]))


@app.command()
def figures_data() -> None:
    """Figures of the series and of the protocol (block 2, before any model)."""
    from w25.figures_data import all_figures

    for p in all_figures(load_protocol(), BLOCK2 / "figures"):
        print(p)


@app.command()
def forecast(models: list[str] = typer.Option(None, "--model", "-m", help="subset of models")) -> None:
    """Run the rolling-origin backtest and store the forecasts in results/forecasts/."""
    from w25.run import run

    run(load_protocol(), models)


@app.command()
def backtest(stress: bool = typer.Option(False, help="also print the lockdown window")) -> None:
    """Score the stored forecasts and print the metrics x models table of the test year."""
    from w25.study import fmt, score, table, verdicts

    proto = load_protocol()
    po, _ = score(proto)
    print("Test year 2019, mean over origins (smape = utilsforecast's |e|/(|y|+|f|), half the textbook one)")
    print(fmt(table(po)))
    print(
        "\nPre-registered verdict against AutoETS (MASE ratio, 95% block-bootstrap CI; coverage of the 90%)"
    )
    print(fmt(verdicts(po, proto).set_index("model").drop(columns="regime")))
    ref = proto["verdict"]["reference_deviation"]
    print(f"\nDEVIATION: same verdict against {ref} (AutoETS forecasts a flat line at most origins)")
    print(fmt(verdicts(po, proto, ref=ref).set_index("model").drop(columns="regime")))
    if stress:
        print("\nLockdown window 2020-03-17 to 2020-05-10")
        print(fmt(table(po, "stress")))


if __name__ == "__main__":
    app()
