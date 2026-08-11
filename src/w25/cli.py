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


@app.command()
def figures_results() -> None:
    """Block 2 figures from the scored backtest, and verdicts by regime (results/verdicts_by_regime.csv)."""
    from w25.figures_results import all_figures

    for ref, s in all_figures(load_protocol(), BLOCK2 / "figures").items():
        print(f"6 origins vs {ref}: {s['agree']:.1%} of draws agree with the 365-origin ranking", s)


@app.command()
def ablation(rerun: bool = typer.Option(False, help="recompute instead of reading the CSV")) -> None:
    """EXPLORATORY: Chronos context length on the test year (results/ablation_context.csv)."""
    import pandas as pd

    from w25.ablation import figure, run
    from w25.config import RESULTS

    proto = load_protocol()
    ab = run(proto) if rerun else pd.read_csv(RESULTS / "ablation_context.csv")
    print(figure(ab, pd.read_parquet(RESULTS / "per_origin.parquet"), BLOCK2 / "figures"))


@app.command()
def router(rerun: bool = typer.Option(True, help="recompute both branches")) -> None:
    """Block 3: both branches on the panel, the routing units, policies, frontier and drift figures."""
    import pandas as pd

    from w25.data import RAW
    from w25.router import load_panel
    from w25.router_eval import run_all
    from w25.router_run import OUT, load_cfg, run, units

    cfg = load_cfg()
    panel = load_panel(cfg, RAW)
    if rerun:
        run(cfg)
    units(cfg, panel, pd.read_parquet(OUT / "per_origin.parquet"))
    print(run_all(cfg, panel, Path("03_complexity_router/figures")))
    print(pd.read_csv(OUT / "policies.csv").query("history == 'all'").to_string(index=False))


@app.command()
def figures_misc() -> None:
    """Figures of blocks 1, 3 (architecture) and 4."""
    from w25 import plots as P
    from w25.figures_design import architecture
    from w25.figures_mase import mape_vs_mase, scale_choice, scale_free_panel, smape_asymmetry
    from w25.figures_watch import router_claim

    P.setup()
    router_claim(Path("01_tech_watch/figures"))
    architecture(Path("03_complexity_router/figures"))
    out = Path("04_reference_reading/figures")
    mape_vs_mase(out)
    smape_asymmetry(out)
    scale_choice(out)
    scale_free_panel(out)


@app.command()
def calendar(
    rerun: bool = typer.Option(False, help="recompute instead of reading the stored scores"),
) -> None:
    """EXPLORATORY: Chronos-2 with known holiday and bridge-day flags as covariates (block 2)."""
    import pandas as pd

    from w25.calendar_cov import figure, run
    from w25.config import RESULTS

    po = run(load_protocol()) if rerun else pd.read_parquet(RESULTS / "per_origin_calendar.parquet")
    print(po.groupby("day_type")[["mase", "cov90"]].mean().round(3))
    print(figure(po, BLOCK2 / "figures"))


if __name__ == "__main__":
    app()
