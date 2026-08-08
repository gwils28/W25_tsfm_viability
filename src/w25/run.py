"""Run the block 2 backtest and store the forecasts under results/."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from w25.backtest import (
    chronos_forecast,
    cutoffs,
    is_tsfm,
    load_pipeline,
    statistical_model,
    statsforecast_cv,
)
from w25.config import RESULTS
from w25.data import clean, download, load_columns

FORECASTS = RESULTS / "forecasts"


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - global state some libraries read
    torch.manual_seed(seed)


def prepare(proto: dict) -> tuple[pd.Series, pd.Series]:
    d = proto["data"]
    download(d["url"], d["sha256"])
    raw = load_columns([d["column"]])[d["column"]]
    return clean(raw, tuple(d["physical_range"]), d["fill_linear_max_hours"])


def save_actuals(filled: pd.Series, observed: pd.Series, out: Path = RESULTS) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"ds": filled.index, "y": filled.to_numpy(), "observed": observed.to_numpy()})
    path = out / "actuals_fr.parquet"
    df.to_parquet(path, index=False)
    return path


def run_model(
    name: str, cfg: dict, series: pd.Series, cuts: pd.DatetimeIndex, proto: dict
) -> tuple[pd.DataFrame, float]:
    horizon, qs = proto["origins"]["horizon"], proto["quantiles"]
    if is_tsfm(cfg):
        pipe = load_pipeline(cfg["checkpoint"])
        return chronos_forecast(pipe, series, cuts, horizon, cfg["context_hours"], qs, name)
    t0 = time.perf_counter()
    model = statistical_model(name, cfg)
    fc = statsforecast_cv(series, model, name, cuts, horizon, cfg.get("train_hours"), qs)
    return fc, time.perf_counter() - t0


def run(proto: dict, models: list[str] | None = None, out: Path = FORECASTS) -> dict[str, float]:
    seed_everything(proto["seed"])
    filled, observed = prepare(proto)
    save_actuals(filled, observed)
    o = proto["origins"]
    cuts = cutoffs(o["start"], o["end"], o["hour_utc"])
    out.mkdir(parents=True, exist_ok=True)
    timing_path = out / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    for name, cfg in proto["models"].items():
        if models and name not in models:
            continue
        fc, seconds = run_model(name, cfg, filled, cuts, proto)
        fc.to_parquet(out / f"{name}.parquet", index=False)
        timing[name] = {"seconds": round(seconds, 2), "origins": len(cuts)}
        print(f"{name:22s} {len(cuts)} origins in {seconds:7.1f} s")
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    return timing


def load_forecasts(models: list[str], path: Path = FORECASTS) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(path / f"{m}.parquet") for m in models], ignore_index=True)


def load_actuals(path: Path = RESULTS / "actuals_fr.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)
