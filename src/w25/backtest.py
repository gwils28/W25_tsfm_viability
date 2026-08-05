"""Rolling-origin backtest. One forecast frame per model, in a single long format:

    unique_id, model, cutoff, ds, h, q0.05, q0.1, ..., q0.95

`cutoff` is the last hour the model may read (origin minus one hour), `h` runs from 1 to 24, and
q0.5 is the point forecast of every model (median for Chronos, mean for the statistical models,
whose intervals are symmetric).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

QCOLS_FMT = "q{:g}"


def qcol(q: float) -> str:
    return QCOLS_FMT.format(q)


def cutoffs(start: str, end: str, hour_utc: int = 0) -> pd.DatetimeIndex:
    """Last readable hour of each daily origin between start and end (both included)."""
    origins = pd.date_range(start, end, freq="D", tz="UTC") + pd.Timedelta(hours=hour_utc)
    return origins - pd.Timedelta(hours=1)


def levels_for(quantiles: list[float]) -> list[int]:
    """Symmetric interval levels that produce the requested quantiles (median excluded)."""
    lv = sorted({round(abs(1 - 2 * q) * 100) for q in quantiles if q != 0.5})
    return [int(v) for v in lv]


def _to_naive(idx: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).tz_convert("UTC").tz_localize(None)


def statsforecast_cv(
    series: pd.Series,
    model,
    name: str,
    cuts: pd.DatetimeIndex,
    horizon: int,
    train_hours: int | None,
    quantiles: list[float],
    uid: str = "FR",
) -> pd.DataFrame:
    """Refit `model` at every cutoff on its last `train_hours` hours and forecast `horizon` steps.

    StatsForecast's cross_validation places its windows from the end of the frame backwards,
    so the frame is cut to end exactly at the last target hour: the windows then fall on `cuts`.
    """
    from statsforecast import StatsForecast

    step = pd.Timedelta(hours=24)
    if not (cuts[1:] - cuts[:-1] == step).all():
        raise ValueError("cutoffs must be consecutive days")
    last_target = cuts[-1] + pd.Timedelta(hours=horizon)
    s = series.loc[:last_target]
    df = pd.DataFrame({"unique_id": uid, "ds": _to_naive(s.index), "y": s.to_numpy()})
    sf = StatsForecast(models=[model], freq="h", n_jobs=1)
    levels = levels_for(quantiles)
    cv = sf.cross_validation(
        df=df, h=horizon, n_windows=len(cuts), step_size=24, input_size=train_hours, level=levels, refit=True
    )
    got = pd.DatetimeIndex(cv.cutoff.unique()).sort_values()
    if not got.equals(_to_naive(cuts)):
        raise RuntimeError("cross_validation windows do not match the protocol cutoffs")
    alias = str(model)
    out = pd.DataFrame(
        {
            "unique_id": uid,
            "model": name,
            "cutoff": pd.DatetimeIndex(cv.cutoff).tz_localize("UTC"),
            "ds": pd.DatetimeIndex(cv.ds).tz_localize("UTC"),
        }
    )
    out["h"] = ((out.ds - out.cutoff) / pd.Timedelta(hours=1)).astype(int)
    for q in quantiles:
        if q == 0.5:
            out[qcol(q)] = cv[alias].to_numpy()
        else:
            lv = round(abs(1 - 2 * q) * 100)
            side = "lo" if q < 0.5 else "hi"
            out[qcol(q)] = cv[f"{alias}-{side}-{lv}"].to_numpy()
    return out


def contexts(series: pd.Series, cuts: pd.DatetimeIndex, context_hours: int) -> np.ndarray:
    """Matrix (n_cutoffs, context_hours) of the values up to and including each cutoff."""
    pos = series.index.get_indexer(cuts)
    if (pos < 0).any():
        raise KeyError("a cutoff is not in the series index")
    if (pos + 1 - context_hours < 0).any():
        raise ValueError("not enough history for the requested context")
    values = series.to_numpy(dtype=np.float32)
    return np.stack([values[p + 1 - context_hours : p + 1] for p in pos])


def chronos_forecast(
    pipeline,
    series: pd.Series,
    cuts: pd.DatetimeIndex,
    horizon: int,
    context_hours: int,
    quantiles: list[float],
    name: str,
    batch_size: int = 64,
    uid: str = "FR",
) -> tuple[pd.DataFrame, float]:
    """Zero-shot forecasts at every cutoff. Returns (frame, seconds of inference).

    Each batch element is a different origin of the same series. Chronos-2 can share information
    across a batch (`cross_learning`); here that would let an old origin read a newer context, i.e.
    the future. It is passed as False explicitly.
    """
    import torch

    ctx = contexts(series, cuts, context_hours)
    kwargs = {"cross_learning": False} if type(pipeline).__name__ == "Chronos2Pipeline" else {}
    preds = []
    t0 = time.perf_counter()
    for i in range(0, len(ctx), batch_size):
        batch = torch.tensor(ctx[i : i + batch_size])
        if kwargs:
            q, _ = pipeline.predict_quantiles(
                list(batch), prediction_length=horizon, quantile_levels=quantiles, **kwargs
            )
            q = torch.stack([x.squeeze(0) for x in q])  # (batch, horizon, n_q)
        else:
            q, _ = pipeline.predict_quantiles(batch, prediction_length=horizon, quantile_levels=quantiles)
        preds.append(q.float().cpu().numpy())
    seconds = time.perf_counter() - t0
    arr = np.concatenate(preds)  # (n_cutoffs, horizon, n_q)

    n = len(cuts)
    cut_rep = np.repeat(cuts, horizon)
    h = np.tile(np.arange(1, horizon + 1), n)
    out = pd.DataFrame(
        {
            "unique_id": uid,
            "model": name,
            "cutoff": cut_rep,
            "ds": cut_rep + pd.to_timedelta(h, unit="h"),
            "h": h,
        }
    )
    for j, q in enumerate(quantiles):
        out[qcol(q)] = arr[:, :, j].reshape(-1)
    return out, seconds


def load_pipeline(checkpoint: str, device: str | None = None):
    import torch
    from chronos import BaseChronosPipeline

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return BaseChronosPipeline.from_pretrained(checkpoint, device_map=device, dtype=torch.float32)


def statistical_model(name: str, cfg: dict):
    from statsforecast.models import MSTL, AutoETS, SeasonalNaive

    if name.startswith("seasonal_naive"):
        return SeasonalNaive(season_length=cfg["season"])
    if name.startswith("auto_ets"):
        return AutoETS(season_length=cfg["season"], model=cfg.get("spec", "ZZZ"))
    if name == "mstl":
        return MSTL(season_length=cfg["seasons"], trend_forecaster=AutoETS(model="ZZN"))
    raise KeyError(name)


def is_tsfm(cfg: dict) -> bool:
    return "checkpoint" in cfg
