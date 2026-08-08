"""Block 3: run both branches on the panel under each history length, score, build the routing units."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import yaml

from w25.backtest import chronos_forecast, levels_for, load_pipeline, qcol
from w25.config import RESULTS, ROOT
from w25.data import RAW
from w25.metrics import per_origin
from w25.router import FEATURES, country, features, load_panel

ROUTER_CFG = ROOT / "configs" / "router.yaml"
OUT = RESULTS / "router"
QUANTILES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_cfg() -> dict:
    return yaml.safe_load(ROUTER_CFG.read_text())


def router_cutoffs(cfg: dict) -> pd.DatetimeIndex:
    o = cfg["origins"]
    origins = pd.date_range(o["start"], o["end"], freq=f"{o['step_days']}D", tz="UTC")
    return origins + pd.Timedelta(hours=o["hour_utc"]) - pd.Timedelta(hours=1)


def specialist_for(history: int, cfg: dict):
    from statsforecast.models import MSTL, AutoETS

    for rule in cfg["branches"]["specialist"]:
        if history >= rule["when_history_at_least"]:
            if rule["model"] == "mstl":
                return MSTL(season_length=rule["seasons"], trend_forecaster=AutoETS(model="ZZN")), rule[
                    "window"
                ]
            return AutoETS(season_length=rule["season"], model="ZZA"), rule["window"]
    raise ValueError(history)


def run_specialist(
    panel: dict, cuts: pd.DatetimeIndex, history: int, cfg: dict
) -> tuple[pd.DataFrame, float]:
    from statsforecast import StatsForecast

    model, window = specialist_for(history, cfg)
    horizon = cfg["origins"]["horizon"]
    last = cuts[-1] + pd.Timedelta(hours=horizon)
    frames = []
    for uid, (filled, _) in panel.items():
        s = filled.loc[:last]
        frames.append(pd.DataFrame({"unique_id": uid, "ds": s.index.tz_localize(None), "y": s.to_numpy()}))
    df = pd.concat(frames, ignore_index=True)
    step = cfg["origins"]["step_days"] * 24
    sf = StatsForecast(models=[model], freq="h", n_jobs=-1)
    t0 = time.perf_counter()
    cv = sf.cross_validation(
        df=df,
        h=horizon,
        n_windows=len(cuts),
        step_size=step,
        input_size=min(history, window),
        level=levels_for(QUANTILES),
        refit=True,
    )
    seconds = time.perf_counter() - t0
    got = pd.DatetimeIndex(cv.cutoff.unique()).sort_values()
    if not got.equals(cuts.tz_localize(None)):
        raise RuntimeError("cross_validation windows do not match the router cutoffs")
    alias = str(model)
    out = pd.DataFrame(
        {
            "unique_id": cv.unique_id.to_numpy(),
            "model": f"specialist@{history}",
            "cutoff": pd.DatetimeIndex(cv.cutoff).tz_localize("UTC"),
            "ds": pd.DatetimeIndex(cv.ds).tz_localize("UTC"),
        }
    )
    out["h"] = ((out.ds - out.cutoff) / pd.Timedelta(hours=1)).astype(int)
    for q in QUANTILES:
        lv = round(abs(1 - 2 * q) * 100)
        col = alias if q == 0.5 else f"{alias}-{'lo' if q < 0.5 else 'hi'}-{lv}"
        out[qcol(q)] = cv[col].to_numpy()
    return out, seconds


def run_tsfm(
    panel: dict, cuts: pd.DatetimeIndex, history: int, pipe, cfg: dict
) -> tuple[pd.DataFrame, float]:
    frames, seconds = [], 0.0
    for uid, (filled, _) in panel.items():
        fc, sec = chronos_forecast(
            pipe,
            filled,
            cuts,
            cfg["origins"]["horizon"],
            history,
            QUANTILES,
            f"tsfm@{history}",
            batch_size=128,
            uid=uid,
        )
        frames.append(fc)
        seconds += sec
    return pd.concat(frames, ignore_index=True), seconds


def run(cfg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel(cfg, RAW)
    cuts = router_cutoffs(cfg)
    pipe = load_pipeline(cfg["branches"]["tsfm"]["checkpoint"])
    timing, fcs = {}, []
    for H in cfg["history_hours"]:
        tf, ts = run_tsfm(panel, cuts, H, pipe, cfg)
        sp, ss = run_specialist(panel, cuts, H, cfg)
        n = len(panel) * len(cuts)
        timing[H] = {"tsfm_gpu_s": round(ts, 2), "specialist_32cores_s": round(ss, 2), "forecasts": n}
        print(H, timing[H])
        fcs += [tf, sp]
    fc = pd.concat(fcs, ignore_index=True)
    fc.to_parquet(OUT / "forecasts.parquet", index=False)
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    score(cfg, panel, fc)


def score(cfg: dict, panel: dict, fc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, (filled, observed) in panel.items():
        actuals = pd.DataFrame({"ds": filled.index, "y": filled.to_numpy(), "observed": observed.to_numpy()})
        po = per_origin(fc[fc.unique_id == uid], actuals, 24, uid=uid)
        rows.append(po.assign(unique_id=uid))
    po = pd.concat(rows, ignore_index=True)
    po[["branch", "history"]] = po.model.str.split("@", expand=True)
    po["history"] = po.history.astype(int)
    po.to_parquet(OUT / "per_origin.parquet", index=False)
    return po


def units(cfg: dict, panel: dict, po: pd.DataFrame) -> pd.DataFrame:
    """One row per (series, history): ex-ante features and the mean MASE of each branch."""
    first = router_cutoffs(cfg)[0]
    loss = po.groupby(["unique_id", "history", "branch"]).agg(mase=("mase", "mean"), cov90=("cov90", "mean"))
    loss = loss.unstack("branch")
    rows = []
    for uid, (filled, _) in panel.items():
        hist = filled.loc[:first].to_numpy()
        for H in cfg["history_hours"]:
            f = features(hist[-H:])
            u = loss.loc[(uid, H)]
            rows.append(
                {
                    "unique_id": uid,
                    "country": country(uid),
                    "history": H,
                    **f,
                    "loss_tsfm": u[("mase", "tsfm")],
                    "loss_spec": u[("mase", "specialist")],
                    "cov90_tsfm": u[("cov90", "tsfm")],
                    "cov90_spec": u[("cov90", "specialist")],
                }
            )
    u = pd.DataFrame(rows)
    assert not u[FEATURES].isna().any().any()
    u.to_parquet(OUT / "units.parquet", index=False)
    return u


def load_units() -> pd.DataFrame:
    return pd.read_parquet(OUT / "units.parquet")


def drift(cfg: dict, panel: dict, stump: dict) -> pd.DataFrame:
    """Route of every series at every origin, with features on the trailing window only."""
    from w25.router import apply_stump

    n = cfg["drift"]["trailing_hours"]
    rows = []
    for uid, (filled, _) in panel.items():
        v = filled.to_numpy()
        pos = filled.index.get_indexer(router_cutoffs(cfg))
        for c, p in zip(router_cutoffs(cfg), pos, strict=True):
            rows.append({"unique_id": uid, "cutoff": c, **features(v[p + 1 - n : p + 1])})
    d = pd.DataFrame(rows)
    d["route_tsfm"] = apply_stump(stump, d)
    return d


def cpu_latency(panel: dict, cfg: dict, uid: str = "FR", n_origins: int = 20) -> dict:
    """Wall time of one forecast on one CPU core-equivalent call, per branch (no batching for the TSFM)."""
    import torch
    from statsforecast import StatsForecast

    torch.set_num_threads(4)
    filled = panel[uid][0]
    cuts = router_cutoffs(cfg)[:n_origins]
    pipe = load_pipeline(cfg["branches"]["tsfm"]["checkpoint"], "cpu")
    out = {}
    for H in [168, 2048]:
        t = []
        for c in cuts:
            _, s = chronos_forecast(pipe, filled, pd.DatetimeIndex([c]), 24, H, QUANTILES, "t", batch_size=1)
            t.append(s)
        out[f"tsfm_cpu4_ms@{H}"] = 1000 * float(np.median(t))
        model, window = specialist_for(H, cfg)
        t = []
        for c in cuts:
            s = filled.loc[:c].iloc[-min(H, window) :]
            df = pd.DataFrame({"unique_id": uid, "ds": s.index.tz_localize(None), "y": s.to_numpy()})
            t0 = time.perf_counter()
            StatsForecast(models=[model], freq="h", n_jobs=1).forecast(df=df, h=24, level=[90])
            t.append(time.perf_counter() - t0)
        out[f"specialist_cpu1_ms@{H}"] = 1000 * float(np.median(t))
    return out
