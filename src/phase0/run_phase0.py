# -*- coding: utf-8 -*-
"""Phase 0 — валидация сигнала (раздел 3, чек-лист раздела 12).

Пайплайн:
  1. Загрузка датасета (фичи + метки) из data/features/dataset.parquet,
     сплиты по разделу 3.3 с защитой OOS_FINAL (конфиг — единый источник).
  2. Базовое обучение XGBoost (мультикласс, sample_weight) + early stopping.
  3. Optuna-оптимизация гиперпараметров по Sharpe на ВАЛИДАЦИИ.
  4. Walk-forward валидация с embargo=72.
  5. Non-overlapping бэктест на OOS_ITERATIVE с ATR-выходом и funding,
     Newey-West / block bootstrap (maxlags=block_size=24).
  6. Бенчмарки 1–4 (одинаковые механики симуляции).
  7. KILL CRITERIA (3.11): единая per-bar шкала Sharpe, Information Ratio,
     gain-важность → вердикт GO / NO-GO.
  8. Сохранение результатов в JSON.

Использование:
  python3 run_phase0.py [--trials 40] [--bench-runs 300] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from config import load_config, split_bounds
from features import FEATURE_COLUMNS, generate_labels, atr
from backtest import (simulate_non_overlapping, metrics_from_trades,
                      newey_west_significance, block_bootstrap_significance,
                      returns_from_trades, returns_from_buyhold, sharpe_per_bar,
                      info_ratio, max_drawdown_from_returns, funding_by_side)
from benchmarks import run_all_benchmarks, sma_cross_preds
from oos_guard import guard_split, assert_no_final_overlap

CFG = load_config()
DATA_DIR = Path("data")
FEATURE_FILE = DATA_DIR / "features" / "dataset.parquet"
SYM = CFG["data"]["bybit_symbol"]
FUNDING_FILE = DATA_DIR / "funding_rates" / f"{SYM}_USDT" / "funding_history.parquet"

FEE = CFG["costs"]["taker_fee"]
SLIP = CFG["costs"]["slippage"]
HOLDING = CFG["backtest"]["holding_bars"]
EMBARGO = CFG["backtest"]["embargo_bars"]
ATR_EXIT = CFG["backtest"]["atr_exit"]
ATR_MULT = CFG["backtest"]["atr_multiplier"]
ATR_PERIOD = CFG["labels"]["atr_period"]
EARLY_STOP = CFG["optuna"]["early_stopping_rounds"]
BENCH_PCT = CFG["benchmarks"]["percentile_threshold"] * 100


# ───────────────────────── данные ─────────────────────────

def load_dataset() -> pd.DataFrame:
    ds = pd.read_parquet(FEATURE_FILE)
    if "label" not in ds.columns:
        lab, dist = generate_labels(ds)
        ds["label"] = lab
        print("Распределение классов:", {int(k): round(v, 3) for k, v in dist.items()})
    return ds.sort_values("timestamp").reset_index(drop=True)


def load_funding() -> pd.DataFrame | None:
    if not FUNDING_FILE.exists():
        print("  [warn] funding не найден, бэктест пойдёт без funding")
        return None
    return pd.read_parquet(FUNDING_FILE)


def align_funding_per_bar(ds: pd.DataFrame, funding: pd.DataFrame) -> np.ndarray:
    """Per-bar ставка funding: последний расчёт ≤ начала бара (без look-ahead)."""
    f = funding.sort_values("timestamp")
    ft = f["timestamp"].to_numpy()
    fr = f["fundingRate"].to_numpy(dtype=float)
    bars = ds["timestamp"].to_numpy()
    idx = np.searchsorted(ft, bars, side="right") - 1
    idx = np.clip(idx, 0, len(f) - 1)
    return fr[idx]


def get_split(ds: pd.DataFrame, name: str, allow_final: bool = False) -> pd.DataFrame:
    oos_start = pd.Timestamp(CFG["splits"]["oos_final"][0], tz="UTC")
    oos_end = pd.Timestamp(CFG["splits"]["oos_final"][1], tz="UTC")
    guard_split(name, allow_final, oos_start=oos_start, oos_end=oos_end)
    lo, hi = split_bounds(CFG, name)
    part = ds[(ds["timestamp"] >= lo) & (ds["timestamp"] <= hi)]
    assert_no_final_overlap(part["timestamp"], name, allow_final,
                            oos_start=oos_start, oos_end=oos_end)
    return part.reset_index(drop=True)


def backtest_kwargs(df: pd.DataFrame, funding_aligned: np.ndarray | None) -> dict:
    kw = {"fee": FEE, "slippage": SLIP, "holding": HOLDING}
    if ATR_EXIT:
        kw["atr_target"] = atr(df, ATR_PERIOD).to_numpy()
        kw["atr_multiplier"] = ATR_MULT
    if funding_aligned is not None:
        kw["funding_per_8h"] = funding_aligned
    return kw


# ───────────────────────── модель ─────────────────────────

def make_model(params: dict) -> xgb.XGBClassifier:
    p = dict(params)
    p.setdefault("random_state", 42)  # воспроизводимость даже после Optuna
    return xgb.XGBClassifier(**p, objective="multi:softprob", num_class=3,
                             eval_metric="mlogloss", tree_method="hist", n_jobs=-1)


def encode_y(y: pd.Series) -> np.ndarray:
    return (y + 1).to_numpy()  # -1,0,1 -> 0,1,2


def decode_pred(p: np.ndarray) -> np.ndarray:
    return p - 1


def train_and_predict(X_tr, y_tr, X_ev, y_ev, params):
    model = make_model(params)
    sw = compute_sample_weight("balanced", y_tr)
    model.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_ev, y_ev)],
              early_stopping_rounds=EARLY_STOP, verbose=False)
    return model


def val_sharpe(model, X_val, df_val) -> float:
    """Sharpe non-overlapping бэктеста предсказаний модели на валидации."""
    preds = decode_pred(model.predict(X_val))
    kw = backtest_kwargs(df_val, None)
    trades = simulate_non_overlapping(df_val, preds, **kw)
    if len(trades) < 5:
        return -99.0
    r = np.array([t["pnl_pct"] for t in trades])
    return sharpe_per_bar(returns_from_trades(trades, len(df_val)))


def optimize_params(X_tr, y_tr, X_va, df_va, n_trials: int):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = train_and_predict(X_tr, y_tr, X_va, encode_y(df_va["label"]), params)
        return val_sharpe(model, X_va, df_va)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ───────────────────────── walk-forward (3.8) ─────────────────────────

def walk_forward(ds: pd.DataFrame, params: dict,
                 funding: pd.DataFrame | None) -> list[dict]:
    results = []
    data = ds[ds["timestamp"] < pd.Timestamp(CFG["splits"]["oos_final"][0], tz="UTC")] \
        .reset_index(drop=True)
    fund_all = align_funding_per_bar(data, funding) if funding is not None else None
    X = data[FEATURE_COLUMNS]
    y = encode_y(data["label"])
    tscv = TimeSeriesSplit(n_splits=CFG["walkforward"]["n_splits"])
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(data)):
        tr_idx = tr_idx[:-EMBARGO]
        if len(tr_idx) == 0:
            continue
        model = train_and_predict(X.iloc[tr_idx], y[tr_idx],
                                  X.iloc[te_idx], y[te_idx], params)
        preds = decode_pred(model.predict(X.iloc[te_idx]))
        df_te = data.iloc[te_idx].reset_index(drop=True)
        fund_te = fund_all[te_idx] if fund_all is not None else None
        kw = backtest_kwargs(df_te, fund_te)
        trades = simulate_non_overlapping(df_te, preds, **kw)
        m = metrics_from_trades(trades)
        m["sharpe_pb"] = sharpe_per_bar(returns_from_trades(trades, len(df_te)))
        m.update({"fold": fold,
                  "period": f"{df_te['timestamp'].min():%Y-%m} .. {df_te['timestamp'].max():%Y-%m}",
                  "n_trades_exit_atr": sum(1 for t in trades if t["exit_reason"] == "atr")})
        results.append(m)
        print(f"  WF fold {fold}: {m.get('n_trades', 0)} сделок, "
              f"sharpe_pb={m.get('sharpe_pb', 0):.2f}, ret={m.get('total_return', 0):+.2%}")
    return results


# ───────────────────────── kill criteria (3.11) ─────────────────────────

def kill_criteria(oos_metrics: dict, rets_model: np.ndarray, rets_bh: np.ndarray,
                  rets_sma: np.ndarray | None, wf: list[dict], bm: dict,
                  model) -> dict:
    checks = {}
    sh_pb = sharpe_per_bar(rets_model)
    sh_bh = sharpe_per_bar(rets_bh)
    sh_sma = sharpe_per_bar(rets_sma) if rets_sma is not None else 0.0
    dd_mod = max_drawdown_from_returns(rets_model)
    dd_bh = max_drawdown_from_returns(rets_bh)
    ir = info_ratio(rets_model, rets_bh)

    checks["return_positive"] = oos_metrics["total_return"] > 0
    checks["sharpe_gt_bh_plus_0_3"] = sh_pb > sh_bh + 0.3
    checks["maxdd_lt_buyhold"] = dd_mod < dd_bh
    checks["ir_gt_0_5"] = ir > 0.5

    n = oos_metrics["n_trades"]
    t_req = 2.0 if n >= 300 else 2.5
    checks["n_trades_ge_100"] = n >= 100
    checks["t_stat_ok"] = oos_metrics.get("t_statistic", 0) > t_req
    checks["p_value_lt_0_05"] = oos_metrics.get("p_value", 1) < 0.05

    checks["bench2_top5pct"] = bm["benchmark_2_timing"]["model_percentile"] > BENCH_PCT
    checks["bench3_top5pct"] = bm["benchmark_3_direction"]["model_percentile"] > BENCH_PCT
    checks["better_sma_by_sharpe"] = sh_pb > sh_sma

    sharpes = [f["sharpe_pb"] for f in wf if f.get("n_trades", 0) > 0]
    checks["wf_all_positive"] = bool(sharpes) and all(s > 0 for s in sharpes)
    cv = None
    if len(sharpes) >= 2 and np.mean(sharpes) != 0:
        cv = float(np.std(sharpes) / abs(np.mean(sharpes)))
        checks["wf_cv_lt_0_5"] = cv < 0.5
    else:
        checks["wf_cv_lt_0_5"] = False

    # вклад фичи: gain (реальный вклад), а не number of splits
    imp = model.get_booster().feature_importances(importance_type="gain")
    top_share = float(imp.max() / imp.sum()) if imp.sum() else 1.0
    checks["no_dominant_feature"] = top_share <= 0.90
    checks["alpha_interpretation"] = bm["interpretation"].startswith(("FULL", "TIMING", "DIRECTION")) \
        and not bm["interpretation"].startswith("NO ALPHA")

    verdict = "GO" if all(checks.values()) else "NO-GO"
    return {"checks": checks, "verdict": verdict,
            "failed": [k for k, v in checks.items() if not v],
            "sharpe_pb_model": sh_pb, "sharpe_pb_buyhold": sh_bh, "sharpe_pb_sma": sh_sma,
            "maxdd_model": dd_mod, "maxdd_buyhold": dd_bh, "info_ratio": ir,
            "wf_cv": cv, "top_feature_share": top_share}


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None,
                    help="число Optuna-итераций (по умолчанию из конфига)")
    ap.add_argument("--bench-runs", type=int, default=None,
                    help="число прогонов случайных бенчмарков")
    ap.add_argument("--quick", action="store_true", help="без Optuna/WF (smoke-тест)")
    args = ap.parse_args()

    n_trials = args.trials if args.trials is not None else CFG["optuna"]["n_trials"]
    n_runs = args.bench_runs if args.bench_runs is not None else CFG["benchmarks"]["n_runs"]
    if args.quick:
        n_trials = n_runs = int(CFG["benchmarks"]["n_runs"] * 0.04) or 40

    print("=== PHASE 0: ВАЛИДАЦИЯ СИГНАЛА (SOL_USDT 1h) ===")
    ds = load_dataset()
    funding = load_funding()
    print(f"Датасет: {len(ds)} строк, {ds['timestamp'].min()} .. {ds['timestamp'].max()}")

    tr = get_split(ds, "train")
    va = get_split(ds, "validation")
    oos = get_split(ds, "oos_iterative")
    print(f"train={len(tr)} val={len(va)} oos_iterative={len(oos)}")

    X_tr, y_tr = tr[FEATURE_COLUMNS], encode_y(tr["label"])
    X_va, y_va = va[FEATURE_COLUMNS], encode_y(va["label"])

    base_params = CFG["model"]["base_params"]
    best_params, best_val_sharpe = base_params, None
    if not args.quick:
        print("\n[Optuna] подбор гиперпараметров по Val Sharpe...")
        best_params, best_val_sharpe = optimize_params(X_tr, y_tr, X_va, va, n_trials)
        print(f"  best val sharpe = {best_val_sharpe:.3f}; params = {best_params}")

    model = train_and_predict(X_tr, y_tr, X_va, y_va, best_params)

    print("\n[Walk-forward]")
    wf = [] if args.quick else walk_forward(ds, best_params, funding)

    print("\n[OOS_ITERATIVE: non-overlapping бэктест]")
    fund_oos = align_funding_per_bar(oos, funding) if funding is not None else None
    kw = backtest_kwargs(oos, fund_oos)
    preds_oos = decode_pred(model.predict(oos[FEATURE_COLUMNS]))
    trades = simulate_non_overlapping(oos, preds_oos, **kw)
    m = metrics_from_trades(trades)
    rets_model = returns_from_trades(trades, len(oos))
    m["sharpe_pb"] = sharpe_per_bar(rets_model)
    m["n_exit_atr"] = sum(1 for t in trades if t["exit_reason"] == "atr")
    m["funding_by_side"] = funding_by_side(trades)
    rets = np.array([t["pnl_pct"] for t in trades])
    nw = newey_west_significance(rets, maxlags=CFG["labels"]["forward_period"])
    bb = block_bootstrap_significance(rets, block_size=CFG["labels"]["forward_period"])
    m.update(nw)
    print(f"  сделок={m['n_trades']} return={m['total_return']:+.2%} sharpe_pb={m['sharpe_pb']:.2f} "
          f"MaxDD={max_drawdown_from_returns(rets_model):.1%} "
          f"ATR-выходы={m['n_exit_atr']}/{m['n_trades'] if m['n_trades'] else 0}")
    print(f"  Newey-West: t={nw['t_statistic']:.2f} p={nw['p_value']:.4f}; "
          f"bootstrap p={bb['p_value']:.4f}")

    print("\n[Бенчмарки 1-4]")
    bm = run_all_benchmarks(oos, preds_oos, m["total_return"], n_runs=n_runs,
                            fee=FEE, slip=SLIP, holding=HOLDING,
                            atr_target=kw.get("atr_target"), atr_multiplier=kw.get("atr_multiplier", 2.0),
                            funding_per_8h=kw.get("funding_per_8h"))
    print(f"  Buy&Hold:      ret={bm['buyhold']['total_return']:+.2%} sharpe_pb={bm['buyhold']['sharpe_pb']:.2f}")
    print(f"  B2 timing:     mean={bm['benchmark_2_timing']['mean']:+.2%} "
          f"percentile модели={bm['benchmark_2_timing']['model_percentile']:.1f}%")
    print(f"  B3 direction:  mean={bm['benchmark_3_direction']['mean']:+.2%} "
          f"percentile модели={bm['benchmark_3_direction']['model_percentile']:.1f}%")
    print(f"  B4 SMA cross:  ret={bm['benchmark_4_sma'].get('total_return', 0):+.2%} "
          f"sharpe_pb={bm['benchmark_4_sma'].get('sharpe_pb', 0):.2f}")
    print(f"  Интерпретация: {bm['interpretation']}")

    def _sample(arr: np.ndarray, cap: int = 300) -> list:
        arr = np.asarray(arr, dtype=float)
        return list(map(float, arr[::max(1, len(arr) // cap)][:cap])) if len(arr) else []

    # SMA per-bar для решения (та же механика, что у модели — но hold-to-reverse)
    sma_preds = sma_cross_preds(oos)
    sma_trades = simulate_non_overlapping(oos, sma_preds, fee=FEE, slippage=SLIP, holding=len(oos))
    rets_sma = returns_from_trades(sma_trades, len(oos))
    rets_bh = returns_from_buyhold(oos, FEE, SLIP)

    print("\n[KILL CRITERIA → вердикт]")
    verdict = kill_criteria(m, rets_model, rets_bh, rets_sma, wf, bm, model)
    print(f"  IR={verdict['info_ratio']:.2f} | доминантная фича "
          f"{verdict['top_feature_share']:.1%} | CV(folds)={verdict['wf_cv']}")
    print(f"  Не пройдено: {verdict['failed'] or 'нет'}")
    print(f"  >>> ВЕРДИКТ: {verdict['verdict']} <<<")

    out = {
        "phase": 0, "symbol": CFG["data"]["symbol"], "timeframe": CFG["data"]["timeframe"],
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "config": {"splits": CFG["splits"], "costs": CFG["costs"], "backtest": CFG["backtest"],
                   "n_trials": n_trials, "n_bench_runs": n_runs},
        "best_params": {k: v for k, v in best_params.items()},
        "best_val_sharpe": best_val_sharpe,
        "oos_iterative_metrics": m,
        "block_bootstrap": bb,
        "walkforward": wf,
        "benchmarks": {k: v for k, v in bm.items() if not k.startswith("_raw")},
        "bench2_dist_sample": _sample(bm["_raw_b2"]),
        "bench3_dist_sample": _sample(bm["_raw_b3"]),
        "kill_criteria": verdict,
        "funding_by_side": m["funding_by_side"],
        "feature_importance_gain": dict(zip(
            FEATURE_COLUMNS,
            map(float, model.get_booster().feature_importances(importance_type="gain")))),
    }
    Path("backtests/reports").mkdir(parents=True, exist_ok=True)
    path = f"backtests/reports/phase0_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nРезультаты сохранены: {path}")


if __name__ == "__main__":
    main()