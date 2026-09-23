# -*- coding: utf-8 -*-
"""Phase 0 — валидация сигнала (раздел 3, чек-лист раздела 12).

Пайплайн:
  1. Загрузка датасета (фичи + метки), сплиты по разделу 3.3 с защитой OOS_FINAL.
  2. Базовое обучение XGBoost (мультикласс, sample_weight — НЕ scale_pos_weight).
  3. Optuna-оптимизация гиперпараметров по Sharpe на ВАЛИДАЦИИ (n_trials из конфига).
  4. Walk-forward валидация с embargo=72 (раздел 3.8).
  5. Non-overlapping бэктест на OOS_ITERATIVE + Newey-West / block bootstrap (3.9).
  6. Бенчмарки 1–4 (раздел 3.10).
  7. Проверка KILL CRITERIA (раздел 3.11) → вердикт GO / NO-GO.
  8. Сохранение результатов в JSON + печать сводки.

Использование:
  python3 run_phase0.py [--trials 40] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURE_COLUMNS, generate_labels  # noqa: E402
from backtest import (simulate_non_overlapping, metrics_from_trades,  # noqa: E402
                      newey_west_significance, block_bootstrap_significance,
                      equity_curve, sharpe_from_trades)
from benchmarks import run_all_benchmarks  # noqa: E402
from oos_guard import guard_split, assert_no_final_overlap  # noqa: E402

DATA_DIR = Path("data")
SPLITS = {
    "train": ("2022-01-01", "2024-06-30"),
    "validation": ("2024-07-01", "2024-12-31"),
    "oos_iterative": ("2025-01-01", "2026-02-28"),
}
FEE, SLIP, HOLDING, EMBARGO = 0.001, 0.0005, 24, 72
# Раздел 3.6: торговать только сильными сигналами (Phase 0 — сырые вероятности,
# калибровка добавляется в Phase 1). Порог подбирается на ВАЛИДАЦИИ, никогда на OOS.
THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]


def predict_with_threshold(model, X) -> np.ndarray:
    """Дискретные предсказания с порогом уверенности (argmax, если max prob >= thr)."""
    return predict_proba_with_threshold(model, X)[0]


def predict_proba_with_threshold(model, X):
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1) - 1
    return pred, proba


def select_threshold(model, X_val, df_val) -> tuple[float, float]:
    """Выбор порога по Sharpe на валидации; при равенстве — более высокий порог (меньше сделок)."""
    proba = model.predict_proba(X_val)
    best_thr, best_sh = THRESHOLD_GRID[0], -99.0
    for thr in THRESHOLD_GRID:
        preds = proba.argmax(axis=1) - 1
        preds[proba.max(axis=1) < thr] = 0
        trades = simulate_non_overlapping(df_val, preds, fee=FEE, slippage=SLIP, holding=HOLDING)
        if len(trades) < 5:
            continue
        sh = sharpe_from_trades(np.array([t["pnl_pct"] for t in trades]), 365)
        if sh >= best_sh:
            best_thr, best_sh = thr, sh
    return best_thr, best_sh


def load_dataset() -> pd.DataFrame:
    ds = pd.read_parquet(DATA_DIR / "features/dataset.parquet")
    if "label" not in ds.columns:
        lab, dist = generate_labels(ds)
        ds["label"] = lab
        print("Распределение классов:", {int(k): round(v, 3) for k, v in dist.items()})
    return ds.sort_values("timestamp").reset_index(drop=True)


def get_split(ds: pd.DataFrame, name: str, allow_final=False) -> pd.DataFrame:
    guard_split(name, allow_final)
    lo, hi = SPLITS[name]
    part = ds[(ds["timestamp"] >= pd.Timestamp(lo, tz="UTC")) &
              (ds["timestamp"] <= pd.Timestamp(hi, tz="UTC") + pd.Timedelta(hours=23))]
    assert_no_final_overlap(part["timestamp"], name, allow_final)
    return part.reset_index(drop=True)


def make_model(params: dict) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**params, objective="multi:softprob", num_class=3,
                             eval_metric="mlogloss", tree_method="hist", n_jobs=-1)


def encode_y(y: pd.Series) -> np.ndarray:
    return (y + 1).to_numpy()  # -1,0,1 -> 0,1,2


def decode_pred(p: np.ndarray) -> np.ndarray:
    return p - 1


def train_and_predict(X_tr, y_tr, X_ev, y_ev, params):
    model = make_model(params)
    sw = compute_sample_weight("balanced", y_tr)
    model.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_ev, y_ev)], verbose=False)
    return model


def val_sharpe(model, X_val, df_val) -> float:
    """Sharpe non-overlapping бэктеста предсказаний модели на валидации."""
    preds = decode_pred(model.predict(X_val))
    trades = simulate_non_overlapping(df_val, preds, fee=FEE, slippage=SLIP, holding=HOLDING)
    if len(trades) < 5:
        return -99.0
    r = np.array([t["pnl_pct"] for t in trades])
    return sharpe_from_trades(r, 365)


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


def walk_forward(ds: pd.DataFrame, params: dict) -> list[dict]:
    """Раздел 3.8: TimeSeriesSplit по всему обучающему периметру + embargo 72."""
    from sklearn.model_selection import TimeSeriesSplit
    results = []
    data = ds[ds["timestamp"] < pd.Timestamp("2026-03-01", tz="UTC")].reset_index(drop=True)
    X = data[FEATURE_COLUMNS]
    y = encode_y(data["label"])
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(data)):
        tr_idx = tr_idx[:-EMBARGO]
        if len(tr_idx) == 0:
            continue
        model = train_and_predict(X.iloc[tr_idx], y[tr_idx],
                                  X.iloc[te_idx], y[te_idx], params)
        preds = decode_pred(model.predict(X.iloc[te_idx]))
        df_te = data.iloc[te_idx].reset_index(drop=True)
        trades = simulate_non_overlapping(df_te, preds, fee=FEE, slippage=SLIP, holding=HOLDING)
        m = metrics_from_trades(trades)
        m.update({"fold": fold,
                  "period": f"{df_te['timestamp'].min():%Y-%m} .. {df_te['timestamp'].max():%Y-%m}"})
        results.append(m)
        print(f"  WF fold {fold}: {m.get('n_trades', 0)} сделок, "
              f"sharpe={m.get('sharpe', 0):.2f}, ret={m.get('total_return', 0):+.2%}")
    return results


def kill_criteria(oos_metrics: dict, wf: list[dict], bm: dict, model, X_oos) -> dict:
    """Раздел 3.11 — бинарный вердикт GO/NO-GO по пред-записанным условиям."""
    checks = {}
    r_model = oos_metrics["total_return"]
    sh_model = oos_metrics["sharpe"]
    checks["return_positive"] = r_model > 0
    checks["sharpe_gt_bh_plus_0_3"] = sh_model > bm["buyhold"]["sharpe"] + 0.3
    checks["maxdd_lt_buyhold"] = oos_metrics["max_drawdown"] < bm["buyhold"]["max_drawdown"]
    n = oos_metrics["n_trades"]
    checks["n_trades_ge_100"] = n >= 100
    t_req = 2.0 if n >= 300 else 2.5
    checks["t_stat_ok"] = oos_metrics.get("t_statistic", 0) > t_req
    checks["p_value_lt_0_05"] = oos_metrics.get("p_value", 1) < 0.05
    checks["bench2_top5pct"] = bm["benchmark_2_timing"]["model_percentile"] > 95
    checks["bench3_top5pct"] = bm["benchmark_3_direction"]["model_percentile"] > 95
    checks["better_sma_by_sharpe"] = sh_model > bm["benchmark_4_sma"].get("sharpe", 99)
    sharpes = [f["sharpe"] for f in wf if f.get("n_trades", 0) > 0]
    checks["wf_all_positive"] = bool(sharpes) and all(s > 0 for s in sharpes)
    if len(sharpes) >= 2 and np.mean(sharpes) != 0:
        cv = float(np.std(sharpes) / abs(np.mean(sharpes)))
        checks["wf_cv_lt_0_5"] = cv < 0.5
    else:
        cv, checks["wf_cv_lt_0_5"] = None, False
    imp = model.get_booster().feature_importances_
    top_share = float(imp.max() / imp.sum()) if imp.sum() else 1.0
    checks["no_dominant_feature"] = top_share <= 0.90
    checks["alpha_interpretation"] = bm["interpretation"].startswith(("FULL", "TIMING", "DIRECTION")) \
        and not bm["interpretation"].startswith("NO ALPHA")
    verdict = "GO" if all(checks.values()) else "NO-GO"
    return {"checks": checks, "verdict": verdict,
            "failed": [k for k, v in checks.items() if not v],
            "wf_cv": cv, "top_feature_share": top_share}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--quick", action="store_true", help="без Optuna/WF (быстрый smoke-тест)")
    args = ap.parse_args()

    print("=== PHASE 0: ВАЛИДАЦИЯ СИГНАЛА (SOL_USDT 1h) ===")
    ds = load_dataset()
    print(f"Датасет: {len(ds)} строк, {ds['timestamp'].min()} .. {ds['timestamp'].max()}")

    tr = get_split(ds, "train")
    va = get_split(ds, "validation")
    oos = get_split(ds, "oos_iterative")
    print(f"train={len(tr)} val={len(va)} oos_iterative={len(oos)}")

    X_tr, y_tr = tr[FEATURE_COLUMNS], encode_y(tr["label"])
    X_va, y_va = va[FEATURE_COLUMNS], encode_y(va["label"])

    base_params = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42)

    best_params, best_val_sharpe = base_params, None
    if not args.quick:
        print("\n[Optuna] подбор гиперпараметров по Val Sharpe...")
        best_params, best_val_sharpe = optimize_params(X_tr, y_tr, X_va, va, args.trials)
        print(f"  best val sharpe = {best_val_sharpe:.3f}; params = {best_params}")

    model = train_and_predict(X_tr, y_tr, X_va, y_va, best_params)

    print("\n[Walk-forward]")
    wf = [] if args.quick else walk_forward(ds, best_params)

    print("\n[OOS_ITERATIVE: non-overlapping бэктест]")
    preds_oos = decode_pred(model.predict(oos[FEATURE_COLUMNS]))
    trades = simulate_non_overlapping(oos, preds_oos, fee=FEE, slippage=SLIP, holding=HOLDING)
    m = metrics_from_trades(trades)
    rets = np.array([t["pnl_pct"] for t in trades])
    nw = newey_west_significance(rets, maxlags=5)
    bb = block_bootstrap_significance(rets, block_size=10)
    m.update(nw)
    print(f"  сделок={m['n_trades']} return={m['total_return']:+.2%} sharpe={m['sharpe']:.2f} "
          f"MaxDD={m['max_drawdown']:.1%} winrate={m['win_rate']:.1%}")
    print(f"  Newey-West: t={nw['t_statistic']:.2f} p={nw['p_value']:.4f}; "
          f"bootstrap p={bb['p_value']:.4f}")

    print("\n[Бенчмарки 1-4]")
    bm = run_all_benchmarks(oos, preds_oos, m["total_return"], n_runs=150 if not args.quick else 40,
                            fee=FEE, slippage=SLIP, holding=HOLDING)
    print(f"  Buy&Hold:      ret={bm['buyhold']['total_return']:+.2%} sharpe={bm['buyhold']['sharpe']:.2f}")
    print(f"  B2 timing:     mean={bm['benchmark_2_timing']['mean']:+.2%} "
          f"percentile модели={bm['benchmark_2_timing']['model_percentile']:.1f}%")
    print(f"  B3 direction:  mean={bm['benchmark_3_direction']['mean']:+.2%} "
          f"percentile модели={bm['benchmark_3_direction']['model_percentile']:.1f}%")
    print(f"  B4 SMA cross:  ret={bm['benchmark_4_sma'].get('total_return', 0):+.2%} "
          f"sharpe={bm['benchmark_4_sma'].get('sharpe', 0):.2f}")
    print(f"  Интерпретация: {bm['interpretation']}")

    print("\n[KILL CRITERIA → вердикт]")
    verdict = kill_criteria(m, wf, bm, model, oos[FEATURE_COLUMNS])
    print(f"  Не пройдено: {verdict['failed'] or 'нет'}")
    print(f"  >>> ВЕРДИКТ: {verdict['verdict']} <<<")

    out = {
        "phase": 0, "symbol": "SOL_USDT", "timeframe": "1h",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "splits": {k: list(v) for k, v in SPLITS.items()},
        "best_params": {k: v for k, v in best_params.items()},
        "best_val_sharpe": best_val_sharpe,
        "oos_iterative_metrics": m,
        "block_bootstrap": bb,
        "walkforward": wf,
        "benchmarks": {k: v for k, v in bm.items() if not k.startswith("_raw")},
        "kill_criteria": verdict,
        "feature_importance": dict(zip(FEATURE_COLUMNS,
                                       map(float, model.get_booster().feature_importances_))),
    }
    Path("backtests/reports").mkdir(parents=True, exist_ok=True)
    path = f"backtests/reports/phase0_{pd.Timestamp.now(tz='UTC'):%Y-%m-%d}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nРезультаты сохранены: {path}")


if __name__ == "__main__":
    main()
