# -*- coding: utf-8 -*-
"""Правильные бенчмарки (раздел 3.10): 4 типа, раздельная проверка навыков.

Ревизия: случайные бенчмарки (B2/B3) теперь симулируются С ТЕМИ ЖЕ механиками,
что и модель (holding, ATR-выход, funding) — иначе сравнение несправедливо.
Добавлен sma_cross_preds (единый источник для бенчмарка 4 и решения GO/NO-GO).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import (simulate_non_overlapping, metrics_from_trades,
                      returns_from_buyhold, returns_from_trades, sharpe_per_bar)


def _zero_preds(n: int) -> np.ndarray:
    return np.zeros(n, dtype=int)


def bench_buyhold(df: pd.DataFrame, fee=0.001, slip=0.0005) -> dict:
    """Бенчмарк 1: купил в начале, продал в конце (с издержками)."""
    rets = returns_from_buyhold(df, fee, slip)
    ret = float(rets.sum())  # лог-сумма ≈ доходность (для отчёта)
    eq = np.cumprod(1 + rets)
    return {
        "total_return": float(eq[-1] - 1),
        "max_drawdown": float(abs((eq / np.maximum.accumulate(eq) - 1).min())),
        "sharpe": float(np.mean(df["close"].pct_change().dropna()) /
                        np.std(df["close"].pct_change().dropna()) * np.sqrt(24 * 365)),
        "sharpe_pb": sharpe_per_bar(rets, 24 * 365),  # единая шкала с моделью
    }


def sma_cross_preds(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> np.ndarray:
    """Предсказания простой стратегии: пересечение SMA fast/slow, вход при пересечении."""
    s_f = df["close"].rolling(fast).mean().to_numpy()
    s_s = df["close"].rolling(slow).mean().to_numpy()
    preds = _zero_preds(len(df))
    pos = np.sign(s_f - s_s)
    for i in range(1, len(df)):
        if pos[i] != pos[i - 1] and not np.isnan(pos[i]):
            preds[i] = int(pos[i])   # вход при пересечении; выход — реверс
    return preds


def bench_sma_cross(df: pd.DataFrame, fast=20, slow=50, fee=0.001, slip=0.0005) -> dict:
    """Бенчмарк 4: пересечение SMA 20/50; сравнение ПО SHARPE (единая шкала)."""
    preds = sma_cross_preds(df, fast, slow)
    trades = simulate_non_overlapping(df, preds, fee=fee, slippage=slip, holding=len(df))
    m = metrics_from_trades(trades) if trades else {"n_trades": 0, "total_return": 0.0,
                                                    "sharpe": 0.0, "max_drawdown": 0.0}
    m["sharpe_pb"] = sharpe_per_bar(returns_from_trades(trades, len(df)),
                                    24 * 365) if trades else 0.0
    return m


def bench_random_timing(df: pd.DataFrame, model_preds: np.ndarray, n_runs: int = 1000,
                        fee=0.001, slip=0.0005, holding: int = 24,
                        atr_target: np.ndarray | None = None, atr_multiplier: float = 2.0,
                        funding_per_8h: np.ndarray | None = None, seed: int = 7) -> dict:
    """Бенчмарк 2: СЛУЧАЙНЫЕ таймстампы + направление в том же соотношении, что у модели.

    Проверяет навык выбора МОМЕНТА входа.
    """
    rng = np.random.default_rng(seed)
    active = model_preds[model_preds != 0]
    k = len(active)
    if k == 0:
        return {"mean": 0.0, "std": 0.0, "distribution": np.array([])}
    long_ratio = float((active == 1).mean())
    results = []
    for _ in range(n_runs):
        preds = _zero_preds(len(df))
        idx = rng.choice(len(df), size=k, replace=False)
        dirs = np.where(rng.random(k) < long_ratio, 1, -1)
        preds[idx] = dirs
        trades = simulate_non_overlapping(df, preds, fee=fee, slippage=slip, holding=holding,
                                          atr_target=atr_target, atr_multiplier=atr_multiplier,
                                          funding_per_8h=funding_per_8h)
        m = metrics_from_trades(trades)
        results.append(m.get("total_return", 0.0))
    arr = np.array(results)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "distribution": arr}


def bench_random_direction(df: pd.DataFrame, model_preds: np.ndarray, n_runs: int = 1000,
                           fee=0.001, slip=0.0005, holding: int = 24,
                           atr_target: np.ndarray | None = None, atr_multiplier: float = 2.0,
                           funding_per_8h: np.ndarray | None = None, seed: int = 11) -> dict:
    """Бенчмарк 3: таймстампы МОДЕЛИ + СЛУЧАЙНОЕ направление.

    Проверяет навык выбора НАПРАВЛЕНИЯ.
    """
    rng = np.random.default_rng(seed)
    mask = model_preds != 0
    k = int(mask.sum())
    if k == 0:
        return {"mean": 0.0, "std": 0.0, "distribution": np.array([])}
    timestamps_idx = np.flatnonzero(mask)
    results = []
    for _ in range(n_runs):
        preds = _zero_preds(len(df))
        preds[timestamps_idx] = rng.choice([1, -1], size=k)
        trades = simulate_non_overlapping(df, preds, fee=fee, slippage=slip, holding=holding,
                                          atr_target=atr_target, atr_multiplier=atr_multiplier,
                                          funding_per_8h=funding_per_8h)
        m = metrics_from_trades(trades)
        results.append(m.get("total_return", 0.0))
    arr = np.array(results)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "distribution": arr}


def interpret(model_return: float, b2: dict, b3: dict, pct: float = 0.95) -> str:
    """Интерпретация: FULL ALPHA / TIMING ONLY / DIRECTION ONLY / NO ALPHA."""
    t_ok = bool(b2["distribution"].size and (b2["distribution"] < model_return).mean() > pct)
    d_ok = bool(b3["distribution"].size and (b3["distribution"] < model_return).mean() > pct)
    if t_ok and d_ok:
        return "FULL ALPHA: модель умеет и момент, и направление"
    if t_ok:
        return "TIMING ONLY: модель выбирает момент, но не направление"
    if d_ok:
        return "DIRECTION ONLY: модель выбирает направление, но не момент"
    return "NO ALPHA: модель не лучше случайности"


def run_all_benchmarks(df: pd.DataFrame, model_preds: np.ndarray, model_return: float,
                       n_runs: int = 1000, fee=0.001, slip=0.0005, holding=24,
                       atr_target: np.ndarray | None = None, atr_multiplier: float = 2.0,
                       funding_per_8h: np.ndarray | None = None) -> dict:
    b2 = bench_random_timing(df, model_preds, n_runs=n_runs, fee=fee, slip=slip, holding=holding,
                             atr_target=atr_target, atr_multiplier=atr_multiplier,
                             funding_per_8h=funding_per_8h)
    b3 = bench_random_direction(df, model_preds, n_runs=n_runs, fee=fee, slip=slip, holding=holding,
                                atr_target=atr_target, atr_multiplier=atr_multiplier,
                                funding_per_8h=funding_per_8h)
    b1 = bench_buyhold(df, fee, slip)
    b4 = bench_sma_cross(df, fee=fee, slip=slip)
    return {
        "buyhold": b1,
        "benchmark_2_timing": {k: v for k, v in b2.items() if k != "distribution"} |
            {"model_percentile": float((b2["distribution"] < model_return).mean() * 100)
             if b2["distribution"].size else 0.0},
        "benchmark_3_direction": {k: v for k, v in b3.items() if k != "distribution"} |
            {"model_percentile": float((b3["distribution"] < model_return).mean() * 100)
             if b3["distribution"].size else 0.0},
        "benchmark_4_sma": b4,
        "interpretation": interpret(model_return, b2, b3),
        "_raw_b2": b2["distribution"], "_raw_b3": b3["distribution"],
    }