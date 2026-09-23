# -*- coding: utf-8 -*-
"""Non-overlapping бэктест + статистическая значимость (разделы 3.9, 0.2).

Правила симуляции Phase 0:
- Вход по сигналу модели на ЗАКРЫТИИ свечи t (следующее исполнение — close t с издержками).
- Одна позиция на пару: пока открыта, новые входы запрещены (non-overlapping).
- Выход: time-based (max_holding баров) либо достижение ATR-барьера (метка уже про
  барьер 2×ATR за 24 бара; для фазы 0 упрощение — выход через forward_period баров
  по close; SL/TP-политика добавляется в фазе 1, раздел 4.6).
- Издержки: тейкер 0.1% вход + 0.1% выход + проскальзывание 0.05%×2 + funding за время удержания.
- Направление симметричное: лонг (+1) и шорт (-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_non_overlapping(df: pd.DataFrame, predictions: np.ndarray,
                             fee: float = 0.001, slippage: float = 0.0005,
                             holding: int = 24, funding_per_8h: np.ndarray | None = None
                             ) -> list[dict]:
    """Симуляция сделок без перекрытия. Возвращает список сделок с pnl_pct."""
    close = df["close"].to_numpy()
    n = len(close)
    trades: list[dict] = []
    i = 0
    while i < n:
        side = predictions[i]
        if side == 0:
            i += 1
            continue
        entry_i = i
        exit_i = min(i + holding, n - 1)
        entry_px = close[entry_i] * (1 + slippage * np.sign(side))
        exit_px = close[exit_i] * (1 - slippage * np.sign(side))
        gross = (exit_px / entry_px - 1.0) * side
        costs = 2 * fee
        # funding: платит лонг при положительном рейте (каждые 8ч → holding/8 начислений)
        fund = 0.0
        if funding_per_8h is not None:
            fr = funding_per_8h[entry_i: exit_i + 1]
            n_charge = max(1, holding // 8)
            fund = np.nanmean(fr) * side * n_charge / 3  # средний рейт × кол-во начислений/3(8ч из 24)
        pnl = gross - costs - fund
        trades.append({
            "entry_idx": entry_i, "exit_idx": exit_i,
            "entry_ts": str(df["timestamp"].iloc[entry_i]),
            "side": int(side), "pnl_pct": float(pnl),
            "win": bool(pnl > 0),
        })
        i = exit_i + 1  # блокировка входа до закрытия позиции
    return trades


def sharpe_from_trades(trade_returns: np.ndarray, periods_per_year: float) -> float:
    if len(trade_returns) < 2 or trade_returns.std() == 0:
        return 0.0
    return float(trade_returns.mean() / trade_returns.std() * np.sqrt(periods_per_year))


def equity_curve(trades: list[dict], start_cash: float = 1.0) -> np.ndarray:
    eq = [start_cash]
    for t in trades:
        eq.append(eq[-1] * (1 + t["pnl_pct"]))
    return np.array(eq)


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak - 1.0).min()
    return float(abs(dd))


def metrics_from_trades(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0}
    r = np.array([t["pnl_pct"] for t in trades])
    # ~1 сделка максимум каждые 24ч → годовых периодов ≈ 365
    sharpe = sharpe_from_trades(r, periods_per_year=365)
    eq = equity_curve(trades)
    total_return = float(eq[-1] / eq[0] - 1)
    win_rate = float(np.mean([t["win"] for t in trades]))
    return {
        "n_trades": len(trades),
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(eq),
        "win_rate": win_rate,
        "avg_pnl": float(r.mean()),
    }


def newey_west_significance(trade_returns: np.ndarray, maxlags: int = 5):
    """t-статистика средней доходности с поправкой на автокорреляцию (раздел 3.9)."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    r = np.asarray(trade_returns, dtype=float)
    X = add_constant(np.zeros((len(r), 1)))  # константная регрессия: return ~ 1
    model = OLS(r, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "t_statistic": float(model.tvalues[0]),
        "p_value": float(model.pvalues[0]),
        "mean_ret": float(r.mean()),
    }


def block_bootstrap_significance(trade_returns: np.ndarray, block_size: int = 10,
                                 n_bootstrap: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    r = np.asarray(trade_returns, dtype=float)
    n_blocks = max(1, len(r) // block_size)
    means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        sample = np.concatenate([r[b * block_size:(b + 1) * block_size] for b in idx])
        means.append(sample.mean())
    means = np.array(means)
    return {
        "mean": float(r.mean()),
        "p_value": float((means <= 0).mean()),
        "ci_95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }
