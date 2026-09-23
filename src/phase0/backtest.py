# -*- coding: utf-8 -*-
"""Non-overlapping бэктест + статистическая значимость (разделы 3.9, 0.2).

Правила симуляции Phase 0:
- Вход по сигналу модели на ЗАКРЫТИИ свечи t (исполнение — close t с издержками).
- Одна позиция на пару: пока открыта, новые входы запрещены (non-overlapping).
- Выход: достижение ATR-барьера (2×ATR, соответствует меткам 3.5) ЛИБО по времени
  (holding_bars), что наступит раньше. Ранее был безусловный выход через 24 бара —
  это расходилось с семантикой метки (пробой 2×ATR).
- Издержки: тейкер 0.1% вход + 0.1% выход + проскальзывание 0.05%×2 + funding.
- Направление симметричное: лонг (+1) и шорт (-1).

Ревизия: добавлены per-bar доходности (единая шкала Sharpe между моделью,
Buy&Hold и SMA — раньше модель/бенчмарки считались по-разному), Information
Ratio и корректные параметры статистики (Newey-West maxlags=24, bootstrap
block_size=24 по разделу 3.9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_non_overlapping(df: pd.DataFrame, predictions: np.ndarray,
                             fee: float = 0.001, slippage: float = 0.0005,
                             holding: int = 24,
                             atr_target: np.ndarray | None = None,
                             atr_multiplier: float = 2.0,
                             funding_per_8h: np.ndarray | None = None
                             ) -> list[dict]:
    """Симуляция сделок без перекрытия. Возвращает список сделок с pnl_pct.

    atr_target: per-bar ATR (той же длины, что df). Если задан и atr_exit —
    позиция закрывается при достижении atr_multiplier×ATR(на входе), иначе
    по времени через `holding` баров.
    funding_per_8h: per-bar эффективная ставка funding (см. align_funding_per_bar).
    """
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
        exit_reason = "time"
        entry_px = close[entry_i] * (1 + slippage * np.sign(side))

        # ранний выход при пробое ATR-барьера (совпадает с метками 3.5)
        if atr_target is not None:
            thr = atr_multiplier * float(atr_target[entry_i])
            for j in range(entry_i + 1, exit_i + 1):
                if side * (close[j] - entry_px) >= thr:
                    exit_i = j
                    exit_reason = "atr"
                    break

        exit_px = close[exit_i] * (1 - slippage * np.sign(side))
        gross = (exit_px / entry_px - 1.0) * side
        costs = 2 * fee

        # funding: лонг платит положительный рейт (каждые 8ч), шорт — наоборот.
        # n_settlements/3 масштабирует средний per-bar рейт на число расчётов.
        fund = 0.0
        if funding_per_8h is not None:
            seg = funding_per_8h[entry_i: exit_i + 1]
            valid = seg[~np.isnan(seg)]
            n_settle = max(1, (exit_i - entry_i + 1) // 8)
            mean_fr = float(np.mean(valid)) if len(valid) else 0.0
            fund = side * mean_fr * (n_settle / 3.0)

        pnl = gross - costs - fund
        trades.append({
            "entry_idx": entry_i, "exit_idx": exit_i,
            "entry_ts": str(df["timestamp"].iloc[entry_i]),
            "exit_ts": str(df["timestamp"].iloc[exit_i]),
            "exit_reason": exit_reason,
            "side": int(side), "pnl_pct": float(pnl),
            "gross_pnl": float(gross), "funding_paid": float(fund),
            "win": bool(pnl > 0),
        })
        i = exit_i + 1  # блокировка входа до закрытия позиции (1 позиция на пару)
    return trades


# ───────────────────────── агрегированные метрики ─────────────────────────

def sharpe_from_trades(trade_returns: np.ndarray, periods_per_year: float) -> float:
    """Sharpe по доходностям сделок (аннуализация на ~сделку/день при 1h-свечах)."""
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
        return {"n_trades": 0, "total_return": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0, "avg_pnl": 0.0}
    r = np.array([t["pnl_pct"] for t in trades])
    sharpe = sharpe_from_trades(r, periods_per_year=365)
    eq = equity_curve(trades)
    return {
        "n_trades": len(trades),
        "total_return": float(eq[-1] / eq[0] - 1),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(eq),
        "win_rate": float(np.mean([t["win"] for t in trades])),
        "avg_pnl": float(r.mean()),
    }


# ───────────────────────── per-bar метрики (единая шкала) ─────────────────────────

def returns_from_trades(trades: list[dict], n_bars: int) -> np.ndarray:
    """Per-bar доходности: pnl сделки записывается в бар выхода, иначе 0."""
    rets = np.zeros(n_bars)
    for t in trades:
        e = min(int(t["exit_idx"]), n_bars - 1)
        rets[e] += float(t["pnl_pct"])
    return rets


def returns_from_buyhold(df: pd.DataFrame, fee: float = 0.001,
                         slippage: float = 0.0005) -> np.ndarray:
    """Per-bar доходности Buy&Hold (1h): лог-доходность close, издержки в первом баре."""
    c = df["close"].to_numpy()
    r = np.zeros(len(c))
    r[1:] = c[1:] / c[:-1] - 1
    r[0] = -(2 * fee + 2 * slippage)
    return r


def equity_from_returns(rets: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + np.asarray(rets, dtype=float))


def max_drawdown_from_returns(rets: np.ndarray) -> float:
    return max_drawdown(equity_from_returns(rets))


def sharpe_per_bar(rets: np.ndarray, periods_per_year: float = 24 * 365) -> float:
    """Sharpe по per-bar доходностям; для 1h периодов/год = 8760."""
    rets = np.asarray(rets, dtype=float)
    std = rets.std()
    if std == 0:
        return 0.0
    return float(rets.mean() / std * np.sqrt(periods_per_year))


def info_ratio(rets_model: np.ndarray, rets_bench: np.ndarray,
               periods_per_year: float = 24 * 365) -> float:
    """Information Ratio = (return_model − return_bench) / tracking_error (3.11 или #1)."""
    d = np.asarray(rets_model, dtype=float) - np.asarray(rets_bench, dtype=float)
    sd = d.std()
    if sd == 0:
        return 0.0
    return float(d.mean() / sd * np.sqrt(periods_per_year))


def funding_by_side(trades: list[dict]) -> dict:
    """Агрегация funding по сторонам (чек-лист п.12: анализ фандинга по сторонам)."""
    def agg(side: int) -> dict:
        g = [t for t in trades if t["side"] == side]
        fp = np.array([t["funding_paid"] for t in g]) if g else np.array([])
        return {
            "n": len(g),
            "total_paid": float(fp.sum()) if len(fp) else 0.0,
            "mean_paid": float(fp.mean()) if len(fp) else 0.0,
        }
    return {"long": agg(1), "short": agg(-1)}


# ───────────────────────── статистика с поправкой (3.9) ─────────────────────────

def newey_west_significance(trade_returns: np.ndarray, maxlags: int = 24):
    """t-статистика средней доходности с поправкой на автокорреляцию.
    maxlags = forward_period (24) по разделу 3.9."""
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


def block_bootstrap_significance(trade_returns: np.ndarray, block_size: int = 24,
                                 n_bootstrap: int = 2000, seed: int = 42):
    """Block bootstrap, block_size = forward_period (24) по разделу 3.9."""
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